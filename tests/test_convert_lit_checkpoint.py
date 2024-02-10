import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import ANY
from urllib.request import urlretrieve

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


def test_convert_lit_checkpoint(tmp_path):
    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import convert_lit_checkpoint

    ours_config = Config.from_name("Llama-2-7b-hf", block_size=8, n_layer=2, n_embd=32, n_head=2, padding_multiple=128)
    ours_model = GPT(ours_config)
    checkpoint_path = tmp_path / "foo.ckpt"
    config_path = tmp_path / "foo.json"
    torch.save(ours_model.state_dict(), checkpoint_path)
    with open(config_path, "w") as fp:
        json.dump(asdict(ours_config), fp)
    output_path = tmp_path / "generated.bin"

    convert_lit_checkpoint(checkpoint_path, output_path, config_path)
    assert set(os.listdir(tmp_path)) == {"foo.ckpt", "foo.json", "generated.bin"}

    # check checkpoint is unwrapped
    torch.save({"model": ours_model.state_dict()}, checkpoint_path)
    convert_lit_checkpoint(checkpoint_path, output_path, config_path)
    converted_sd = torch.load(output_path)
    assert "model" not in converted_sd


@torch.inference_mode()
def test_against_falcon_40b():
    from transformers.models.falcon.configuration_falcon import FalconConfig
    from transformers.models.falcon.modeling_falcon import FalconForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_falcon as copy_to_theirs

    ours_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = FalconConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_hidden_layers=ours_config.n_layer,
        num_attention_heads=ours_config.n_head,
        num_kv_heads=ours_config.n_query_groups,
        new_decoder_architecture=True,
        parallel_attn=ours_config.parallel_residual,
        bias=ours_config.bias,
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_to_theirs("40b", theirs_state_dict, ours_state_dict)

    theirs_model = FalconForCausalLM(theirs_config)
    # assign must be set to True for torch.testing.assert_close to pass
    theirs_model.load_state_dict(theirs_state_dict, assign=True)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_gpt_neox():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_gpt_neox as copy_to_theirs

    ours_config = Config(block_size=64, vocab_size=100, n_layer=4, n_head=8, n_embd=16)
    assert ours_config.padded_vocab_size == 512
    theirs_config = GPTNeoXConfig(
        hidden_act="gelu",
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        initializer_range=0.02,
        intermediate_size=ours_config.intermediate_size,
        layer_norm_eps=1e-05,
        max_position_embeddings=ours_config.block_size,
        rotary_emb_base=10000,
        rotary_pct=ours_config.rotary_percentage,
        vocab_size=ours_config.padded_vocab_size,
        use_parallel_residual=ours_config.parallel_residual,
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_to_theirs(theirs_state_dict, ours_state_dict)
    theirs_model = GPTNeoXForCausalLM(theirs_config)
    # strict=False because we don't save the rotary embeddings inv frequency
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys
    assert all("inv_freq" in k for k in keys.missing_keys)

    # test end to end
    x = torch.randint(0, ours_config.padded_vocab_size, size=(2, ours_config.block_size), dtype=torch.int64)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs", [{"name": "Llama-2-7b-hf"}, {"name": "CodeLlama-7b-hf"}, {"name": "Llama-2-70b-chat-hf"}]
)
def test_against_hf_llama2(ours_kwargs):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_llama

    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, intermediate_size=86, **ours_kwargs
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=1e-5,
        num_query_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = LlamaForCausalLM(theirs_config)
    theirs_model.load_state_dict(theirs_state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_open_llama_3b():
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_llama

    ours_config = Config.from_name("open_llama_3b", n_layer=2, n_head=8, n_embd=32, intermediate_size=86)
    T = 5
    theirs_config = LlamaConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = LlamaForCausalLM(theirs_config)
    theirs_model.load_state_dict(theirs_state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_hf_phi():
    workdir = wd / "tests" / "reference_models"
    workdir.mkdir(parents=True, exist_ok=True)
    file_paths = [workdir / "original_phi_1_5.py", workdir / "configuration_phi.py"]
    urls = [
        "https://huggingface.co/microsoft/phi-1_5/raw/main/modeling_phi.py",
        "https://huggingface.co/microsoft/phi-1_5/raw/main/configuration_phi.py",
    ]
    for file_path, url in zip(file_paths, urls):
        if not file_path.is_file():
            urlretrieve(url=url, filename=file_path)

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_phi
    from tests.reference_models.configuration_phi import PhiConfig
    from tests.reference_models.original_phi_1_5 import PhiForCausalLM

    ours_config = Config.from_name(
        "phi-1_5", padded_vocab_size=10000, n_layer=2, n_head=4, n_embd=256, rotary_percentage=0.5
    )
    T = 5
    theirs_config = PhiConfig(
        n_positions=ours_config.block_size,
        n_embd=ours_config.n_embd,
        n_head=ours_config.n_head,
        n_layer=ours_config.n_layer,
        rotary_dim=ours_config.rope_n_elem,
        architecture={"block_cls": "parallel", "mixer": {}, "mlp": {"mlp_cls": "mlp"}},
    )
    theirs_config.vocab_size = ours_config.padded_vocab_size

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_phi(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = PhiForCausalLM(theirs_config)
    # strict=False because we don't save the rotary embeddings inv frequency
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys
    assert all("inv_freq" in k for k in keys.missing_keys)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


def test_check_conversion_supported_adapter():
    from scripts.convert_lit_checkpoint import check_conversion_supported

    lit_weights = {"some.key.name": ANY, "error.key.gating_factor": ANY}
    with pytest.raises(NotImplementedError, match="Converting adapter"):
        check_conversion_supported(lit_weights=lit_weights)

    lit_weights = {"some.key.name": ANY, "error.key.adapter_bias": ANY}
    with pytest.raises(NotImplementedError, match="Converting adapter"):
        check_conversion_supported(lit_weights=lit_weights)


def test_check_conversion_supported_lora():
    from scripts.convert_lit_checkpoint import check_conversion_supported

    lit_weights = {"some.key.name": ANY, "error.key.lora": ANY}
    with pytest.raises(ValueError, match=r"LoRA.*cannot be converted"):
        check_conversion_supported(lit_weights=lit_weights)


def test_qkv_split():
    from lit_gpt import Config
    from scripts.convert_lit_checkpoint import qkv_split

    # MHA
    config = Config(n_embd=4, n_head=4)
    qkv = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35],
            [36, 37, 38, 39],
            [40, 41, 42, 43],
            [44, 45, 46, 47],
        ]
    )
    q, k, v = qkv_split(qkv, config)
    torch.testing.assert_close(q, torch.tensor([[0, 1, 2, 3], [12, 13, 14, 15], [24, 25, 26, 27], [36, 37, 38, 39]]))
    torch.testing.assert_close(k, torch.tensor([[4, 5, 6, 7], [16, 17, 18, 19], [28, 29, 30, 31], [40, 41, 42, 43]]))
    torch.testing.assert_close(v, torch.tensor([[8, 9, 10, 11], [20, 21, 22, 23], [32, 33, 34, 35], [44, 45, 46, 47]]))

    # GQA
    config = Config(n_embd=4, n_head=4, n_query_groups=2)
    qkv = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
        ]
    )
    q, k, v = qkv_split(qkv, config)
    torch.testing.assert_close(q, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [16, 17, 18, 19], [20, 21, 22, 23]]))
    torch.testing.assert_close(k, torch.tensor([[8, 9, 10, 11], [24, 25, 26, 27]]))
    torch.testing.assert_close(v, torch.tensor([[12, 13, 14, 15], [28, 29, 30, 31]]))

    # MQA
    config = Config(n_embd=4, n_head=4, n_query_groups=1)
    qkv = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    )
    q, k, v = qkv_split(qkv, config)
    torch.testing.assert_close(q, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]))
    torch.testing.assert_close(k, torch.tensor([[16, 17, 18, 19]]))
    torch.testing.assert_close(v, torch.tensor([[20, 21, 22, 23]]))

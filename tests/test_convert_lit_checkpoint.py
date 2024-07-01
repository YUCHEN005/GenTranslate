from pathlib import Path
from unittest import mock
from urllib.request import urlretrieve

import lightning as L
import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


def test_convert_lit_checkpoint(tmp_path):
    from scripts.convert_lit_checkpoint import convert_lit_checkpoint

    ckpt_name = "lit_model.pth"

    with pytest.raises(RuntimeError, match="open file failed because of errno 2 on fopen"):
        convert_lit_checkpoint(checkpoint_name=ckpt_name, out_dir=tmp_path, model_name="falcon-7b")

    ckpt_path = tmp_path / "lit_model.pth"
    ckpt_path.touch()
    with mock.patch("scripts.convert_lit_checkpoint.lazy_load") as load:
        convert_lit_checkpoint(checkpoint_name=ckpt_name, out_dir=tmp_path, model_name="falcon-7b")
    load.assert_called_with(ckpt_path)

    assert {p.name for p in tmp_path.glob("*")} == {"lit_model.pth", "lit_model.bin"}


def test_convert_lit_checkpoint_llama2(tmp_path):
    from finetune.full import save_checkpoint
    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import convert_lit_checkpoint

    # fabric is needed for finetune.full::save_checkpoint
    fabric = L.Fabric(devices=1)

    ckpt_path: Path = tmp_path / "lit_model_finetune.pth"
    ckpt_name = ckpt_path.name

    model_name = "Llama-2-7b-hf"
    ours_config = Config.from_name(model_name, block_size=8, n_layer=2, n_embd=32, n_head=2, padding_multiple=128)
    ours_model = GPT(ours_config)

    # save checkpoint to avoid RunTimeError for PytorchStreamReader
    save_checkpoint(fabric, ours_model, ckpt_path)
    # this should not cause a TypeError
    convert_lit_checkpoint(checkpoint_name=ckpt_name, out_dir=tmp_path, model_name=model_name)


@torch.inference_mode()
def test_against_original_falcon_40b():
    file_path = wd / "tests" / "original_falcon_40b.py"
    url = "https://gist.githubusercontent.com/carmocca/feed39b1bc65a29f73c1cecc58a01167/raw/a9a65f2b93716b3c09ec9f354d535ae5953de08f/original_falcon_40b.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_falcon as copy_to_theirs
    from tests.original_falcon_40b import RWConfig, RWForCausalLM

    ours_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = RWConfig(
        hidden_size=32, n_head=8, n_head_kv=4, n_layer=2, parallel_attn=True, vocab_size=65024, bias=False
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_to_theirs("40b", theirs_state_dict, ours_state_dict)

    theirs_model = RWForCausalLM(theirs_config)
    # assign must be set to True for torch.testing.assert_close to pass
    theirs_model.load_state_dict(theirs_state_dict, strict=False, assign=True)

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

    ours_config = Config.from_name("pythia-1b", block_size=2048, n_layer=2, n_embd=2048, n_head=8, padding_multiple=128)
    theirs_config = GPTNeoXConfig(
        hidden_size=ours_config.n_embd,
        intermediate_size=ours_config.intermediate_size,
        num_hidden_layers=ours_config.n_layer,
        num_attention_heads=ours_config.n_head,
        n_head_kv=ours_config.n_query_groups,
        vocab_size=ours_config.padded_vocab_size,
        bias=ours_config.bias,
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_to_theirs(theirs_state_dict, ours_state_dict)

    theirs_model = GPTNeoXForCausalLM(theirs_config)
    # assign must be set to True for torch.testing.assert_close to pass
    theirs_model.load_state_dict(theirs_state_dict, strict=False, assign=True)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("size", ("7b", "70b"))
def test_against_original_llama2(size):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import copy_weights_llama as copy_to_theirs

    if size == "7b":
        ours_kwargs = {"name": "Llama-2-7b-hf"}
        theirs_kwargs = {}
    else:
        ours_kwargs = {"name": "Llama-2-70b-chat-hf", "n_query_groups": 2}
        theirs_kwargs = {"num_key_value_heads": 2}

    ours_config = Config.from_name(n_layer=2, n_head=8, n_embd=32, intermediate_size=86, **ours_kwargs)
    T = 5
    theirs_config = LlamaConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=1e-5,
        **theirs_kwargs,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_to_theirs(ours_config, theirs_state_dict, ours_state_dict)

    theirs_model = LlamaForCausalLM(theirs_config)
    # assign must be set to True for torch.testing.assert_close to pass
    theirs_model.load_state_dict(theirs_state_dict, strict=False, assign=True)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


def test_maybe_unwrap_state_dict(tmp_path):
    from finetune.full import save_checkpoint
    from lit_gpt import GPT, Config
    from scripts.convert_lit_checkpoint import convert_lit_checkpoint

    # fabric is needed for finetune.full::save_checkpoint
    fabric = L.Fabric(devices=1)

    ckpt_path: Path = tmp_path / "lit_model_finetune.pth"
    ckpt_name = ckpt_path.name

    model_name = "pythia-70m"
    ours_config = Config.from_name(model_name, block_size=8, n_layer=2, n_embd=32, n_head=2, padding_multiple=128)
    ours_model = GPT(ours_config)

    # save checkpoint and check for model key
    save_checkpoint(fabric, ours_model, ckpt_path)
    statedict_with_model_key = torch.load(ckpt_path)
    assert statedict_with_model_key.get("model")
    assert len(statedict_with_model_key) == 1

    # convert and check that model key does not exist
    # and that a known key for pythia exists
    convert_lit_checkpoint(checkpoint_name=ckpt_name, out_dir=tmp_path, model_name=model_name)
    bin_file = ckpt_path.with_suffix(".bin")
    ckpt_from_unwrapped = torch.load(bin_file)
    assert ckpt_from_unwrapped.get("model") is None
    assert ckpt_from_unwrapped.get("embed_out.weight") is not None

    # assert maybe_unwrap_state_dict is called
    with mock.patch("scripts.convert_lit_checkpoint.maybe_unwrap_state_dict") as maybe_unwrap:
        convert_lit_checkpoint(checkpoint_name=ckpt_name, out_dir=tmp_path, model_name=model_name)
    maybe_unwrap.assert_called()


def test_check_conversion_supported_adapter():
    from scripts.convert_lit_checkpoint import check_conversion_supported

    lit_weights = {"some.key.name": "some.key.value", "error.key.gating_factor": "some.key.value"}

    with pytest.raises(NotImplementedError, match="Converting models finetuned with adapter *"):
        check_conversion_supported(lit_weights=lit_weights)


def test_check_conversion_supported_adapter_v2():
    from scripts.convert_lit_checkpoint import check_conversion_supported

    lit_weights = {"some.key.name": "some.key.value", "error.key.adapter_bias": "some.key.value"}

    with pytest.raises(NotImplementedError, match="Converting models finetuned with adapter_v2"):
        check_conversion_supported(lit_weights=lit_weights)


def test_check_conversion_supported_lora():
    from scripts.convert_lit_checkpoint import check_conversion_supported

    lit_weights = {"some.key.name": "some.key.value", "error.key.lora": "some.key.value"}

    with pytest.raises(ValueError, match=r"Model weights must be merged using"):
        check_conversion_supported(lit_weights=lit_weights)

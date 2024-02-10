import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_full_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    import finetune.full as module

    module.gradient_accumulation_iters = 1
    module.save_interval = 2
    module.eval_interval = 2
    module.eval_iters = 2
    module.eval_max_new_tokens = 1
    module.max_iters = 6

    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "train.pt")
    torch.save(data, tmp_path / "test.pt")

    from lit_gpt.config import name_to_config

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    monkeypatch.setitem(name_to_config, "tmp", model_config)
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **kwargs: torch.tensor([3, 2, 1], **kwargs)
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(data_dir=tmp_path, checkpoint_dir=fake_checkpoint_dir, out_dir=tmp_path, precision="32-true")

    assert {p.name for p in tmp_path.glob("*.pth")} == {
        "iter-000002-ckpt.pth",
        "iter-000004-ckpt.pth",
        "iter-000006-ckpt.pth",
        "lit_model_finetuned.pth",
    }
    assert (tmp_path / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == module.max_iters
    assert logs.count("val loss") == module.max_iters // module.eval_interval
    assert "of trainable parameters: 1,888" in logs


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_pretrain_tiny_llama(tmp_path, fake_checkpoint_dir, monkeypatch):
    import pretrain.tinyllama as module

    module.save_step_interval = 1
    module.eval_step_interval = 1
    module.log_step_interval = 1
    module.log_iter_interval = 1
    module.eval_iters = 2
    module.max_iters = 3
    module.devices = 1
    module.global_batch_size = 1
    module.micro_batch_size = 1
    module.batch_size = 1
    module.gradient_accumulation_steps = 1
    module.model_name = "tmp"
    module.out_dir = tmp_path

    # Patch torch.compile, because need torch nightly, otherwise we get
    # AssertionError: expected size 4==4, stride 2==4 at dim=1
    monkeypatch.setattr(module.torch, "compile", lambda x: x)

    from lit_gpt.config import name_to_config

    model_config = dict(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    monkeypatch.setitem(name_to_config, "tmp", model_config)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    module.create_dataloaders = Mock(return_value=(dataloader, dataloader))

    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup()

    assert {p.name for p in tmp_path.glob("*.pth")} == {"step-00000001.pth", "step-00000002.pth", "step-00000003.pth"}

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == module.max_iters
    assert logs.count("val loss") == module.max_iters
    assert "Total parameters: 1,888" in logs

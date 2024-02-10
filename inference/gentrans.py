import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.gentrans import GPT, Block, Config
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
parser.add_argument('--dataset', type=str)
parser.add_argument('--srclang', type=str)
parser.add_argument('--tgtlang', type=str)
parser.add_argument('--task', type=str, default='st')
parser.add_argument('--llmsize', type=str)
args = parser.parse_args()

import sacrebleu
bleu_metric = sacrebleu.BLEU(tokenize='13a')

devices = args.d
dataset = args.dataset
srclang = args.srclang
tgtlang = args.tgtlang
task = args.task
llmsize = args.llmsize

### Customized paths
exp_dir = ''        # experiment directory
data_dir = ''       # test data directory
llm_dir = ''        # llm checkpoint directory
adapter_dir = ''    # adapter checkpoint directory

exp_path = f'{exp_dir}/runs/gentrans_{dataset}_{srclang}_{tgtlang}_{task}'
sl = f'{exp_path}/predictions'  # place to save predictions

data_path = f'{data_dir}/test_{dataset}_{srclang}_{tgtlang}_{task}.pt'

precision = None
quantize = None
strategy: str = "auto"
torch.set_float32_matmul_precision("high")

precision = precision or get_default_supported_precision(training=False)
fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
fabric.launch()

checkpoint_dir = Path(f'{llm_dir}/Llama-2-{llmsize}-hf')
check_valid_checkpoint_dir(checkpoint_dir)

with open(checkpoint_dir / "lit_config.json") as fp:
    config = Config(**json.load(fp))

checkpoint_path = checkpoint_dir / "lit_model.pth"

with fabric.init_module(empty_init=True), quantization(quantize):
    model = GPT(config)

tokenizer = Tokenizer(checkpoint_dir)
data = torch.load(data_path)


def result(adapter_path, model):
    # LOADING CORRESPOINDG ADAPTER MODEL
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)

    model.eval()
    model = fabric.setup(model)

    return_dict = {}
    pr, gt = [], []
    to_json = []
    for datapoint in data:
        encoded = datapoint['input_ids_no_response'].to(model.device)
        ground_truth = datapoint['ground_truth']

        max_returned_tokens = encoded.size(0) + 150

        y = generate(
            model=model,
            idx=encoded,
            max_returned_tokens=max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=0.2,
            top_k=1,
            eos_id=tokenizer.eos_id
        )

        model.reset_cache()
        output = tokenizer.decode(y)

        inf = output[len(tokenizer.decode(encoded)):].split('\n')[0].strip()
        ref = ground_truth.strip()

        pr.append(inf)
        gt.append(ref)
        to_json.append({'inference': inf, 'ground_truth': ref})

    return_dict['adapter_path'] = adapter_path.split('/')[-1]

    # BLEU score
    bleu_score = bleu_metric.corpus_score(pr, [gt]).score
    print(f'BLEU is {bleu_score:.2f}')
    return_dict['BLEU'] = bleu_score

    print('*********************')

    to_json.append({'BLEU': bleu_score})

    os.system(f'mkdir -p {sl}')
    with open(os.path.join(sl, adapter_path.split('/')[-1].split('.pth')[0] + '.json'), 'w') as f:
        f.write(json.dumps(to_json, indent=4, ensure_ascii=False))
    print(os.path.join(sl, adapter_path.split('/')[-2] + '.json'))

    return return_dict


adapter_path = f'{adapter_dir}/{dataset}_{srclang}_{tgtlang}_{task}.pth'

result_dict = result(adapter_path, model)
print(result_dict)


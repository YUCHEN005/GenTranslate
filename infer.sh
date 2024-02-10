#!/usr/bin/env bash

source activate <your-conda-env>

### Configurations: 1. fleurs fr-en st 7b, 2. fleurs en-fr st 13b.
dataset=fleurs
srclang=fr
tgtlang=en
task=st
llmsize=7b

$cmd log/infer_gentrans_${dataset}_${srclang}_${tgtlang}_${task}.log \
python inference/gentrans.py --dataset ${dataset} --srclang ${srclang} --tgtlang ${tgtlang} --task ${task} --llmsize ${llmsize}


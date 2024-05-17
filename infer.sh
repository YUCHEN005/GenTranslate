#!/usr/bin/env bash

source activate <your-conda-env>

### Note: "Llama-2-7b-hf" for x-en, and "Llama-2-13b-hf" for en-x;
dataset=fleurs
srclang=fr
tgtlang=en
task=st
seamless_size=large
data_dir=<your-data-directory>
llm_dir=<your-llama-directory>
adapter_path=<your-adapter-ckpt>

python inference/gentrans.py \
        --dataset ${dataset} --srclang ${srclang} --tgtlang ${tgtlang} --task ${task} \
        --seamless_size ${seamless_size} --data_dir ${data_dir} --llm_dir ${llm_dir} \
        --adapter_path ${adapter_path}

# GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators

[[Paper]](https://arxiv.org/abs/2402.06894) [[Data]](https://huggingface.co/datasets/PeacefulData/HypoTranslate) [[Model]](https://huggingface.co/PeacefulData/GenTranslate) [![Language](https://img.shields.io/badge/Language-multilingual-lightgrey#model-badge)](#datasets) | ACL 2024, Oral

<p align="center">  <img src="https://github.com/YUCHEN005/GenTranslate/blob/master/tutorials/gentranslate.png" height ="300"> </p>

This work proposes a generative paradigm for translation tasks that leverages LLMs to generate higher-quality translation results based on the N-best hypotheses decoded from foundation model (e.g., SeamlessM4T-Large-V2).
We also release a HypoTranslate dataset to support LLM finetuning, which contains over 592K pairs of N-best hypotheses and ground-truth translation in 11 languages.
Experiments show that our GenTranslate significantly outperforms the state-of-the-art SeamlessM4T-Large-V2 on various speech and machine translation benchmarks.

## Conda Environment Configuration

Our code is built based on [lit-gpt](https://github.com/Lightning-AI/lit-gpt), please refer to [official tutorial](https://github.com/Lightning-AI/lit-gpt#setup) to build the conda environment. Then, please install the required packages using following command:
```bash
pip install -r requirements.txt
```

## Models and Checkpoints

- For LLMs, please refer to [tutorial](https://github.com/YUCHEN005/GenTranslate/tree/master/tutorials) for download and conversion, which support many mainstream LLMs like [LLaMA-2](https://github.com/YUCHEN005/GenTranslate/blob/master/tutorials/download_llama_2.md) (we use `Llama-2-7b-hf` and `Llama-2-13b-hf` in this work);
- For well-trained adapter checkpoints, please refer to our [HuggingFace repo](https://huggingface.co/PeacefulData/GenTranslate).

## Dataset

We have released our HypoTranslate dataset at [HuggingFace](https://huggingface.co/datasets/PeacefulData/HypoTranslate).

## Finetune
We provide a finetuning script `finetune.sh`, please first enter it and specify some settings:
- `<your-conda-env>`: your conda environment name;
- `dataset`: training data source;
- `srclang`: source language code;
- `tgtlang`: target language code;
- `task`: task id (options: `st`, `mt`);
- `seamless_size`: SeamlessM4T size (options: `large`);
- `data_dir`: data directory where the `.pt` files are put in;
- `llm_dir`: llama checkpoint directory (options: `Llama-2-7b-hf`, `Llama-2-13b-hf`);

**NOTE:** please use `Llama-2-7b-hf` for x-en, and `Llama-2-13b-hf` for en-x;

Then, you can start finetuning by command:

```bash
bash finetune.sh
```

The trained adapter weights will be saved in `runs/gentrans_{dataset}_{srclang}_{tgtlang}_{task}_{seamless_size}/`.

## Inference
We provide an inference script `infer.sh`, please first enter it and specify some settings:
- `<your-conda-env>`: your conda environment name;
- `dataset`: test data source;
- `srclang`: source language code;
- `tgtlang`: target language code;
- `task`: task id (options: `st`, `mt`);
- `seamless_size`: SeamlessM4T size (options: `large`, `largev2`);
- `data_dir`: data directory where the `.pt` files are put in;
- `llm_dir`: llama checkpoint directory (options: `Llama-2-7b-hf`, `Llama-2-13b-hf`);
- `adapter_path`: path of well-trained adapter checkpoint (`.pth` file);

**NOTE:** please use `Llama-2-7b-hf` for x-en, and `Llama-2-13b-hf` for en-x;

Now, you can run inference on your specified language pair by:
```bash
bash infer.sh
```

You will see the BLEU results of GenTranslate on your specified test set.


## References
If you consider this work would be related or useful for your research, please kindly consider to cite the work below. Thank you.

```bib
@inproceedings{hu2024gentranslate,
    title = "GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators",
    author = "Hu, Yuchen and Chen, Chen and Yang, Chao-Han Huck and Li, Ruizhe and Zhang, Dong and Chen, Zhehuai and Chng, Eng Siong",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    publisher = "Association for Computational Linguistics",
    year = "2024"
}
```

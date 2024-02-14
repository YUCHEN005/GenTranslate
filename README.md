# GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators

[[Paper]](https://arxiv.org/abs/2402.06894) [[Data]](https://huggingface.co/datasets/PeacefulData/HypoTranslate) [[Model]](https://huggingface.co/PeacefulData/GenTranslate)

<p align="center">  <img src="https://github.com/YUCHEN005/GenTranslate/blob/master/tutorials/genst-iclr24.png" height ="450"> </p>

This work proposes a generative paradigm for translation tasks that leverages LLMs to generate higher-quality translation results based on the N-best hypotheses decoded from foundation model (e.g., SeamlessM4T-Large-V2).
We also release a HypoTranslate dataset to support LLM finetuning, which contains over 592K pairs of N-best hypotheses and ground-truth translation in 11 languages.
Experiments show that our GenTranslate significantly outperforms the state-of-the-art SeamlessM4T-Large-V2 on various speech and machine translation benchmarks.

**TIP:** At this time (before publication), we provide inference script, test data and partial well-trained models only for inference use. Full-version resources of this paper, including training script, the entire HypoTranslate dataset and all the models, will be open sourced upon publication to benefit the community.

## Conda Environment Configuration

Our code is built based on [lit-gpt](https://github.com/Lightning-AI/lit-gpt), please refer to [official tutorial](https://github.com/Lightning-AI/lit-gpt#setup) to build the conda environment. Then, please install the required packages using following command:
```bash
pip install -r requirements.txt
```

## Code

- Model code: `lit_gpt/gentrans.py`;
- Inference script: `infer.sh`;

## Models

- For LLMs, please refer to [tutorial](https://github.com/Lightning-AI/lit-gpt/tree/main/tutorials) for configuration steps, which support many mainstream LLMs like [LLaMA-2](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_llama_2.md);
- For well-trained adapter checkpoints, please refer to our [HuggingFace repo](https://huggingface.co/PeacefulData/GenTranslate).

## Dataset

We have released our HypoTranslate dataset at [HuggingFace](https://huggingface.co/datasets/PeacefulData/HypoTranslate).


## Inference Usage
We provide two well-trained models and corresponding test sets for inference use, i.e., FLEURS Fr-En and En-Fr ST tasks.
Before running inference, please follow the steps below for preparation:
1. Go to `infer.sh`:
   - Specify you conda environment `<your-conda-env>`;
   - Specify the source-target language pair, where we provide two example pairs `fr-en` and `en-fr`;
   - Specify the LLM size: `7b` for `fr-en`, `13b` for `en-fr`;
2. Download and convert LLaMA-2 pre-trained checkpoint:
   - Please refer to [official tutorial](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_llama_2.md) to configure `Llama-2-7b-hf` and `Llama-2-13b-hf`;
3. Go to `inference/gentrans.py`:
   - Specify the experiment directory `exp_dir`: the root path of this README.md file;
   - Specify the data directory `data_dir`: the absolute path of test data (`.pt` file);
   - Specify the LLM directory `llm_dir`: the absolute path of your downloaded LLaMA-2 checkpoint;
   - Specify the adapter directory `adapter_dir`: the absolute path of our released adapter checkpoint;

Now you can run inference on your specified language direction by:
```bash
bash infer.sh
```

You will see the BLEU results of GenTranslate on your specified test set.


## References
```bib
@article{hu2024gentranslate,
  title={GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators},
  author={Hu, Yuchen and Chen, Chen and Yang, Chao-Han Huck and Li, Ruizhe and Zhang, Dong and Chen, Zhehuai and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2402.06894},
  year={2024}
}
```

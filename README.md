<div align="center">
<img src="docs/bandai.png" alt="Logo" width="500">
</div>

<h1 align="center">Thinking with Programming Vision: Towards a Unified View for Thinking with Images</h1>


<div align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2512.03746" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
</div>

## Overview

- **Introduction**: A framework leveraging code-as-tool and comprehensive SFT/RL datasets for "thinking with images".

- **Features**: Supports multi-turn agent loops for the Qwen2.5-VL and Qwen3-VL series.

- **Datasets**: Includes an SFT dataset constructed using GPT-5-High and an RL dataset covering diverse domains.


<div align="center">
<img src="docs/adv.png" alt="Overview" width="350">
</div>


## Getting Started

### Environment Setup

For RL training:

```bash
pip install "torch==2.8.0" "torchvision==0.23.0"

# vllm >= 0.11.0 or sglang >= 0.5.3 for Qwen3-VL series support
# Pick one stack: vLLM OR SGLang (install the one you need)
pip install vllm==0.11.0          # option 1: vLLM stack
pip install "sglang[all]==0.5.3"  # option 2: SGLang stack

# transformers >= 4.57.0 for Qwen3-VL series support
pip3 install transformers==4.57.0

# FlashAttention
pip install --no-cache-dir --use-pep517 flash-attn==2.8.3 --no-build-isolation

# Other dependencies
pip install -r requirements-runtime.txt
```


### Running

#### Stage 1: SFT

The construction pipeline of the SFT dataset:

<div align="center">
<img src="docs/sftdata.png" alt="sftdata" width="500">
</div>


First, download the [CodeVision-SFT Dataset](https://huggingface.co/datasets/kkwok/CodeVision-SFT) for SFT. We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for our SFT training. Update the config file in [LLaMA-Factory/examples/train_full/qwen2_5vl_full_sft.yaml](LLaMA-Factory/examples/train_full/qwen2_5vl_full_sft.yaml) and [LLaMA-Factory/examples/train_full/qwen3vl.yaml](LLaMA-Factory/examples/train_full/qwen3vl.yaml). Update the data file path in [dataset_info.json](LLaMA-Factory/data/dataset_info.json). Then run the following command to launch the training:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen3vl.yaml
```

#### Stage 2: RL

Waiting for internal approval... Coming soon...



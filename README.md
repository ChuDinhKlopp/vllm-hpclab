## About
This fork implements **expert-granularity offloading** and **partial experts prefetching** for vLLM. For **expert predictor** testing, refer to `gpt_oss.py`, where the logic is implemented and used.

## Supported Models
This repository currently supports the following **GPT-OSS** models:
- **gpt-oss-20b**
- **gpt-oss-120b**

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ChuDinhKlopp/vllm-hpclab.git
cd vllm-hpclab
```

### 2. Create a virtual env with uv
```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

### 3. Set up using Python-only build (without compilation)
```bash
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

## Run the project
Use vllm serve with the local model path and --enforce-eager:
```bash
vllm serve /path/to/gpt-oss-120b --enforce-eager
```

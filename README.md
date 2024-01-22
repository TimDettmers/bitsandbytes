# bitsandbytes

The bitsandbytes is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and quantization functions.



Resources:
- [8-bit Optimizer Paper](https://arxiv.org/abs/2110.02861) --  [Video](https://www.youtube.com/watch?v=IxrlHAJtqKE) -- [Docs](https://bitsandbytes.readthedocs.io/en/latest/)

- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339) -- [LLM.int8() Software Blog Post](https://huggingface.co/blog/hf-bitsandbytes-integration) -- [LLM.int8() Emergent Features Blog Post](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)

## TL;DR
**Requirements**
Python >=3.8. Linux distribution (Ubuntu, MacOS, etc.) + CUDA > 10.0.

(Deprecated: CUDA 10.0 is deprecated and only CUDA >= 11.0) will be supported with release 0.39.0)

**Installation**:

``pip install bitsandbytes``

In some cases it can happen that you need to compile from source. If this happens please consider submitting a bug report with `python -m bitsandbytes` information. What now follows is some short instructions which might work out of the box if `nvcc` is installed. If these do not work see further below.

Compilation quickstart:

```bash
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
```

For CUDA
```bash
# CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
# make argument in {cuda110, cuda11x, cuda12x}
# if you do not know what CUDA you have, try looking at the output of: python -m bitsandbytes
CUDA_VERSION=117 make cuda11x
python setup.py install
```

For ROCm
```bash
# Requiers ROCm 5.6+
# Check if your GPU supports Wave32 with rocminfo | grep "Wavefront Size"
# If this doesn't output 32 and instead 64 this library won't work

# Your ROCm target can be found with rocminfo | grep gfx
ROCM_TARGET=gfx1030 make hip
pip install .
```

**Using Int8 inference with HuggingFace Transformers**

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
  'decapoda-research/llama-7b-hf',
  device_map='auto',
  load_in_8bit=True,
  max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')
```

A more detailed example, can be found in [examples/int8_inference_huggingface.py](examples/int8_inference_huggingface.py).

**Using 8-bit optimizer**:
1. Comment out optimizer: ``#torch.optim.Adam(....)``
2. Add 8-bit optimizer of your choice ``bnb.optim.Adam8bit(....)`` (arguments stay the same)
3. Replace embedding layer if necessary: ``torch.nn.Embedding(..) -> bnb.nn.Embedding(..)``


**Using 8-bit Inference**:
1. Comment out torch.nn.Linear: ``#linear = torch.nn.Linear(...)``
2. Add bnb 8-bit linear light module: ``linear = bnb.nn.Linear8bitLt(...)`` (base arguments stay the same)
3. There are two modes:
   - Mixed 8-bit training with 16-bit main weights. Pass the argument ``has_fp16_weights=True`` (default)
   - Int8 inference. Pass the argument ``has_fp16_weights=False``
4. To use the full LLM.int8() method, use the ``threshold=k`` argument. We recommend ``k=6.0``.
```python
# LLM.int8()
linear = bnb.nn.Linear8bitLt(dim1, dim2, bias=True, has_fp16_weights=False, threshold=6.0)
# inputs need to be fp16
out = linear(x.to(torch.float16))
```


## Features
- 8-bit Matrix multiplication with mixed precision decomposition
- LLM.int8() inference
- 8-bit Optimizers: Adam, AdamW, RMSProp, LARS, LAMB, Lion (saves 75% memory)
- Stable Embedding Layer: Improved stability through better initialization, and normalization
- 8-bit quantization: Quantile, Linear, and Dynamic quantization
- Fast quantile estimation: Up to 100x faster than other algorithms

## Requirements & Installation

Requirements: anaconda, cudatoolkit, pytorch

Hardware requirements:
 - LLM.int8(): NVIDIA Turing (RTX 20xx; T4) or Ampere GPU (RTX 30xx; A4-A100); (a GPU from 2018 or newer).
 - 8-bit optimizers and quantization: NVIDIA Kepler GPU or newer (>=GTX 78X).

Supported CUDA versions: 10.2 - 12.0

The bitsandbytes library is currently only supported on Linux distributions. Windows is not supported at the moment.

The requirements can best be fulfilled by installing pytorch via anaconda. You can install PyTorch by following the ["Get Started"](https://pytorch.org/get-started/locally/) instructions on the official website.

To install run:

``pip install bitsandbytes``

## Using bitsandbytes

### Using Int8 Matrix Multiplication

For straight Int8 matrix multiplication with mixed precision decomposition you can use ``bnb.matmul(...)``. To enable mixed precision decomposition, use the threshold parameter:
```python
bnb.matmul(..., threshold=6.0)
```

For instructions how to use LLM.int8() inference layers in your own code, see the TL;DR above or for extended instruction see [this blog post](https://huggingface.co/blog/hf-bitsandbytes-integration).

### Using the 8-bit Optimizers

With bitsandbytes 8-bit optimizers can be used by changing a single line of code in your codebase. For NLP models we recommend also to use the StableEmbedding layers (see below) which improves results and helps with stable 8-bit optimization.  To get started with 8-bit optimizers, it is sufficient to replace your old optimizer with the 8-bit optimizer in the following way:
```python
import bitsandbytes as bnb

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # comment out old optimizer
adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) # equivalent


torch.nn.Embedding(...) ->  bnb.nn.StableEmbedding(...) # recommended for NLP models
```

Note that by default all parameter tensors with less than 4096 elements are kept at 32-bit even if you initialize those parameters with 8-bit optimizers. This is done since such small tensors do not save much memory and often contain highly variable parameters (biases) or parameters that require high precision (batch norm, layer norm). You can change this behavior like so:
```python
# parameter tensors with less than 16384 values are optimized in 32-bit
# it is recommended to use multiplies of 4096
adam = bnb.optim.Adam8bit(model.parameters(), min_8bit_size=16384)
```

### Change Bits and other Hyperparameters for Individual Parameters

If you want to optimize some unstable parameters with 32-bit Adam and others with 8-bit Adam, you can use the `GlobalOptimManager`. With this, we can also configure specific hyperparameters for particular layers, such as embedding layers. To do that, we need two things: (1) register the parameter while they are still on the CPU, (2) override the config with the new desired hyperparameters (anytime, anywhere). See our [guide](howto_config_override.md) for more details

### Fairseq Users

To use the Stable Embedding Layer, override the respective `build_embedding(...)` function of your model. Make sure to also use the `--no-scale-embedding` flag to disable scaling of the word embedding layer (nor replaced with layer norm). You can use the optimizers by replacing the optimizer in the respective file (`adam.py` etc.).

## Release and Feature History

For upcoming features and changes and full history see [Patch Notes](CHANGELOG.md).

## Errors

1. RuntimeError: CUDA error: no kernel image is available for execution on the device. [Solution](errors_and_solutions.md#No-kernel-image-available)
2. __fatbinwrap_.. [Solution](errors_and_solutions.md#fatbinwrap_)

## Compile from source
To compile from source, you need an installation of CUDA. If `nvcc` is not installed, you can install the CUDA Toolkit with nvcc through the following commands.

```bash
wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/install_cuda.sh
# Syntax cuda_install CUDA_VERSION INSTALL_PREFIX EXPORT_TO_BASH
#   CUDA_VERSION in {110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122}
#   EXPORT_TO_BASH in {0, 1} with 0=False and 1=True 

# For example, the following installs CUDA 11.7 to ~/local/cuda-11.7 and exports the path to your .bashrc
bash install_cuda.sh 117 ~/local 1 
```

To use a specific CUDA version just for a single compile run, you can set the variable `CUDA_HOME`, for example the following command compiles `libbitsandbytes_cuda117.so` using compiler flags for cuda11x with the cuda version at `~/local/cuda-11.7`:

``CUDA_HOME=~/local/cuda-11.7 CUDA_VERSION=117 make cuda11x``

For more detailed instruction, please follow the [compile_from_source.md](compile_from_source.md) instructions.

## License

The majority of bitsandbytes is licensed under MIT, however portions of the project are available under separate license terms: Pytorch is licensed under the BSD license.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.

## How to cite us
If you found this library and found LLM.int8() useful, please consider citing our work:

```bibtex
@article{dettmers2022llmint8,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

For 8-bit optimizers or quantization routines, please consider citing the following work:

```bibtex
@article{dettmers2022optimizers,
  title={8-bit Optimizers via Block-wise Quantization},
  author={Dettmers, Tim and Lewis, Mike and Shleifer, Sam and Zettlemoyer, Luke},
  journal={9th International Conference on Learning Representations, ICLR},
  year={2022}
}
```

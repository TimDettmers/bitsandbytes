# Compiling from source

Basic steps.
1. `CUDA_VERSION=XXX make [target]` where `[target]` is among `cuda92, cuda10x, cuda110, cuda11x, cuda12x, cpuonly`
2. `python setup.py install`

To run these steps you will need to have the nvcc compiler installed that comes with a CUDA installation. If you use anaconda (recommended) then you can figure out which version of CUDA you are using with PyTorch via the command `conda list | grep cudatoolkit`. Then you can install the nvcc compiler by downloading and installing the same CUDA version from the [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive).

You can install CUDA locally without sudo by following the following steps:

```bash
wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/install_cuda.sh
# Syntax cuda_install CUDA_VERSION INSTALL_PREFIX EXPORT_TO_BASH
#   CUDA_VERSION in {110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121}
#   EXPORT_TO_BASH in {0, 1} with 0=False and 1=True 

# For example, the following installs CUDA 11.7 to ~/local/cuda-11.7 and exports the path to your .bashrc
bash install_cuda.sh 117 ~/local 1 
```

By default, the Makefile will look at your `CUDA_HOME` environmental variable to find your CUDA version for compiling the library. If this path is not set it is inferred from the path of your `nvcc` compiler.

Either `nvcc` needs to be in path for the `CUDA_HOME` variable needs to be set to the CUDA directory root (e.g. `/usr/local/cuda`) in order for compilation to succeed

If you type `nvcc` and it cannot be found, you might need to add to your path or set the CUDA_HOME variable. You can run `python -m bitsandbytes` to find the path to CUDA. For example if `python -m bitsandbytes` shows you the following:
```
++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++
/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudart.so
```
You can set `CUDA_HOME` to `/usr/local/cuda-11.7`. For example, you might be able to compile like this.

``CUDA_HOME=~/local/cuda-11.7 CUDA_VERSION=117 make cuda11x``


If you have problems compiling the library with these instructions from source, please open an issue.

## Compilation with Kepler

Since 0.39.1 bitsandbytes installed via pip no longer provides Kepler binaries and these need to be compiled from source. Follow the steps above and instead of `cuda11x_nomatmul` etc use `cuda11x_nomatmul_kepler`

## Compilation with ROCm

Since this library requires hipblasLt this only supports **ROCm 5.6+**.
Works well with these docker images:
- [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch)
- [rocm/pytorch-nightly](https://hub.docker.com/r/rocm/pytorch-nightly).

For installation do:
```bash
make hip ROCM_TARGET=gfx1030
pip install .
```
see https://www.llvm.org/docs/AMDGPUUsage.html#processors for finding ROCM_TARGET (e.g. gfx1030 for 6800XT,6900XT) or do `rocminfo | grep gfx`.
import ctypes as ct
from warnings import warn

import torch

from bitsandbytes.cuda_setup.main import CUDASetup

setup = CUDASetup.get_instance()
if setup.initialized != True:
    setup.run_cuda_setup()

lib = setup.lib
try:
    if lib is None and torch.cuda.is_available():
        CUDASetup.get_instance().generate_instructions()
        CUDASetup.get_instance().print_log_stack()
        raise RuntimeError('''
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues''')
    _ = lib.cadam32bit_grad_fp32  # runs on an error if the library could not be found -> COMPILED_WITH_CUDA=False
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    lib.cget_managed_ptr.restype = ct.c_void_p
    try:
        lib.initCudaLibs()
    except Exception:
        # ignore
        pass
    COMPILED_WITH_CUDA = True
except AttributeError as ex:
    warn("The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.")
    COMPILED_WITH_CUDA = False
    print(str(ex))


# print the setup details after checking for errors so we do not print twice
#if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
    #setup.print_log_stack()

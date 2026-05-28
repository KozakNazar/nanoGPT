import os
import sys
import copy
import time
import pickle
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import tiktoken

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(
    os.path.join(script_dir, "nvcc_tmp"),
    exist_ok=True
)

# Знаходимо шлях до батьківської папки (кореня)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#old_sys_path = sys.path.copy() ## sys.path[:] = old_sys_path #restore

sys.path.append(parent_dir)
from model import GPTConfig, GPT

#
#print(old_sys_path)
#print(sys.path)
#
#sys.path.append(parent_dir)

#os.makedirs("nvcc_tmp", exist_ok=True)
#exit()

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
#temperature = 0.001  #! only for torch.float32 and torch.bfloat16
temperature = 0.01  #!
#top_k = 0  #!
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
#compile = True
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
#start = start * 10 #!

#torch.backends.cudnn.deterministic = True #!
#torch.backends.cudnn.benchmark = False #!
#
torch.use_deterministic_algorithms(True, warn_only=True) #!
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ptdtype = torch.float16 #!
ptdtype = torch.float32 #!
#ptdtype = torch.bfloat16 #!
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
#ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype, enabled=False)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

#os.makedirs("nvcc_tmp", exist_ok=True)

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

#nvcc_keep_dir = os.path.join(project_root, "nvcc_tmp")
nvcc_keep_dir = os.path.join(script_dir, "nvcc_tmp")

print(nvcc_keep_dir)

os.makedirs(nvcc_keep_dir, exist_ok=True)

#exit()

# 1. JIT-КОМПІЛЯЦІЯ CUDA-ЯДРА
# Це викликає компілятор NVIDIA і створює Python-модуль "pure_cuda_linear" (для підміни стандартного nn.Linear)
if ptdtype == torch.float32:
    cuda_backend = load(
        name="pure_cuda_linear", 
        sources=["my_cores/custom_linear_forward__fp32.cu"],    
        extra_include_paths=["third_party/cutlass/include"],
        #extra_cflags=['/std:c++20'],  # Використовуємо C++20 для MSVC
        extra_cuda_cflags=[
            '-std=c++20'             # Використовуємо C++20 для NVCC
            #, '-lcublas'             # Важливо!
            ,"--keep"
            #,"--ptx"
            ,"--keep-dir"
            ,nvcc_keep_dir #"nvcc_tmp"
        ],
        extra_ldflags=['cublas.lib'],  # Додаємо cuBLAS для Windows
        verbose=True
    )
else: # torch.float16
    cuda_backend = load(
        name="pure_cuda_linear", 
        sources=["my_cores/custom_linear_forward__fp16.cu"],
        extra_include_paths=["third_party/cutlass/include"],
        #extra_cflags=['/std:c++20'],  # Використовуємо C++20 для MSVC
        extra_cuda_cflags=[
            '-std=c++20'             # Використовуємо C++20 для NVCC
            #, '-lcublas'             # Важливо!
            ,"--keep"
            #,"--ptx"
            ,"--keep-dir"
            ,nvcc_keep_dir #"nvcc_tmp"
        ],
        extra_ldflags=['cublas.lib'],  # Додаємо cuBLAS для Windows
        verbose=True
    )
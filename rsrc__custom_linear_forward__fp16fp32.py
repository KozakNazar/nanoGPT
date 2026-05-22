"""
Sample from a trained model
"""
import os
import time
import pickle
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
compile = True
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ptdtype = torch.float16 #!
#ptdtype = torch.float32 #!
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

# 1. JIT-КОМПІЛЯЦІЯ ЧИСТОГО CUDA-ЯДРА
# Це викликає компілятор NVIDIA і створює Python-модуль "pure_cuda_linear" (для підміни стандартного nn.Linear)
if ptdtype == torch.float32:
    cuda_backend = load(
        name="pure_cuda_linear", 
        sources=["my_cores/custom_linear_forward__fp32.cu"], 
        #extra_cflags=['/std:c++20'],  # Використовуємо C++20 для MSVC
        extra_cuda_cflags=[
            '-std=c++20'             # Використовуємо C++20 для NVCC
        ],
        verbose=True
    )
else: # torch.float16
    cuda_backend = load(
        name="pure_cuda_linear", 
        sources=["my_cores/custom_linear_forward__fp16.cu"], 
        #extra_cflags=['/std:c++20'],  # Використовуємо C++20 для MSVC
        extra_cuda_cflags=[
            '-std=c++20'             # Використовуємо C++20 для NVCC
        ],
        verbose=True
    )

# 2. PYTHON-ОБГОРТКИ ДЛЯ DROP-IN ЗАМІНИ ШАРУ
class CUDAFP32LinearInference(nn.Module):
    def __init__(self, original_linear_layer):
        super().__init__()
        self.weight = original_linear_layer.weight
        self.bias = original_linear_layer.bias

    @torch.no_grad()
    def forward(self, x):
        orig_shape = x.shape
        x_flatten = x.view(-1, orig_shape[-1])
        
        # Конвертуємо у FP32
        out = cuda_backend.forward(x_flatten.float(), self.weight.float())
        
        out = out.view(*orig_shape[:-1], -1)
        
        if self.bias is not None:
            out += self.bias.float()
        
        # Повертаємо відповідний тип, якщо потрібно для autocast
        return out.to(x.dtype)

class CUDAFP16LinearInference(nn.Module):
    def __init__(self, original_linear_layer):
        super().__init__()
        self.weight = original_linear_layer.weight
        self.bias = original_linear_layer.bias

    @torch.no_grad()
    def forward(self, x):
        orig_shape = x.shape
        x_flatten = x.view(-1, orig_shape[-1])
        
        # Конвертуємо в FP16
        x_fp16 = x_flatten.half()
        weight_fp16 = self.weight.half()
        
        out = cuda_backend.forward(x_fp16, weight_fp16)
        
        out = out.view(*orig_shape[:-1], -1)
        
        if self.bias is not None:
            out += self.bias.half()
        
        # Повертаємо відповідний тип, якщо потрібно для autocast
        return out.to(x.dtype)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

## run generation
#with torch.no_grad():
#    with ctx:
#        for k in range(num_samples):
#            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#            print(decode(y[0].tolist()))
#            print('---------------')

# Крок А: Замір швидкості на оригінальному cuBLAS від NVIDIA
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    # x — стартовий тензор токенів
    y_orig = model.generate(x, max_new_tokens=20, temperature=1.0)
torch.cuda.synchronize()
print(f"Оригінальний cuBLAS час: {time.time() - start_time:.4f} сек")

# Крок Б: Тотальна підміна шарів на кастомне ядро
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]
        
        # Підміняємо стандартний nn.Linear на кастомний інференс-модуль
        if ptdtype == torch.float32:
            setattr(parent_module, child_name, CUDAFP32LinearInference(module))
        else: # torch.float16
            setattr(parent_module, child_name, CUDAFP16LinearInference(module))

print("Усі лінійні шари успішно замінено на чисте CUDA-ядро!")

if device_type == 'cuda' and (ptdtype == torch.float16 or ptdtype == torch.float32):
    # Крок В: Замір швидкості на чистому CUDA-ядрі
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        y_custom = model.generate(x, max_new_tokens=20, temperature=1.0)
    torch.cuda.synchronize()
    print(f"Чисте CUDA-ядро час: {time.time() - start_time:.4f} сек")
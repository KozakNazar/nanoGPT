import torch
import transformers
from dataclasses import dataclass

print(f"--- Перевірка середовища ---")
print(f"Версія Python: {torch.sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")

print(f"\n--- Перевірка GPU ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA доступна: {cuda_available}")

if cuda_available:
    print(f"Ваша карта: {torch.cuda.get_device_name(0)}")
    print(f"Версія CUDA в PyTorch: {torch.version.cuda}")
    
    # Тест обчислень на GPU
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    print("Тестове множення матриць на GPU: УСПІШНО")
    
    # Перевірка bfloat16 (фішка вашої карти)
    print(f"Підтримка bfloat16: {torch.cuda.is_bf16_supported()}")
else:
    print("ПОМИЛКА: PyTorch не бачить вашу відеокарту!")
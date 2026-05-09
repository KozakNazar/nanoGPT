import os
import torch

# Додаємо шлях до бінарників CUDA v13.0
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin") # !
# Також шлях до бібліотек PyTorch
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

from torch.utils.cpp_extension import load

# JIT-компіляція
custom_addmm = load(
    name="custom_addmm",
    sources=["my_cores/custom_addmm.cu"],
    extra_cflags=['/std:c++20'],  # Використовуємо C++20 для MSVC
    extra_cuda_cflags=[
        '-std=c++20',             # Використовуємо C++20 для NVCC
        '--expt-relaxed-constexpr',
        # Вимикаємо проблемний компонент, якщо виникає конфлікт 'std'
        '-DTORCH_COMPILED_AUTOGRAD_ENABLED=0' 
    ],
    verbose=True
)

# Тестові дані (матриці 2x2)
# Формула: res = beta * input + alpha * (mat1 @ mat2)
input_t = torch.ones((2, 2), device='cuda') * 10.0
mat1 = torch.ones((2, 2), device='cuda') * 2.0
mat2 = torch.ones((2, 2), device='cuda') * 3.0

print("\n--- Запуск torch.addmm (кастомне ядро) ---")
# Викликаємо стандартний метод, який тепер перенаправлено на новий CUDA-код
output = torch.addmm(input_t, mat1, mat2, beta=1.0, alpha=1.0)

print(f"Форма результату: {output.shape}")
print("Результат обчислення:")
print(output)

# Перевірка: (1.0 * 10) + (1.0 * (2*3 + 2*3)) = 10 + 12 = 22
expected = 22.0
if torch.allclose(output, torch.full_like(output, expected)):
    print("\n✅ Результат збігається з очікуваним (22.0)!")
else:
    print("\n❌ Результат відрізняється від стандартного. Перевірте логіку в .cu файлі.")
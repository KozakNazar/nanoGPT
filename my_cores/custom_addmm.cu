//#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

__global__ void custom_addmm_kernel(const float* bias, const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor custom_addmm_cuda__OLD(const at::Tensor& bias, const at::Tensor& A, const at::Tensor& B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = at::empty({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    custom_addmm_kernel<<<blocks, threads>>>(
        bias.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}

// Змінюємо сигнатуру: додаємо alpha та beta
at::Tensor custom_addmm_cuda(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha) {
    int M = mat1.size(0);
    int N = mat2.size(1);
    int K = mat1.size(1);

    auto C = at::empty({ M, N }, mat1.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    // У ядрі sum має ініціалізуватися як: beta * bias[col]
    // А добуток матриць множитися на alpha.
    if(1) custom_addmm_kernel << <blocks, threads >> > (
        self.data_ptr<float>(), mat1.data_ptr<float>(), mat2.data_ptr<float>(), C.data_ptr<float>(), M, N, K
        );

    return C;
}

// 1. Реєстрація підміни ядра в системі PyTorch
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    // Важливо, щоб сигнатура custom_addmm_cuda збігається
    m.impl("addmm", custom_addmm_cuda); // TORCH_BOX(&custom_addmm_cuda)
}

// 2. Точка входу для Python (щоб load() не видавав помилку)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Тут можна нічого не писати, якщо не потрібно викликати функцію напряму через custom_addmm.forward(), а використовувати її через torch.addmm
}
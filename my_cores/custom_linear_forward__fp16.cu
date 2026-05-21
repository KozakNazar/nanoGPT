#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Ядро для FP16 (просте)
__global__ void fp16_matmul_kernel(const half* X,
    const half* W,
    half* Y,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;  // Акумулюємо у FP32 для точності
        for (int i = 0; i < K; ++i) {
            // Конвертуємо half -> float, множимо, додаємо
            sum += __half2float(X[row * K + i]) * __half2float(W[col * K + i]);
        }
        // Конвертуємо результат назад у half
        Y[row * N + col] = __float2half(sum);
    }
}

// Функція-обгортка для PyTorch
torch::Tensor custom_linear_forward(torch::Tensor X, torch::Tensor W) {
    // Переконуємось, що вхідні тензори мають тип FP16
    TORCH_CHECK(X.scalar_type() == torch::kFloat16, "X must be Float16");
    TORCH_CHECK(W.scalar_type() == torch::kFloat16, "W must be Float16");
    TORCH_CHECK(X.is_cuda(), "X must be on CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be on CUDA");

    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(0);

    // Створюємо вихідний тензор типу FP16
    auto Y = torch::empty({ M, N }, torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA));

    // Отримуємо сирі вказівники
    const half* d_X = reinterpret_cast<const half*>(X.data_ptr());
    const half* d_W = reinterpret_cast<const half*>(W.data_ptr());
    half* d_Y = reinterpret_cast<half*>(Y.data_ptr());

    // Конфігурація сітки
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Запуск ядра
    fp16_matmul_kernel << <grid, block >> > (d_X, d_W, d_Y, M, N, K);

    // Перевірка помилок (опціонально)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return Y;
}

// Реєстрація модуля
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_linear_forward, "Pure FP16 CUDA MatMul for nn.Linear");
}
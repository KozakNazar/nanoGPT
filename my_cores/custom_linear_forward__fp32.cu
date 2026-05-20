#include <torch/extension.h>
#include <cuda_runtime.h>

// Ядро для FP32 (чистий FP32)
__global__ void fp32_matmul_kernel(const float* X,
    const float* W,
    float* Y,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            // Пряме множення FP32
            sum += X[row * K + i] * W[col * K + i];
        }
        Y[row * N + col] = sum;
    }
}

// Функція-обгортка для PyTorch
torch::Tensor custom_linear_forward(torch::Tensor X, torch::Tensor W) {
    // Переконуємось, що вхідні тензори мають тип FP32
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be Float32");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be Float32");
    TORCH_CHECK(X.is_cuda(), "X must be on CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be on CUDA");

    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(0);

    // Створюємо вихідний тензор типу FP32
    auto Y = torch::empty({ M, N }, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA));

    // Отримуємо сирі вказівники
    const float* d_X = X.data_ptr<float>();
    const float* d_W = W.data_ptr<float>();
    float* d_Y = Y.data_ptr<float>();

    // Конфігурація сітки
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Запуск ядра
    fp32_matmul_kernel << <grid, block >> > (d_X, d_W, d_Y, M, N, K);

    // Перевірка помилок (опціонально)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return Y;
}

// Реєстрація модуля
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_linear_forward, "Pure FP32 CUDA MatMul for nn.Linear");
}
//#define CUDA_CODE
#define CUBLAS_CODE
#ifdef CUDA_CODE

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <iostream> //!
#include <vector> //!
#include <cublas_v2.h> //!
//#include <cuda_runtime.h> //!

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

    //Y.zero_(); return Y;

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

torch::Tensor custom_linear_forward_(torch::Tensor X, torch::Tensor W) {
    // МІНІМАЛЬНА заглушка - майже без overhead
    int M = X.size(0);
    int N = W.size(0);

    return torch::zeros({ M, N }, X.options());
}

// Реєстрація модуля
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_linear_forward, "Pure FP32 CUDA MatMul for nn.Linear");
}

#elif defined(CUBLAS_CODE)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (!g_handle) {
        cublasCreate(&g_handle);
    }
    return g_handle;
}

torch::Tensor custom_linear_forward(
    torch::Tensor X,
    torch::Tensor W)
{
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be CUDA");

    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
        "X must be FP32");

    TORCH_CHECK(W.scalar_type() == torch::kFloat32,
        "W must be FP32");

    X = X.contiguous();
    W = W.contiguous();

    const int M = X.size(0);
    const int K = X.size(1);

    const int N = W.size(0);

    TORCH_CHECK(W.size(1) == K,
        "Shape mismatch");

    auto Y = torch::empty(
        { M, N },
        X.options()
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle = get_handle();

    cublasStatus_t status = cublasSgemm(
        handle,

        // op(A)
        CUBLAS_OP_T,

        // op(B)
        CUBLAS_OP_N,

        // m
        N,

        // n
        M,

        // k
        K,

        &alpha,

        // A = W
        W.data_ptr<float>(),

        // lda
        K,

        // B = X
        X.data_ptr<float>(),

        // ldb
        K,

        &beta,

        // C = Y
        Y.data_ptr<float>(),

        // ldc
        N
    );

    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "cublasSgemm failed"
    );

    return Y;
}

void cleanup() {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
        &custom_linear_forward,
        "cuBLAS FP32 Linear");

    m.def("cleanup",
        &cleanup,
        "cleanup");
}
#endif
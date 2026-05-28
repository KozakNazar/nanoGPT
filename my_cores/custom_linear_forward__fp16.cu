//#define CUDA_CODE
//#define CUBLAS_CODE
#define CUTLASS_CODE
#ifdef CUDA_CODE

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
#elif defined(CUBLAS_CODE)

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// ============================================================
// Global cuBLAS handle
// ============================================================

static cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (!g_handle) {
        cublasCreate(&g_handle);

        // Tensor Cores
        cublasSetMathMode(
            g_handle,
            CUBLAS_TENSOR_OP_MATH
        );
    }

    return g_handle;
}

// ============================================================
// FP16 Linear
//
// Computes:
//
//     Y = X * W^T
//
// X : [M,K] FP16
// W : [N,K] FP16
// Y : [M,N] FP16
//
// ============================================================

torch::Tensor custom_linear_forward(
    torch::Tensor X,
    torch::Tensor W)
{
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be CUDA");

    TORCH_CHECK(
        X.scalar_type() == torch::kFloat16,
        "X must be FP16"
    );

    TORCH_CHECK(
        W.scalar_type() == torch::kFloat16,
        "W must be FP16"
    );

    X = X.contiguous();
    W = W.contiguous();

    const int M = X.size(0);
    const int K = X.size(1);

    const int N = W.size(0);

    TORCH_CHECK(
        W.size(1) == K,
        "Shape mismatch"
    );

    auto Y = torch::empty(
        { M, N },
        X.options()
    );

    cublasHandle_t handle = get_handle();

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);

    // ========================================================
    // PyTorch row-major:
    //
    //     Y = X * W^T
    //
    // cuBLAS column-major trick:
    //
    //     Y^T = W * X^T
    //
    // ========================================================

    cublasStatus_t status = cublasHgemm(
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
        reinterpret_cast<const half*>(
            W.data_ptr<at::Half>()
            ),

        // lda
        K,

        // B = X
        reinterpret_cast<const half*>(
            X.data_ptr<at::Half>()
            ),

        // ldb
        K,

        &beta,

        // C = Y
        reinterpret_cast<half*>(
            Y.data_ptr<at::Half>()
            ),

        // ldc
        N
    );

    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "cublasHgemm failed"
    );

    return Y;
}

// ============================================================

void cleanup() {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def(
        "forward",
        &custom_linear_forward,
        "cuBLAS FP16 Linear"
    );

    m.def(
        "cleanup",
        &cleanup,
        "cleanup"
    );
}

#elif defined(CUTLASS_CODE)

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/half.h>

//#include <cutlass/layout/matrix.h>

#include <cutlass/gemm/device/gemm.h>

//static cudaStream_t get_cuda_stream() {
//    return at::cuda::getDefaultCUDAStream();
//}

// ============================================================
// FP16 GEMM via CUTLASS
//
// Computes:
//
//     Y = X * W^T
//
// Shapes:
//
//     X : [M, K] FP16
//     W : [N, K] FP16
//     Y : [M, N] FP16
//
// PyTorch tensors are ROW-MAJOR.
//
// ============================================================

torch::Tensor custom_linear_forward(
    torch::Tensor X,
    torch::Tensor W)
{
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be CUDA");

    TORCH_CHECK(
        X.scalar_type() == torch::kFloat16,
        "X must be float16"
    );

    TORCH_CHECK(
        W.scalar_type() == torch::kFloat16,
        "W must be float16"
    );

    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");

    X = X.contiguous();
    W = W.contiguous();

    int M = X.size(0);
    int K = X.size(1);

    int N = W.size(0);

    TORCH_CHECK(
        W.size(1) == K,
        "Shape mismatch"
    );

    auto Y = torch::empty(
        { M, N },
        X.options()
    );

    // ========================================================
    // CUTLASS GEMM configuration
    //
    // We compute:
    //
    //     Y = X * W^T
    //
    // PyTorch row-major:
    //
    //     X = [M,K]
    //     W = [N,K]
    //
    // CUTLASS trick:
    //
    // Row-major W[N,K]
    // == Column-major W^T[K,N]
    //
    // ========================================================

    using ElementInputA = cutlass::half_t;
    using LayoutInputA = cutlass::layout::RowMajor;

    using ElementInputB = cutlass::half_t;
    using LayoutInputB = cutlass::layout::ColumnMajor;

    using ElementOutput = cutlass::half_t;
    using LayoutOutput = cutlass::layout::RowMajor;

    // FP32 accumulation
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA,
        LayoutInputA,

        ElementInputB,
        LayoutInputB,

        ElementOutput,
        LayoutOutput,

        ElementAccumulator
    >;

    typename Gemm::Arguments arguments(

        // Problem size
        { M, N, K },

        // Tensor A
        {
            reinterpret_cast<cutlass::half_t*>(
                X.data_ptr<at::Half>()
            ),
            K
        },

        // Tensor B
        {
            reinterpret_cast<cutlass::half_t*>(
                W.data_ptr<at::Half>()
            ),
            K
        },

        // Tensor C
        {
            reinterpret_cast<cutlass::half_t*>(
                Y.data_ptr<at::Half>()
            ),
            N
        },

        // Tensor D
        {
            reinterpret_cast<cutlass::half_t*>(
                Y.data_ptr<at::Half>()
            ),
            N
        },

        // alpha, beta
        {
            1.0f,
            0.0f
        }
    );

    Gemm gemm_op;

    cutlass::Status status = gemm_op(arguments);

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS FP16 GEMM failed"
    );

    return Y;
}

// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def(
        "forward",
        &custom_linear_forward,
        "CUTLASS FP16 Linear"
    );
}

#endif
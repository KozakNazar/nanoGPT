#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/tensor_ref.h"

#include "cutlass/numeric_types.h"


//static cudaStream_t get_cuda_stream() {
//    return at::cuda::getDefaultCUDAStream();
//}

// ============================================================
// FP32 GEMM via CUTLASS
//
// Computes:
//
//     Y = X * W^T
//
// Shapes:
//
//     X : [M, K]
//     W : [N, K]
//     Y : [M, N]
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

    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
        "X must be float32");

    TORCH_CHECK(W.scalar_type() == torch::kFloat32,
        "W must be float32");

    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");

    X = X.contiguous();
    W = W.contiguous();

    int M = (int)X.size(0);
    int K = (int)X.size(1);
    int N = (int)W.size(0);

    TORCH_CHECK(W.size(1) == K,
        "Shape mismatch");

    auto Y = torch::empty(
        { M, N },
        X.options()
    );

    // =====================================================
    // Explicit CUTLASS type definitions
    // =====================================================

    typedef float ElementA;
    typedef float ElementB;
    typedef float ElementC;
    typedef float ElementAccumulator;

    typedef cutlass::layout::RowMajor LayoutA;
    typedef cutlass::layout::ColumnMajor LayoutB;
    typedef cutlass::layout::RowMajor LayoutC;

    typedef cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator
    > GemmOperator;

    // =====================================================
    // Explicit tensor refs
    // =====================================================

    cutlass::TensorRef<
        ElementA const,
        LayoutA
    > ref_A(
        X.data_ptr<float>(),
        K
    );

    cutlass::TensorRef<
        ElementB const,
        LayoutB
    > ref_B(
        W.data_ptr<float>(),
        K
    );

    cutlass::TensorRef<
        ElementC const,
        LayoutC
    > ref_C(
        Y.data_ptr<float>(),
        N
    );

    cutlass::TensorRef<
        ElementC,
        LayoutC
    > ref_D(
        Y.data_ptr<float>(),
        N
    );

    // =====================================================
    // Explicit problem size
    // =====================================================

    cutlass::gemm::GemmCoord problem_size(
        M,
        N,
        K
    );

    // =====================================================
    // Explicit epilogue params
    // =====================================================

    typename GemmOperator::EpilogueOutputOp::Params epilogue(
        1.0f,
        0.0f
    );

    // =====================================================
    // Explicit arguments struct
    // =====================================================

    typename GemmOperator::Arguments args(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        epilogue,
        1,
        nullptr,
        nullptr,
        nullptr
    );

    // =====================================================
    // Explicit operator object
    // =====================================================

    GemmOperator gemm_op;

    // =====================================================
    // Explicit capability check
    // =====================================================

    cutlass::Status status;

    status = GemmOperator::can_implement(args);

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "GEMM configuration not supported"
    );

    // =====================================================
    // Explicit workspace allocation
    // =====================================================

    size_t workspace_size =
        GemmOperator::get_workspace_size(args);

    void* workspace = nullptr;

    if (workspace_size > 0)
    {
        cudaError_t cuda_status =
            cudaMalloc(&workspace, workspace_size);

        TORCH_CHECK(
            cuda_status == cudaSuccess,
            "cudaMalloc failed"
        );
    }

    // =====================================================
    // Explicit initialize
    // =====================================================

    status = gemm_op.initialize(
        args,
        workspace,
        nullptr
    );

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "initialize() failed"
    );

    // =====================================================
    // Explicit kernel launch
    // =====================================================

    status = gemm_op.run(nullptr);

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "run() failed"
    );

    // =====================================================
    // Cleanup
    // =====================================================

    if (workspace)
    {
        cudaFree(workspace);
    }

    return Y;
}

// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &custom_linear_forward,
        "CUTLASS FP32 Linear"
    );
}
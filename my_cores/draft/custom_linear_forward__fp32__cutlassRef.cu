#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h" // GemmIdentityThreadblockSwizzle
#include "cutlass/gemm/kernel/gemm.h" //   /// Kernel parameters object //typename GemmKernel::Params params_;

#include "cutlass/gemm/kernel/default_gemm.h" //+
#include "cutlass/gemm/device/default_gemm_configuration.h" //+

#include "cutlass/layout/permute.h" //+

#include "cutlass/gemm/device/gemm.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/tensor_ref.h"

#include "cutlass/numeric_types.h" // 


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
    
    status = gemm_op.run(nullptr); ////////////////////////////////:
    /// Threadblock-level swizzling operator
    //typename ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>;
    //using ThreadblockSwizzle = ThreadblockSwizzle_;
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> threadblock_swizzle; // ThreadblockSwizzle threadblock_swizzle; // typename ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>

    /// Operator class tag
    typedef cutlass::arch::OpClassSimt OperatorClass__;//typename OperatorClass_ = arch::OpClassSimt; 

        /// Tag indicating architecture to tune for
    typedef cutlass::arch::Sm70 ArchTag__; //typename ArchTag_ = arch::Sm70;

    /// Threadblock-level tile size (concept: GemmShape)
    typedef cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::ThreadblockShape
        ThreadblockShape__;

    /// Warp-level tile size (concept: GemmShape)
    typedef cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::WarpShape
        WarpShape__;

    /// Instruction-level tile size (concept: GemmShape)
    typedef cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::InstructionShape
        InstructionShape__;

    /// Epilogue output operator
    typedef cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::EpilogueOutputOp
        EpilogueOutputOp__;

    /// Threadblock-level swizzling operator
    typedef cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>
        ThreadblockSwizzle__;
      
    /// Number of stages used in the pipelined mainloop       
    int Stages =     
        cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kStages,

    /// Access granularity of A matrix in units of elements
    /*!*/int AlignmentA =
        cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kAlignmentA;
        /// Access granularity of B matrix in units of elements
    /*!*/int AlignmentB =
        cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kAlignmentB;

    /// If true, kernel supports split-K with serial reduction
    /*!*/bool SplitKSerial = false;

    /// Operation performed by GEMM
    typedef cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::Operator
        Operator__;

    /// Gather operand A by using an index array
    /*!*/bool GatherA = false;
       
    /// Gather operand B by using an index array
    /*!*/bool GatherB = false;
    
    /// Scatter result D by using an index array
    /*!*/bool ScatterD = false;

    /// Permute result D
    typedef cutlass::layout::NoPermute PermuteDLayout;

//#define kAlignmentA AlignmentA
//    /*static*/ int const kAlignmentB AlignmentB
//    /*static*/ int const kAlignmentC = EpilogueOutputOp__::kCount;
    ///*static*/ int const kStages = Stages;
    ///*static*/ bool const kSplitKSerial = SplitKSerial;

    /// Define the kernel
    typedef 
        cutlass::gemm::kernel::DefaultGemm <
        ElementA, // ElementA
        LayoutA, // LayoutA
        //kAlignmentA,
            cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
            ElementC/*_*/, ElementAccumulator/*_*/>::kAlignmentA,
        ElementB, // ElementB
        LayoutB, // LayoutB
        //kAlignmentB,
            cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
            ElementC/*_*/, ElementAccumulator/*_*/>::kAlignmentB,
        ElementC, // ElementC
        LayoutC, // LayoutC
        ElementAccumulator, // ElementAccumulator
        OperatorClass__/*OperatorClass*/,
        ArchTag__/*ArchTag_*/,
        ThreadblockShape__/*ThreadblockShape*/,
        WarpShape__/*WarpShape*/,
        InstructionShape__/*InstructionShape*/,
        EpilogueOutputOp__/*EpilogueOutputOp*/,
        ThreadblockSwizzle__/*ThreadblockSwizzle*//*cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>*/,
        //kStages,
            cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
            ElementC/*_*/, ElementAccumulator/*_*/>::kStages,
        false/*kSplitKSerial*/,
        Operator__/*Operator*/,
        cutlass::gemm::SharedMemoryClearOption::kNone, //!
        false/*GatherA*/,
        false/*GatherB*/,
        false/*ScatterD*/,
        PermuteDLayout
    > ::GemmKernel
        GemmKernel__;

#if 0

    /// Define the kernel
    using GemmKernel = typename kernel::DefaultGemm <
        ElementA,
        LayoutA,
        kAlignmentA,
        ElementB,
        LayoutB,
        kAlignmentB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp,
        ThreadblockSwizzle,
        kStages,
        kSplitKSerial,
        Operator,
        SharedMemoryClearOption::kNone,
        GatherA,
        GatherB,
        ScatterD,
        PermuteDLayout
    > ::GemmKernel;

dim3 grid =
    threadblock_swizzle.get_grid_shape(
        GemmKernel::Params.grid_tiled_shape // params_.grid_tiled_shape
    );

dim3 block(
    GemmKernel::kThreadCount,
    1,
    1
);

cudaError_t result;

int smem_size =
    int(sizeof(typename GemmKernel::SharedStorage));

if (smem_size >= (48 << 10))
{
    result =
        cudaFuncSetAttribute(
            Kernel<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

    if (result != cudaSuccess)
    {
        status = cutlass::Status::kErrorInternal;
    }
}

cutlass::arch::synclog_setup();

cutlass::Kernel<GemmKernel>
<<<grid, block, smem_size, nullptr>>>(
    params_
);

result = cudaGetLastError();

status =
    (result == cudaSuccess)
        ? cutlass::Status::kSuccess
        : cutlass::Status::kErrorInternal;
#endif
    //////////////////////////////////////////////////////////////////////

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
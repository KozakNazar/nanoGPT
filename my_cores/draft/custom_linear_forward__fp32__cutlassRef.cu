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
    int Stages__ =
        cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kStages;

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
    typedef cutlass::layout::NoPermute PermuteDLayout__;

    //#define kAlignmentA AlignmentA
    //    /*static*/ int const kAlignmentB AlignmentB
    //    /*static*/ int const kAlignmentC = EpilogueOutputOp__::kCount;
        ///*static*/ int const kStages = Stages;
        ///*static*/ bool const kSplitKSerial = SplitKSerial;


    /// Threadblock-level swizzling operator
    //typename ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>;
    //using ThreadblockSwizzle = ThreadblockSwizzle_;
    ThreadblockSwizzle__/*cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>*/ threadblock_swizzle__1; // ThreadblockSwizzle threadblock_swizzle; // typename ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>

    /// Define the kernel
    using GemmKernel__ = typename
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
        PermuteDLayout__
        > ::GemmKernel
        ;//GemmKernel__;

  /// Kernel parameters object
    typename GemmKernel__::Params params_;


    // =====================================================
    // Explicit initialize
    // =====================================================

#if 0
    status = gemm_op.initialize(
        args,
        workspace,
        nullptr
    );
#else
    //Arguments const& 
    args; 
    //void* 
    workspace/*= nullptr*/; 
    cudaStream_t stream = nullptr;
    // Determine grid shape
    ThreadblockSwizzle__ threadblock_swizzle__2;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle__2.get_tiled_shape(
        args.problem_size,
        { ThreadblockShape__::kM, ThreadblockShape__::kN, ThreadblockShape__::kK },
        args.split_k_slices);

    using UnderlyingOperator = cutlass::gemm::device::Gemm <
        ElementB, // ElementB
        typename cutlass::layout::LayoutTranspose<LayoutB>::type,
        ElementA, // ElementA
        typename cutlass::layout::LayoutTranspose<LayoutA>::type,
        ElementC, // ElementC
        cutlass::layout::RowMajor,
        ElementAccumulator, // ElementAccumulator
        OperatorClass__/*OperatorClass*/,
        ArchTag__/*ArchTag_*/,
        ThreadblockShape__/*ThreadblockShape*/,
        WarpShape__/*WarpShape*/,
        InstructionShape__/*InstructionShape*/,
        EpilogueOutputOp__/*EpilogueOutputOp*/,
        ThreadblockSwizzle__/*ThreadblockSwizzle*//*cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>*/,
        //Stages__,
        cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kStages,
        //kAlignmentB, // B
        cutlass::gemm::device::DefaultGemmConfiguration < OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kAlignmentB,
        //kAlignmentA, // A
        cutlass::gemm::device::DefaultGemmConfiguration < OperatorClass__/*OperatorClass_*/, ArchTag__/*ArchTag_*/, ElementA/*_*/, ElementB/*_*/,
        ElementC/*_*/, ElementAccumulator/*_*/>::kAlignmentA,
        false/*kSplitKSerial*/,
        Operator__/*Operator*/,
        false/*GatherB*/, // B
        false/*GatherA*/, // A
        false/*ScatterD*/,
        PermuteDLayout__
    > ;

    //using UnderlyingArguments = typename UnderlyingOperator::Arguments;

    if (false/*kSplitKSerial*/) {
        if (args.split_k_slices > 1) {
            if (!workspace) {
                //return cutlass::Status::kErrorWorkspaceNull;
                exit(0);
            }

            auto get_workspace_size__ = [](GemmOperator::Arguments const& args) -> size_t {

                size_t bytes = 0;

                // Determine grid shape
                ThreadblockSwizzle__ threadblock_swizzle__l;

                cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle__l.get_tiled_shape(
                    args.problem_size,
                    { ThreadblockShape__::kM, ThreadblockShape__::kN, ThreadblockShape__::kK },
                    args.split_k_slices);

                if (false/*kSplitKSerial*/ && args.split_k_slices > 1) {

                    bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
                }

                return bytes;
                };

            //UnderlyingArguments to_underlying_arguments__(Arguments const& args) {}


            size_t bytes = get_workspace_size__(args);  // size_t bytes = get_workspace_size(args);

            cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

            if (result != cudaSuccess) {
                //return cutlass::Status::kErrorInternal;
                exit(0);
            }
        }
    }
    else {

        if (args.split_k_slices > 1) {
            //return Status::kErrorInvalidProblem;
            exit(0);
        }
    }
#if 0
    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int*>(workspace),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    //return Status::kSuccess;
#endif
#endif

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "initialize() failed"
    );

    // =====================================================
    // Explicit kernel launch
    // =====================================================
    
    //status = gemm_op.run(nullptr); ////////////////////////////////:

    //dim3 grid =
    //    threadblock_swizzle__1.get_grid_shape(
    //        GemmKernel__::Params.grid_tiled_shape // params_.grid_tiled_shape
    //    );

    //dim3 block(
    //    GemmKernel__::kThreadCount,
    //    1,
    //    1
    //);

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
#endif

#if 0



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
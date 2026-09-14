#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void tree_search_kernel_shared(
    const float* __restrict__ inputs,
    const float* __restrict__ codebook,
    int64_t* __restrict__ output_indices,
    int num_vectors,
    int dim,
    int max_depth,
    int num_codebook_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_codebook[];
    int total = num_codebook_nodes * dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();

    if (idx >= num_vectors) return;

    const float* cur_vec = inputs + (int64_t)idx * dim;

    int64_t curr_node = 0;

    #pragma unroll 1
    for (int d = 0; d < max_depth; ++d) {
        int64_t left_child  = 2 * curr_node + 1;
        int64_t right_child = 2 * curr_node + 2;

        float dist_left = 0.0f;
        float dist_right = 0.0f;

        const float* left_ptr  = s_codebook + left_child  * (int64_t)dim;
        const float* right_ptr = s_codebook + right_child * (int64_t)dim;

        for (int k = 0; k < dim; ++k) {
            float v = cur_vec[k];

            float dl = v - left_ptr[k];
            dist_left += dl * dl;

            float dr = v - right_ptr[k];
            dist_right += dr * dr;
        }

        curr_node = (dist_left <= dist_right) ? left_child : right_child;
    }

    output_indices[idx] = curr_node;
}

__global__ void tree_search_kernel_global(
    const float* __restrict__ inputs,
    const float* __restrict__ codebook,
    int64_t* __restrict__ output_indices,
    int num_vectors,
    int dim,
    int max_depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;

    const float* cur_vec = inputs + (int64_t)idx * dim;
    int64_t curr_node = 0;

    #pragma unroll 1
    for (int d = 0; d < max_depth; ++d) {
        int64_t left_child  = 2 * curr_node + 1;
        int64_t right_child = 2 * curr_node + 2;

        float dist_left = 0.0f;
        float dist_right = 0.0f;

        const float* left_ptr  = codebook + left_child  * (int64_t)dim;
        const float* right_ptr = codebook + right_child * (int64_t)dim;

        for (int k = 0; k < dim; ++k) {
            float v = cur_vec[k];

            float dl = v - left_ptr[k];
            dist_left += dl * dl;

            float dr = v - right_ptr[k];
            dist_right += dr * dr;
        }

        curr_node = (dist_left <= dist_right) ? left_child : right_child;
    }

    output_indices[idx] = curr_node;
}

torch::Tensor tree_search_cuda_wrapper(
    torch::Tensor input,
    torch::Tensor codebook,
    int64_t max_depth)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "codebook must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat32, "codebook must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (N, D)");
    TORCH_CHECK(codebook.dim() == 2, "codebook must be 2D (M, D)");
    TORCH_CHECK(input.size(1) == codebook.size(1), "dim mismatch: input.size(1) != codebook.size(1)");
    TORCH_CHECK(max_depth >= 1, "max_depth must be >= 1");

    auto input_c = input.contiguous();
    auto codebook_c = codebook.contiguous();

    int64_t num_vectors64 = input_c.size(0);
    int64_t dim64 = input_c.size(1);

    TORCH_CHECK(num_vectors64 <= INT_MAX, "num_vectors too large for this kernel");
    TORCH_CHECK(dim64 <= INT_MAX, "dim too large for this kernel");

    int num_vectors = (int)num_vectors64;
    int dim = (int)dim64;

    int64_t needed_nodes = ((int64_t)1 << (max_depth + 1)) - 1;
    TORCH_CHECK(codebook_c.size(0) >= needed_nodes,
                "codebook too small for max_depth. need at least ",
                needed_nodes, " nodes, but got ", codebook_c.size(0));
    TORCH_CHECK(needed_nodes <= INT_MAX, "codebook nodes too large for this kernel");

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(input_c.device());
    torch::Tensor output = torch::empty({num_vectors64}, options);

    const int threads = 256;
    const int blocks = (num_vectors + threads - 1) / threads;
    const size_t shared_mem_bytes = (size_t)needed_nodes * (size_t)dim * sizeof(float);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const size_t default_shared_limit = static_cast<size_t>(prop.sharedMemPerBlock);
#if CUDART_VERSION >= 9000
    const size_t optin_shared_limit = static_cast<size_t>(prop.sharedMemPerBlockOptin);
#else
    const size_t optin_shared_limit = default_shared_limit;
#endif
    const size_t shared_limit = optin_shared_limit > 0 ? optin_shared_limit : default_shared_limit;

    if (shared_mem_bytes <= shared_limit) {
        if (shared_mem_bytes > default_shared_limit) {
#if CUDART_VERSION >= 9000
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                tree_search_kernel_shared,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_mem_bytes)
            ));
#else
            TORCH_CHECK(false,
                        "tree_search_kernel_shared requires more than the default shared-memory limit, "
                        "but this CUDA runtime does not support opting in to larger dynamic shared memory.");
#endif
        }

        tree_search_kernel_shared<<<blocks, threads, shared_mem_bytes, stream>>>(
            input_c.data_ptr<float>(),
            codebook_c.data_ptr<float>(),
            output.data_ptr<int64_t>(),
            num_vectors,
            dim,
            (int)max_depth,
            (int)needed_nodes
        );
    } else {
        tree_search_kernel_global<<<blocks, threads, 0, stream>>>(
            input_c.data_ptr<float>(),
            codebook_c.data_ptr<float>(),
            output.data_ptr<int64_t>(),
            num_vectors,
            dim,
            (int)max_depth
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

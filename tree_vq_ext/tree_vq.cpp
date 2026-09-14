#include <torch/extension.h>

torch::Tensor tree_search_cuda_wrapper(
    torch::Tensor input,
    torch::Tensor codebook,
    int64_t max_depth);


torch::Tensor tree_search(torch::Tensor input, torch::Tensor codebook, int64_t max_depth) {
    return tree_search_cuda_wrapper(input, codebook, max_depth);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tree_search", &tree_search, "Tree VQ search (CUDA)");
}

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _rpath_args():
    return ["-Wl,-rpath,/usr/local/lib/python3.12/dist-packages/torch/lib"]


setup(
    name="tree_vq_ext",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="tree_vq_ext._C",
            sources=["tree_vq.cpp", "tree_vq_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
            extra_link_args=_rpath_args(),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# Modified from: https://github.com/NVlabs/tiny-cuda-nn/blob/master/bindings/torch/setup.py

import glob
import os
import re
import shutil
import subprocess

import torch
from packaging.version import parse as parse_version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

package_name = "patchwork_query_ext"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = SCRIPT_DIR


def min_supported_compute_capability(cuda_version):
    if cuda_version >= parse_version("13.0"):
        return 75
    elif cuda_version >= parse_version("12.0"):
        return 50
    else:
        return 20


def max_supported_compute_capability(cuda_version):
    if cuda_version < parse_version("11.0"):
        return 75
    elif cuda_version < parse_version("11.1"):
        return 80
    elif cuda_version < parse_version("11.8"):
        return 86
    elif cuda_version < parse_version("12.8"):
        return 90
    else:
        return 120


if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    compute_capabilities = [major * 10 + minor]
    print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
else:
    raise EnvironmentError(
        "Unknown compute capability. Specify the target compute capabilities in the TCNN_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically."
    )


cpp_standard = 20

# Get CUDA version and make sure the targeted compute capability is compatible
if os.system("nvcc --version") == 0:
    nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
    cuda_version = re.search(r"release (\S+),", nvcc_out)

    if cuda_version:
        cuda_version = parse_version(cuda_version.group(1))
        print(f"Detected CUDA version {cuda_version}")
        if cuda_version >= parse_version("11.0"):
            cpp_standard = 20

        supported_compute_capabilities = [
            cc
            for cc in compute_capabilities
            if cc >= min_supported_compute_capability(cuda_version)
            and cc <= max_supported_compute_capability(cuda_version)
        ]

        if not supported_compute_capabilities:
            supported_compute_capabilities = [
                max_supported_compute_capability(cuda_version)
            ]

        if supported_compute_capabilities != compute_capabilities:
            print(
                f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead."
            )
            compute_capabilities = supported_compute_capabilities

min_compute_capability = min(compute_capabilities)

print(f"Targeting C++ standard {cpp_standard}")

base_nvcc_flags = [
    f"-std=c++{cpp_standard}",
    "--extended-lambda",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
]


if os.name == "posix":
    base_cflags = [f"-std=c++{cpp_standard}"]
    base_nvcc_flags += [
        "-Xcompiler=-Wno-float-conversion",
        "-Xcompiler=-fno-strict-aliasing",
    ]
elif os.name == "nt":
    base_cflags = [f"/std:c++{cpp_standard}"]


# Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# List of sources.
bindings_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))


compute_capability = compute_capabilities[0]
nvcc_flags = base_nvcc_flags + [
    f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}"
    for code in ["compute", "sm"]
]
cflags = base_cflags

setup(
    name=package_name,
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name=f"{package_name}._C",
            sources=glob.glob(f"{package_name}/src/*.cu"),
            include_dirs=[
                os.path.abspath(f"{package_name}/include"),
                os.path.abspath(f"{package_name}/third_party/cudaKDTree"),
                os.path.abspath(f"{package_name}/third_party/eigen"),
                os.path.abspath(f"{package_name}/third_party/vcglib"),
            ],
            extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
            libraries=["cuda"],
        )
    ],
    packages=[package_name],
    zip_safe=False,
    python_requires=">=3.10",
    cmdclass={"build_ext": BuildExtension},
)

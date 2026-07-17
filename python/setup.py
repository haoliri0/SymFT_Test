from pathlib import Path
import os
import platform
import shlex
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_CPP_SRC = PACKAGE_ROOT.parent / "cpp" / "src"
CPP_SRC = REPOSITORY_CPP_SRC if REPOSITORY_CPP_SRC.exists() else PACKAGE_ROOT / "cpp" / "src"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_CUDA = env_flag("SYMFT_PY_ENABLE_CUDA")
CUDA_REAL_DOUBLE = env_flag("SYMFT_PY_CUDA_REAL_DOUBLE")
CUDA_ARCH = os.environ.get("SYMFT_PY_CUDA_ARCH", "").strip()
CUDA_NVCC_FLAGS = os.environ.get("SYMFT_PY_CUDA_NVCC_FLAGS", "").strip()


def find_cuda_home():
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return None


def cuda_library_dirs(cuda_home):
    candidates = [cuda_home / "lib64", cuda_home / "lib", cuda_home / "lib" / "x64"]
    return [str(path) for path in candidates if path.exists()]


def cuda_nvcc(cuda_home):
    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    candidate = cuda_home / "bin" / executable
    if candidate.exists():
        return str(candidate)
    found = shutil.which("nvcc")
    return found or str(candidate)


CUDA_HOME = find_cuda_home() if ENABLE_CUDA else None


class BuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy

        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        state = (
            f"cuda={int(ENABLE_CUDA)};cuda_real_double={int(CUDA_REAL_DOUBLE)};"
            f"cuda_arch={CUDA_ARCH};cuda_nvcc_flags={CUDA_NVCC_FLAGS}"
        )
        marker = Path(self.build_temp) / "symft_build_state.txt"
        if not marker.exists() or marker.read_text() != state:
            self.force = True

        if ENABLE_CUDA:
            cuda_home = CUDA_HOME or find_cuda_home()
            if cuda_home is None:
                raise RuntimeError(
                    "CUDA build requested with SYMFT_PY_ENABLE_CUDA=1, "
                    "but CUDA_HOME/CUDA_PATH or nvcc was not found"
                )
            nvcc = cuda_nvcc(cuda_home)
            if ".cu" not in self.compiler.src_extensions:
                self.compiler.src_extensions.append(".cu")

            original_compile = self.compiler._compile

            def compile_with_nvcc(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if src.endswith(".cu"):
                    nvcc_args = [nvcc, "-c", src, "-o", obj, *cc_args]
                    if platform.system() == "Windows":
                        nvcc_args.extend(["-std=c++20", "-O2"])
                    else:
                        nvcc_args.extend(["-std=c++20", "-O3", "--compiler-options", "-fPIC"])
                    if CUDA_ARCH:
                        nvcc_args.append(f"-arch={CUDA_ARCH}")
                    if CUDA_NVCC_FLAGS:
                        nvcc_args.extend(shlex.split(CUDA_NVCC_FLAGS))
                    self.spawn(nvcc_args)
                else:
                    original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

            self.compiler._compile = compile_with_nvcc

        super().build_extensions()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(state)


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        shutil.copytree(CPP_SRC, Path(base_dir) / "cpp" / "src")


def cpp_source(path):
    return str(CPP_SRC / path)


sources = [
    "src/symft/_native.cpp",
    cpp_source("core/common.cpp"),
    cpp_source("core/pauli.cpp"),
    cpp_source("core/symbolic.cpp"),
    cpp_source("core/frames.cpp"),
    cpp_source("circuit/circuit_lowering.cpp"),
    cpp_source("sampler/active_state.cpp"),
    cpp_source("factored/factored_state.cpp"),
    cpp_source("factored/factored_planner.cpp"),
    cpp_source("sampler/presampled_expression.cpp"),
    cpp_source("sampler/single_shot_sampler.cpp"),
    cpp_source("sampler/prepared_sampler.cpp"),
    cpp_source("sampler/exogenous_presample.cpp"),
    cpp_source("frontend/stim_parser.cpp"),
    cpp_source("frontend/stim_sampling.cpp"),
    cpp_source("sampler/batch_symbols.cpp"),
    cpp_source("sampler/batch_active.cpp"),
    cpp_source("sampler/batch_runtime.cpp"),
    cpp_source("simd/simd_dispatch.cpp"),
    cpp_source("simd/simd_scalar.cpp"),
    cpp_source("simd/batch_simd_scalar.cpp"),
]

if ENABLE_CUDA:
    sources.extend(
        [
            cpp_source("cuda/cuda_program.cpp"),
            cpp_source("cuda/cuda_sampler.cpp"),
            cpp_source("cuda/cuda_runtime.cu"),
        ]
    )

if platform.system() == "Windows":
    compile_args = ["/std:c++20", "/O2"]
else:
    compile_args = ["-std=c++20", "-O3", "-fvisibility=hidden"]
link_args = []
if platform.system() == "Linux":
    compile_args.append("-pthread")
    link_args.append("-pthread")

include_dirs = [str(CPP_SRC)]
library_dirs = []
runtime_library_dirs = []
libraries = []
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if ENABLE_CUDA:
    define_macros.append(("SYMFT_CPP_ENABLE_CUDA", "1"))
    if CUDA_REAL_DOUBLE:
        define_macros.append(("SYMFT_CUDA_REAL_DOUBLE", "1"))
    libraries.append("cudart")
    if CUDA_HOME is not None:
        include_dirs.append(str(CUDA_HOME / "include"))
        library_dirs.extend(cuda_library_dirs(CUDA_HOME))
        if platform.system() == "Linux":
            runtime_library_dirs.extend(cuda_library_dirs(CUDA_HOME))

extensions = [
    Extension(
        "symft._native",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=define_macros,
    )
]

setup(
    name="symft",
    version="0.1.0",
    description="Python bindings for the SymFT Clifford+T simulator",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"symft": ["py.typed", "*.pyi"]},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExt, "sdist": Sdist},
    install_requires=["numpy>=1.20"],
    python_requires=">=3.9",
    zip_safe=False,
)

from pathlib import Path
import platform
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_CPP_SRC = PACKAGE_ROOT.parent / "cpp" / "src"
CPP_SRC = REPOSITORY_CPP_SRC if REPOSITORY_CPP_SRC.exists() else PACKAGE_ROOT / "cpp" / "src"


class BuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy

        self.include_dirs.append(numpy.get_include())


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

if platform.system() == "Windows":
    compile_args = ["/std:c++20", "/O2"]
else:
    compile_args = ["-std=c++20", "-O3", "-fvisibility=hidden"]
link_args = []
if platform.system() == "Linux":
    compile_args.append("-pthread")
    link_args.append("-pthread")

extensions = [
    Extension(
        "symft._native",
        sources=sources,
        include_dirs=[str(CPP_SRC)],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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

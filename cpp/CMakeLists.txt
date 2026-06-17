cmake_minimum_required(VERSION 3.20)

project(symft_cpp LANGUAGES CXX)

include(CheckCXXCompilerFlag)
find_package(Threads REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SYMFT_CPP_BUILD_TESTS "Build C++ SymFT tests" ON)
option(SYMFT_CPP_ENABLE_AVX2 "Build AVX2 active kernels when the compiler supports them" ON)
option(SYMFT_CPP_ENABLE_AVX512 "Build AVX512 active kernels when the compiler supports them" ON)
option(SYMFT_CPP_NATIVE "Build host-vectorized kernels with -march=native" ON)

add_library(symft_cpp
    src/core/common.cpp
    src/core/pauli.cpp
    src/core/symbolic.cpp
    src/core/frames.cpp
    src/circuit/circuit_lowering.cpp
    src/sampler/active_state.cpp
    src/factored/factored_state.cpp
    src/factored/factored_planner.cpp
    src/sampler/single_shot_sampler.cpp
    src/sampler/exogenous_presample.cpp
    src/frontend/stim_parser.cpp
    src/frontend/stim_sampling.cpp
    src/sampler/batch_symbols.cpp
    src/sampler/batch_active.cpp
    src/sampler/batch_runtime.cpp
    src/simd/simd_dispatch.cpp
    src/simd/simd_scalar.cpp
    src/simd/batch_simd_scalar.cpp
)

target_include_directories(symft_cpp PUBLIC src)
target_compile_features(symft_cpp PUBLIC cxx_std_20)
target_compile_options(symft_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

if(SYMFT_CPP_NATIVE)
    target_compile_options(symft_cpp PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>
    )
    target_compile_definitions(symft_cpp PRIVATE SYMFT_CPP_NATIVE_BUILD=1)
endif()

if(SYMFT_CPP_ENABLE_AVX2)
    check_cxx_compiler_flag("-mavx2" SYMFT_HAS_MAVX2)
    check_cxx_compiler_flag("-mfma" SYMFT_HAS_MFMA)
    if(SYMFT_HAS_MAVX2 AND SYMFT_HAS_MFMA)
        target_sources(symft_cpp PRIVATE src/simd/simd_avx2.cpp)
        set_source_files_properties(src/simd/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        target_compile_definitions(symft_cpp PRIVATE SYMFT_COMPILED_AVX2=1)
    endif()
endif()

if(SYMFT_CPP_ENABLE_AVX512)
    check_cxx_compiler_flag("-mavx512f" SYMFT_HAS_MAVX512F)
    check_cxx_compiler_flag("-mavx512dq" SYMFT_HAS_MAVX512DQ)
    if(SYMFT_HAS_MAVX512F AND SYMFT_HAS_MAVX512DQ AND SYMFT_HAS_MFMA)
        target_sources(symft_cpp PRIVATE src/simd/simd_avx512.cpp)
        set_source_files_properties(src/simd/simd_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mfma")
        target_compile_definitions(symft_cpp PRIVATE SYMFT_COMPILED_AVX512=1)
    endif()
endif()

add_executable(symft_cli tools/symft_cli.cpp)
target_link_libraries(symft_cli PRIVATE symft_cpp)

add_executable(symft_bench tools/symft_bench.cpp)
target_link_libraries(symft_bench PRIVATE symft_cpp)

add_executable(symft_batch_bench tools/symft_batch_bench.cpp)
target_link_libraries(symft_batch_bench PRIVATE symft_cpp)

add_executable(symft_active_layout_bench tools/symft_active_layout_bench.cpp)
target_link_libraries(symft_active_layout_bench PRIVATE symft_cpp)

add_executable(symft_batch_active_layout_bench tools/symft_batch_active_layout_bench.cpp)
target_link_libraries(symft_batch_active_layout_bench PRIVATE symft_cpp)

add_executable(symft_rate_bench tools/symft_rate_bench.cpp)
target_link_libraries(symft_rate_bench PRIVATE symft_cpp Threads::Threads)

if(SYMFT_CPP_NATIVE)
    target_compile_options(symft_active_layout_bench PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>
    )
    target_compile_options(symft_batch_active_layout_bench PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>
    )
endif()

if(SYMFT_CPP_BUILD_TESTS)
    enable_testing()
    add_executable(symft_cpp_tests tests/symft_tests.cpp)
    target_link_libraries(symft_cpp_tests PRIVATE symft_cpp)
    add_test(NAME symft_cpp_tests COMMAND symft_cpp_tests)
endif()

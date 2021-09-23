cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cuda_first LANGUAGES CXX CUDA)

add_executable(main main.cu)

target_compile_features(main PUBLIC cxx_std_11)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
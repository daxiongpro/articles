cmake_minimum_required(VERSION 3.5)
project(onnx2trt)
set(tensorrt "/home/daxiongpro/1tb/softwares/tensorrt/TensorRT-8.5.1.7")
# set(tensorrt "/home/daxiongpro/1tb/softwares/tensorrt/TensorRT-7.2.2.3")
set(cuda "/usr/local/cuda")

find_package(CUDA REQUIRED)

cuda_add_executable(main onnx2trt.cpp)

# 添加头文件路径
include_directories(
  ${cuda}/include
  ${tensorrt}/include
  ${tensorrt}/samples/common
)

file(GLOB cuda_SO_LIBSNAMES ${cuda}/lib64/*.so)
file(GLOB trt_SO_LIBSNAMES ${tensorrt}/lib/*.so)
target_link_libraries(main ${cuda_SO_LIBSNAMES})
target_link_libraries(main ${trt_SO_LIBSNAMES})

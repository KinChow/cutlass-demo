cmake_minimum_required(VERSION 3.18)
project(cutlass_demo LANGUAGES CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

# 添加CUTLASS路径（假设CUTLASS已安装或克隆到系统中）
# 如果CUTLASS未安装，需要先下载
set(CUTLASS_ENABLE_HEADERS_ONLY ON)

# 尝试查找系统中的CUTLASS
find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    PATHS 
        /usr/local/include
        /usr/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
)

if(NOT CUTLASS_INCLUDE_DIR)
    message(STATUS "CUTLASS not found in system. Will use bundled version.")
    set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass/include)
endif()

# 包含目录
include_directories(${CUTLASS_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(simple_gemm src/simple_gemm.cu)

# 设置CUDA编译选项
set_property(TARGET simple_gemm PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(simple_gemm ${CUDA_LIBRARIES})

# 设置编译器选项
target_compile_options(simple_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        --extended-lambda
    >
)

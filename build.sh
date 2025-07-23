#!/bin/bash

# CUTLASS演示项目构建脚本

echo "开始构建CUTLASS演示项目..."

# 创建构建目录
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# 运行CMake配置
echo "配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译项目
echo "编译项目..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "构建成功！"
    echo "运行程序："
    echo "./build/simple_gemm"
else
    echo "构建失败！"
    exit 1
fi

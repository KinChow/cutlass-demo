# cutlass-demo

这是一个最简单的CUTLASS演示项目，展示了如何使用CUDA进行矩阵乘法计算。

## 项目结构

```
cutlass-demo/
├── CMakeLists.txt          # CMake构建配置
├── src/
│   └── simple_gemm.cu      # 简单的矩阵乘法实现
├── README.md               # 本文件
└── build.sh                # 构建脚本
```

## 依赖要求

- CUDA Toolkit (版本 11.0 或更高)
- CMake (版本 3.18 或更高)
- C++ 编译器 (支持 C++17)
- NVIDIA GPU (计算能力 7.5 或更高)

## 编译和运行

### 方法 1: 使用构建脚本

```bash
./build.sh
```

### 方法 2: 手动编译

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 运行程序

```bash
cd build
./simple_gemm
```

## 程序说明

当前实现包含一个简单的CUDA矩阵乘法kernel，用于演示基础的GPU计算。程序会：

1. 创建大小为 1024x1024 的随机矩阵
2. 在GPU上执行矩阵乘法 C = A * B
3. 测量执行时间和计算性能(GFLOPS)
4. 验证计算结果

## 扩展CUTLASS

要使用真正的CUTLASS库，您需要：

1. 下载CUTLASS库：
```bash
git clone https://github.com/NVIDIA/cutlass.git extern/cutlass
```

2. 修改代码以使用CUTLASS的高性能kernel

## 性能对比

这个基础实现提供了一个性能基准。真正的CUTLASS实现可以显著提高性能，特别是对于大型矩阵和混合精度计算。

## 故障排除

- 确保NVIDIA驱动程序已正确安装
- 检查CUDA Toolkit版本兼容性
- 验证GPU计算能力是否支持

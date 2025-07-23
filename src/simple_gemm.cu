/***************************************************************************************************
 * CUTLASS Tensor Core GEMM性能对比: FP32 vs FP16 vs 混合精度
 *
 * 本示例演示了CUTLASS在不同精度下的GEMM性能，包括Tensor Core加速的实现。
 **************************************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

// 检查CUDA错误的辅助宏
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA错误位置 " << __FILE__ << ":" << __LINE__ << " - "     \
                << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 用于基准比较的简单CUDA GEMM内核
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// 使用行主序布局的简单GEMM内核
__global__ void SimpleGemm_kernel(int M, int N, int K, float alpha,
                                  const float *A, int lda, const float *B,
                                  int ldb, float beta, float *C, int ldc) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float accumulator = 0.0f;
    for (int k = 0; k < K; ++k) {
      accumulator += A[row * lda + k] * B[k * ldb + col];
    }
    C[row * ldc + col] = alpha * accumulator + beta * C[row * ldc + col];
  }
}

/// 简单GEMM计算包装函数
cudaError_t SimpleGemm(int M, int N, int K, float alpha, const float *A,
                       int lda, const float *B, int ldb, float beta, float *C,
                       int ldc) {

  dim3 blockSize(16, 16);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (M + blockSize.y - 1) / blockSize.y);

  SimpleGemm_kernel<<<gridSize, blockSize>>>(M, N, K, alpha, A, lda, B, ldb,
                                             beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// CUTLASS GEMM包装函数 - 多精度支持
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// FP32 CUTLASS GEMM (SIMT核心)
cudaError_t CutlassGemmFP32(int M, int N, int K, float alpha, const float *A,
                            int lda, const float *B, int ldb, float beta,
                            float *C, int ldc) {

  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemmOp = cutlass::gemm::device::Gemm<float,    // ElementA
                                                    RowMajor, // LayoutA
                                                    float,    // ElementB
                                                    RowMajor, // LayoutB
                                                    float,    // ElementC
                                                    RowMajor  // LayoutC
                                                    >;

  CutlassGemmOp gemm_operator;

  CutlassGemmOp::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc},
                                {C, ldc}, {alpha, beta});

  cutlass::Status status = gemm_operator(args);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

/// FP16 CUTLASS GEMM (Tensor Core优化)
cudaError_t CutlassGemmFP16(int M, int N, int K, float alpha,
                            const cutlass::half_t *A, int lda,
                            const cutlass::half_t *B, int ldb, float beta,
                            cutlass::half_t *C, int ldc) {

  using RowMajor = cutlass::layout::RowMajor;

  // 使用Tensor Core优化的FP16 GEMM
  using CutlassGemmOp =
      cutlass::gemm::device::Gemm<cutlass::half_t, // ElementA
                                  RowMajor,        // LayoutA
                                  cutlass::half_t, // ElementB
                                  RowMajor,        // LayoutB
                                  cutlass::half_t, // ElementC
                                  RowMajor,        // LayoutC
                                  float,           // ElementAccumulator
                                  cutlass::arch::OpClassTensorOp, // OpClass
                                  cutlass::arch::Sm75             // ArchTag
                                  >;

  CutlassGemmOp gemm_operator;

  cutlass::half_t alpha_half = cutlass::half_t(alpha);
  cutlass::half_t beta_half = cutlass::half_t(beta);

  CutlassGemmOp::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc},
                                {C, ldc}, {alpha_half, beta_half});

  cutlass::Status status = gemm_operator(args);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

/// 混合精度 CUTLASS GEMM (FP16输入，FP32输出)
cudaError_t CutlassGemmMixed(int M, int N, int K, float alpha,
                             const cutlass::half_t *A, int lda,
                             const cutlass::half_t *B, int ldb, float beta,
                             float *C, int ldc) {

  using RowMajor = cutlass::layout::RowMajor;

  // 混合精度: FP16输入，FP32累积和输出
  using CutlassGemmOp =
      cutlass::gemm::device::Gemm<cutlass::half_t, // ElementA
                                  RowMajor,        // LayoutA
                                  cutlass::half_t, // ElementB
                                  RowMajor,        // LayoutB
                                  float,           // ElementC
                                  RowMajor,        // LayoutC
                                  float,           // ElementAccumulator
                                  cutlass::arch::OpClassTensorOp, // OpClass
                                  cutlass::arch::Sm75             // ArchTag
                                  >;

  CutlassGemmOp gemm_operator;

  CutlassGemmOp::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc},
                                {C, ldc}, {alpha, beta});

  cutlass::Status status = gemm_operator(args);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 矩阵初始化和内存管理工具 - 多精度支持
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// FP32矩阵初始化内核
__global__ void InitializeMatrixFP32_kernel(float *matrix, int rows,
                                            int columns, int seed = 0) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i * columns + j;
    int const k = 16807;
    int const m = 32768;
    float value = float(((offset + seed) * k % m) - m / 2) / float(m / 4);
    matrix[offset] = value;
  }
}

/// FP16矩阵初始化内核
__global__ void InitializeMatrixFP16_kernel(cutlass::half_t *matrix, int rows,
                                            int columns, int seed = 0) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i * columns + j;
    int const k = 16807;
    int const m = 32768;
    float value = float(((offset + seed) * k % m) - m / 2) / float(m / 4);
    matrix[offset] = cutlass::half_t(value);
  }
}

/// FP32矩阵分配和初始化
cudaError_t AllocateMatrixFP32(float **matrix, int rows, int columns,
                               int seed = 0) {
  cudaError_t result;
  size_t sizeof_matrix = sizeof(float) * rows * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "FP32矩阵分配失败: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "FP32矩阵清零失败: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
  InitializeMatrixFP32_kernel<<<grid, block>>>(*matrix, rows, columns, seed);

  return cudaGetLastError();
}

/// FP16矩阵分配和初始化
cudaError_t AllocateMatrixFP16(cutlass::half_t **matrix, int rows, int columns,
                               int seed = 0) {
  cudaError_t result;
  size_t sizeof_matrix = sizeof(cutlass::half_t) * rows * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "FP16矩阵分配失败: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "FP16矩阵清零失败: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);
  InitializeMatrixFP16_kernel<<<grid, block>>>(*matrix, rows, columns, seed);

  return cudaGetLastError();
}

/// FP32到FP16转换内核
__global__ void ConvertFP32ToFP16_kernel(const float *src, cutlass::half_t *dst,
                                         int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = cutlass::half_t(src[idx]);
  }
}

/// FP16到FP32转换内核
__global__ void ConvertFP16ToFP32_kernel(const cutlass::half_t *src, float *dst,
                                         int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float(src[idx]);
  }
}

/// 数据类型转换函数
cudaError_t ConvertMatrixFP32ToFP16(const float *src, cutlass::half_t *dst,
                                    int size) {
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  ConvertFP32ToFP16_kernel<<<grid, block>>>(src, dst, size);
  return cudaGetLastError();
}

cudaError_t ConvertMatrixFP16ToFP32(const cutlass::half_t *src, float *dst,
                                    int size) {
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  ConvertFP16ToFP32_kernel<<<grid, block>>>(src, dst, size);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 性能计时工具
//
///////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmTiming {
  double time_ms;
  double gflops;
  std::string name;

  GemmTiming(const std::string &n) : time_ms(0.0), gflops(0.0), name(n) {}
};

template <typename GemmFunc>
GemmTiming BenchmarkGemm(GemmFunc gemm_func, const std::string &name, int M,
                         int N, int K, int num_iterations = 10) {
  GemmTiming timing(name);
  cudaError_t result;

  // 预热运行
  result = gemm_func();
  if (result != cudaSuccess) {
    std::cerr << name << " 预热失败: " << cudaGetErrorString(result)
              << std::endl;
    return timing;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 正式计时
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; ++i) {
    result = gemm_func();
    if (result != cudaSuccess) {
      std::cerr << name << " 执行失败: " << cudaGetErrorString(result)
                << std::endl;
      return timing;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  timing.time_ms = duration.count() / 1000.0 / num_iterations;
  timing.gflops = (2.0 * M * N * K) / (timing.time_ms * 1e-3) / 1e9;

  return timing;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 结果验证工具 - 改进的多精度验证
//
///////////////////////////////////////////////////////////////////////////////////////////////////

bool VerifyResults(const std::vector<float> &result1,
                   const std::vector<float> &result2, const std::string &name1,
                   const std::string &name2, double tolerance = 1e-2) {

  if (result1.size() != result2.size()) {
    std::cerr << "矩阵大小不匹配!" << std::endl;
    return false;
  }

  double max_diff = 0.0;
  double avg_diff = 0.0;
  double relative_error_sum = 0.0;
  int error_count = 0;
  int significant_error_count = 0;

  // 检测是否涉及混合精度或FP16计算
  bool is_mixed_precision = (name1.find("混合精度") != std::string::npos ||
                             name2.find("混合精度") != std::string::npos ||
                             name1.find("FP16") != std::string::npos ||
                             name2.find("FP16") != std::string::npos);

  for (size_t i = 0; i < result1.size(); ++i) {
    double val1 = result1[i];
    double val2 = result2[i];
    double diff = std::abs(val1 - val2);

    max_diff = std::max(max_diff, diff);
    avg_diff += diff;

    // 计算相对误差
    double max_val = std::max(std::abs(val1), std::abs(val2));
    if (max_val > 1e-8) {
      double relative_error = diff / max_val;
      relative_error_sum += relative_error;

      // 使用自适应容差
      double adaptive_tolerance = tolerance;
      if (is_mixed_precision) {
        // 对于混合精度，使用相对误差阈值
        adaptive_tolerance = std::max(tolerance, max_val * 0.02); // 2%相对误差
      }

      if (diff > adaptive_tolerance) {
        error_count++;
        // 检查是否为显著误差（>5%相对误差）
        if (relative_error > 0.05) {
          significant_error_count++;
        }
      }
    } else {
      // 对于接近零的值，使用绝对容差
      if (diff > tolerance) {
        error_count++;
      }
    }
  }

  avg_diff /= result1.size();
  double avg_relative_error = relative_error_sum / result1.size();

  std::cout << "\n=== 结果验证: " << name1 << " vs " << name2
            << " ===" << std::endl;
  std::cout << "最大差值: " << std::scientific << std::setprecision(3)
            << max_diff << std::endl;
  std::cout << "平均差值: " << std::scientific << std::setprecision(3)
            << avg_diff << std::endl;
  std::cout << "平均相对误差: " << std::fixed << std::setprecision(4)
            << avg_relative_error * 100 << "%" << std::endl;
  std::cout << "超出容差元素: " << error_count << " / " << result1.size()
            << " (" << std::fixed << std::setprecision(2)
            << (error_count * 100.0 / result1.size()) << "%)" << std::endl;
  std::cout << "显著误差元素: " << significant_error_count << " / "
            << result1.size() << " (" << std::fixed << std::setprecision(2)
            << (significant_error_count * 100.0 / result1.size()) << "%)"
            << std::endl;

  // 调整验证标准
  if (is_mixed_precision) {
    // 对于涉及FP16的计算，使用更宽松的标准
    if (avg_relative_error < 0.05 &&
        significant_error_count < result1.size() * 0.1) {
      std::cout << "✓ 结果验证通过！(混合精度计算，允许一定数值差异)"
                << std::endl;
      return true;
    } else if (avg_relative_error < 0.1 &&
               significant_error_count < result1.size() * 0.2) {
      std::cout << "⚠ 结果可接受（存在预期的FP16精度损失）" << std::endl;
      return true;
    } else {
      std::cout << "✗ 相对误差过大，可能存在计算错误" << std::endl;
      return false;
    }
  } else {
    // 对于FP32计算，使用较严格的标准
    if (error_count < result1.size() * 0.01) {
      std::cout << "✓ 结果验证通过！" << std::endl;
      return true;
    } else {
      std::cout << "✗ 结果验证失败，存在较大数值差异" << std::endl;
      return false;
    }
  }
}

// 专门的FP16结果验证函数
bool VerifyResultsFP16(const std::vector<float> &fp32_result,
                       const std::vector<float> &fp16_result,
                       const std::string &name1, const std::string &name2) {

  if (fp32_result.size() != fp16_result.size()) {
    std::cerr << "矩阵大小不匹配!" << std::endl;
    return false;
  }

  double max_diff = 0.0;
  double avg_diff = 0.0;
  double relative_error_sum = 0.0;
  int large_error_count = 0;

  // 统计不同误差范围的元素数量
  int error_ranges[5] = {0}; // <1%, 1-5%, 5-10%, 10-20%, >20%

  for (size_t i = 0; i < fp32_result.size(); ++i) {
    double fp32_val = fp32_result[i];
    double fp16_val = fp16_result[i];
    double diff = std::abs(fp32_val - fp16_val);

    max_diff = std::max(max_diff, diff);
    avg_diff += diff;

    if (std::abs(fp32_val) > 1e-6) {
      double relative_error = diff / std::abs(fp32_val);
      relative_error_sum += relative_error;

      // 分类相对误差
      if (relative_error < 0.01)
        error_ranges[0]++;
      else if (relative_error < 0.05)
        error_ranges[1]++;
      else if (relative_error < 0.1)
        error_ranges[2]++;
      else if (relative_error < 0.2)
        error_ranges[3]++;
      else {
        error_ranges[4]++;
        large_error_count++;
      }
    }
  }

  avg_diff /= fp32_result.size();
  double avg_relative_error = relative_error_sum / fp32_result.size();

  std::cout << "\n=== FP16精度验证: " << name1 << " vs " << name2
            << " ===" << std::endl;
  std::cout << "最大差值: " << std::scientific << std::setprecision(3)
            << max_diff << std::endl;
  std::cout << "平均差值: " << std::scientific << std::setprecision(3)
            << avg_diff << std::endl;
  std::cout << "平均相对误差: " << std::fixed << std::setprecision(4)
            << avg_relative_error * 100 << "%" << std::endl;

  std::cout << "\n相对误差分布:" << std::endl;
  std::cout << "  < 1%:     " << error_ranges[0] << " (" << std::fixed
            << std::setprecision(1)
            << (error_ranges[0] * 100.0 / fp32_result.size()) << "%)"
            << std::endl;
  std::cout << "  1% - 5%:  " << error_ranges[1] << " ("
            << (error_ranges[1] * 100.0 / fp32_result.size()) << "%)"
            << std::endl;
  std::cout << "  5% - 10%: " << error_ranges[2] << " ("
            << (error_ranges[2] * 100.0 / fp32_result.size()) << "%)"
            << std::endl;
  std::cout << "  10% - 20%:" << error_ranges[3] << " ("
            << (error_ranges[3] * 100.0 / fp32_result.size()) << "%)"
            << std::endl;
  std::cout << "  > 20%:    " << error_ranges[4] << " ("
            << (error_ranges[4] * 100.0 / fp32_result.size()) << "%)"
            << std::endl;

  // FP16验证标准：平均相对误差<5%，大误差元素<1%
  if (avg_relative_error < 0.05 &&
      large_error_count < fp32_result.size() * 0.01) {
    std::cout << "✓ FP16精度验证通过！" << std::endl;
    return true;
  } else if (avg_relative_error < 0.1 &&
             large_error_count < fp32_result.size() * 0.05) {
    std::cout << "⚠ FP16精度可接受（存在预期的精度损失）" << std::endl;
    return true;
  } else {
    std::cout << "✗ FP16精度验证失败，误差过大" << std::endl;
    return false;
  }
}

void DisplayPartialResults(const std::vector<float> &matrix,
                           const std::string &name, int rows, int cols,
                           int display_size = 5) {
  std::cout << "\n"
            << name << " (前" << display_size << "x" << display_size
            << "个元素):" << std::endl;
  for (int i = 0; i < std::min(display_size, rows); ++i) {
    for (int j = 0; j < std::min(display_size, cols); ++j) {
      std::cout << std::fixed << std::setprecision(3) << matrix[i * cols + j]
                << " ";
    }
    std::cout << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// GPU设备信息和理论性能计算
//
///////////////////////////////////////////////////////////////////////////////////////////////////

struct TheoreticalPerformance {
  double peak_fp32_gflops;
  double peak_tensor_gflops;
  double memory_bandwidth_gb;
  bool has_tensor_cores;

  TheoreticalPerformance()
      : peak_fp32_gflops(0), peak_tensor_gflops(0), memory_bandwidth_gb(0),
        has_tensor_cores(false) {}
};

TheoreticalPerformance CalculateTheoreticalPerformance() {
  TheoreticalPerformance perf;
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  int clockRate = prop.clockRate;
  int smCount = prop.multiProcessorCount;

  // 估算CUDA核心数
  int coresPerSM = 0;
  if (prop.major == 9) {
    coresPerSM = 128;
    perf.has_tensor_cores = true;
  } else if (prop.major == 8) {
    coresPerSM = (prop.minor >= 6) ? 128 : 64;
    perf.has_tensor_cores = true;
  } else if (prop.major == 7) {
    coresPerSM = 64;
    perf.has_tensor_cores = true;
  } else if (prop.major == 6) {
    coresPerSM = (prop.minor == 1) ? 128 : 64;
  } else {
    coresPerSM = 32;
  }

  perf.peak_fp32_gflops = (double)smCount * coresPerSM * clockRate * 2.0 / 1e6;

  // Tensor Core理论性能估算
  if (perf.has_tensor_cores) {
    if (prop.major == 9) {
      perf.peak_tensor_gflops = perf.peak_fp32_gflops * 20.0; // Ada Lovelace
    } else if (prop.major == 8) {
      perf.peak_tensor_gflops =
          perf.peak_fp32_gflops * ((prop.minor >= 6) ? 15.0 : 10.0); // Ampere
    } else if (prop.major == 7) {
      perf.peak_tensor_gflops = perf.peak_fp32_gflops * 8.0; // Turing/Volta
    }
  }

  // 内存带宽计算
  int memBusWidth = prop.memoryBusWidth;
  int memClockRate = prop.memoryClockRate;

  if (memBusWidth > 0 && memClockRate > 0) {
    perf.memory_bandwidth_gb =
        (double)memClockRate * memBusWidth * 2.0 / 8.0 / 1e6;
  } else {
    perf.memory_bandwidth_gb = 500.0;
  }

  return perf;
}

void PrintDeviceInfo() {
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  std::cout << "\n=== GPU设备信息 ===" << std::endl;
  std::cout << "设备名称: " << prop.name << std::endl;
  std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "全局内存: " << prop.totalGlobalMem / (1024 * 1024 * 1024)
            << " GB" << std::endl;
  std::cout << "多处理器数量: " << prop.multiProcessorCount << std::endl;
  std::cout << "基础时钟频率: " << prop.clockRate / 1000 << " MHz" << std::endl;

  if (prop.major >= 7) {
    std::cout << "Tensor Core支持: ✓ (第" << (prop.major - 6) << "代)"
              << std::endl;
  } else {
    std::cout << "Tensor Core支持: ✗ (需要SM_70+)" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 主测试函数
//
///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t TestGemmComparison(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  std::cout << "CUTLASS多精度GEMM性能对比 (包含Tensor Core加速)" << std::endl;
  std::cout << "矩阵尺寸: A(" << M << "x" << K << ") * B(" << K << "x" << N
            << ") = C(" << M << "x" << N << ")" << std::endl;
  std::cout << "Alpha: " << alpha << ", Beta: " << beta << std::endl;

  PrintDeviceInfo();

  // 检查Tensor Core支持
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  bool tensor_core_supported = (prop.major >= 7);
  if (!tensor_core_supported) {
    std::cout << "\n警告: 当前GPU不支持Tensor Core，将跳过FP16测试"
              << std::endl;
  }

  TheoreticalPerformance theoretical = CalculateTheoreticalPerformance();

  std::cout << "\n=== 理论性能分析 ===" << std::endl;
  std::cout << "FP32理论峰值: " << std::fixed << std::setprecision(1)
            << theoretical.peak_fp32_gflops << " GFLOPS" << std::endl;
  if (theoretical.has_tensor_cores) {
    std::cout << "Tensor Core理论峰值: " << std::fixed << std::setprecision(1)
              << theoretical.peak_tensor_gflops << " GFLOPS (FP16)"
              << std::endl;
  }
  std::cout << "理论内存带宽: " << std::fixed << std::setprecision(1)
            << theoretical.memory_bandwidth_gb << " GB/s" << std::endl;

  // 计算前导维度
  int lda = K;
  int ldb = N;
  int ldc = N;

  // 分配FP32矩阵
  float *A_fp32, *B_fp32, *C_simple, *C_cutlass_fp32, *C_mixed;

  result = AllocateMatrixFP32(&A_fp32, M, K, 0);
  if (result != cudaSuccess)
    return result;

  result = AllocateMatrixFP32(&B_fp32, K, N, 17);
  if (result != cudaSuccess) {
    cudaFree(A_fp32);
    return result;
  }

  result = AllocateMatrixFP32(&C_simple, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A_fp32);
    cudaFree(B_fp32);
    return result;
  }

  result = AllocateMatrixFP32(&C_cutlass_fp32, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A_fp32);
    cudaFree(B_fp32);
    cudaFree(C_simple);
    return result;
  }

  result = AllocateMatrixFP32(&C_mixed, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A_fp32);
    cudaFree(B_fp32);
    cudaFree(C_simple);
    cudaFree(C_cutlass_fp32);
    return result;
  }

  // 分配FP16矩阵 (如果支持Tensor Core)
  cutlass::half_t *A_fp16 = nullptr, *B_fp16 = nullptr, *C_fp16 = nullptr;

  if (tensor_core_supported) {
    result = AllocateMatrixFP16(&A_fp16, M, K, 0);
    if (result != cudaSuccess) {
      cudaFree(A_fp32);
      cudaFree(B_fp32);
      cudaFree(C_simple);
      cudaFree(C_cutlass_fp32);
      cudaFree(C_mixed);
      return result;
    }

    result = AllocateMatrixFP16(&B_fp16, K, N, 17);
    if (result != cudaSuccess) {
      cudaFree(A_fp32);
      cudaFree(B_fp32);
      cudaFree(C_simple);
      cudaFree(C_cutlass_fp32);
      cudaFree(C_mixed);
      cudaFree(A_fp16);
      return result;
    }

    result = AllocateMatrixFP16(&C_fp16, M, N, 101);
    if (result != cudaSuccess) {
      cudaFree(A_fp32);
      cudaFree(B_fp32);
      cudaFree(C_simple);
      cudaFree(C_cutlass_fp32);
      cudaFree(C_mixed);
      cudaFree(A_fp16);
      cudaFree(B_fp16);
      return result;
    }
  }

  // 确保所有输出矩阵初始值相同
  size_t sizeof_C_fp32 = sizeof(float) * M * N;
  CUDA_CHECK(cudaMemcpy(C_cutlass_fp32, C_simple, sizeof_C_fp32,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(
      cudaMemcpy(C_mixed, C_simple, sizeof_C_fp32, cudaMemcpyDeviceToDevice));

  std::cout << "\n=== 性能测试 ===" << std::endl;

  // 测试1: 简单CUDA GEMM (FP32)
  auto timing_simple = BenchmarkGemm(
      [&]() {
        return SimpleGemm(M, N, K, alpha, A_fp32, lda, B_fp32, ldb, beta,
                          C_simple, ldc);
      },
      "简单CUDA GEMM (FP32)", M, N, K);

  std::cout << timing_simple.name << " - 时间: " << timing_simple.time_ms
            << " ms, 性能: " << timing_simple.gflops << " GFLOPS" << std::endl;

  // 测试2: CUTLASS FP32 GEMM
  auto timing_cutlass_fp32 = BenchmarkGemm(
      [&]() {
        return CutlassGemmFP32(M, N, K, alpha, A_fp32, lda, B_fp32, ldb, beta,
                               C_cutlass_fp32, ldc);
      },
      "CUTLASS GEMM (FP32)", M, N, K);

  std::cout << timing_cutlass_fp32.name
            << " - 时间: " << timing_cutlass_fp32.time_ms
            << " ms, 性能: " << timing_cutlass_fp32.gflops << " GFLOPS"
            << std::endl;

  // 测试3和4: Tensor Core测试 (如果支持)
  GemmTiming timing_cutlass_fp16("CUTLASS GEMM (FP16 Tensor Core)");
  GemmTiming timing_mixed("CUTLASS GEMM (混合精度)");

  if (tensor_core_supported) {
    timing_cutlass_fp16 = BenchmarkGemm(
        [&]() {
          return CutlassGemmFP16(M, N, K, alpha, A_fp16, lda, B_fp16, ldb, beta,
                                 C_fp16, ldc);
        },
        "CUTLASS GEMM (FP16 Tensor Core)", M, N, K);

    std::cout << timing_cutlass_fp16.name
              << " - 时间: " << timing_cutlass_fp16.time_ms
              << " ms, 性能: " << timing_cutlass_fp16.gflops << " GFLOPS"
              << std::endl;

    timing_mixed = BenchmarkGemm(
        [&]() {
          return CutlassGemmMixed(M, N, K, alpha, A_fp16, lda, B_fp16, ldb,
                                  beta, C_mixed, ldc);
        },
        "CUTLASS GEMM (混合精度)", M, N, K);

    std::cout << timing_mixed.name << " - 时间: " << timing_mixed.time_ms
              << " ms, 性能: " << timing_mixed.gflops << " GFLOPS" << std::endl;
  }

  // 性能对比分析
  std::cout << "\n=== 性能对比分析 ===" << std::endl;

  if (timing_simple.gflops > 0) {
    double speedup_fp32 = timing_cutlass_fp32.gflops / timing_simple.gflops;
    std::cout << "CUTLASS FP32加速比: " << std::fixed << std::setprecision(2)
              << speedup_fp32 << "x" << std::endl;

    if (tensor_core_supported && timing_cutlass_fp16.gflops > 0) {
      double speedup_fp16 = timing_cutlass_fp16.gflops / timing_simple.gflops;
      double speedup_mixed = timing_mixed.gflops / timing_simple.gflops;

      std::cout << "Tensor Core FP16加速比: " << std::fixed
                << std::setprecision(2) << speedup_fp16 << "x" << std::endl;
      std::cout << "混合精度加速比: " << std::fixed << std::setprecision(2)
                << speedup_mixed << "x" << std::endl;

      // Tensor Core vs FP32 CUTLASS对比
      double tensor_vs_fp32 =
          timing_cutlass_fp16.gflops / timing_cutlass_fp32.gflops;
      std::cout << "Tensor Core vs CUTLASS FP32: " << std::fixed
                << std::setprecision(2) << tensor_vs_fp32 << "x" << std::endl;
    }
  }

  // 效率分析
  std::cout << "\n=== 效率分析 ===" << std::endl;

  double fp32_efficiency =
      (timing_cutlass_fp32.gflops / theoretical.peak_fp32_gflops) * 100.0;
  std::cout << "CUTLASS FP32效率: " << std::fixed << std::setprecision(1)
            << fp32_efficiency << "%" << std::endl;

  if (tensor_core_supported && theoretical.has_tensor_cores &&
      timing_cutlass_fp16.gflops > 0) {
    double tensor_efficiency =
        (timing_cutlass_fp16.gflops / theoretical.peak_tensor_gflops) * 100.0;
    std::cout << "Tensor Core效率: " << std::fixed << std::setprecision(1)
              << tensor_efficiency << "%" << std::endl;
  }

  // 结果验证 - 使用改进的验证函数
  std::cout << "\n=== 结果验证 ===" << std::endl;

  // 复制结果到主机进行验证
  std::vector<float> host_simple(M * N);
  std::vector<float> host_cutlass_fp32(M * N);
  std::vector<float> host_mixed(M * N);

  CUDA_CHECK(cudaMemcpy(host_simple.data(), C_simple, sizeof_C_fp32,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_cutlass_fp32.data(), C_cutlass_fp32, sizeof_C_fp32,
                        cudaMemcpyDeviceToHost));

  // FP32精度验证
  bool verification_passed = VerifyResults(
      host_simple, host_cutlass_fp32, "简单CUDA GEMM", "CUTLASS FP32", 1e-4);

  // FP16/混合精度验证
  if (tensor_core_supported && timing_mixed.gflops > 0) {
    CUDA_CHECK(cudaMemcpy(host_mixed.data(), C_mixed, sizeof_C_fp32,
                          cudaMemcpyDeviceToHost));

    // 使用专门的混合精度验证函数
    bool mixed_precision_passed = VerifyResults(
        host_simple, host_mixed, "简单CUDA GEMM", "混合精度GEMM", 1e-2);
    verification_passed = verification_passed && mixed_precision_passed;

    // 如果有FP16纯精度结果，也进行验证
    if (timing_cutlass_fp16.gflops > 0) {
      // 将FP16结果转换为FP32进行比较
      std::vector<float> host_fp16_converted(M * N);
      float *temp_fp32;
      CUDA_CHECK(cudaMalloc(&temp_fp32, sizeof(float) * M * N));
      CUDA_CHECK(ConvertMatrixFP16ToFP32(C_fp16, temp_fp32, M * N));
      CUDA_CHECK(cudaMemcpy(host_fp16_converted.data(), temp_fp32,
                            sizeof(float) * M * N, cudaMemcpyDeviceToHost));
      cudaFree(temp_fp32);

      bool fp16_passed = VerifyResultsFP16(host_simple, host_fp16_converted,
                                           "简单CUDA GEMM", "FP16 Tensor Core");
      verification_passed = verification_passed && fp16_passed;
    }
  }

  // 显示部分结果
  std::cout << "\n=== 部分结果展示 ===" << std::endl;
  DisplayPartialResults(host_simple, "简单CUDA GEMM结果", M, N);
  DisplayPartialResults(host_cutlass_fp32, "CUTLASS FP32结果", M, N);
  if (tensor_core_supported && timing_mixed.gflops > 0) {
    DisplayPartialResults(host_mixed, "混合精度结果", M, N);
  }

  // 优化建议
  std::cout << "\n=== 优化建议 ===" << std::endl;
  if (tensor_core_supported) {
    if (timing_cutlass_fp16.gflops > timing_cutlass_fp32.gflops * 1.5) {
      std::cout << "• Tensor Core显著提升性能，建议在精度允许的情况下使用FP16"
                << std::endl;
    }
    if (timing_mixed.gflops > timing_cutlass_fp32.gflops * 1.2) {
      std::cout << "• 混合精度在保持FP32输出精度的同时获得了性能提升"
                << std::endl;
    }
    std::cout << "• 确保矩阵维度为8的倍数以充分利用Tensor Core" << std::endl;
    std::cout << "• 考虑使用更优化的CUTLASS Tile配置" << std::endl;
  } else {
    std::cout << "• 当前GPU不支持Tensor Core，建议升级到SM_70+架构"
              << std::endl;
    std::cout << "• 优化内存访问模式和线程块配置" << std::endl;
  }

  // 清理内存
  cudaFree(A_fp32);
  cudaFree(B_fp32);
  cudaFree(C_simple);
  cudaFree(C_cutlass_fp32);
  cudaFree(C_mixed);

  if (tensor_core_supported) {
    cudaFree(A_fp16);
    cudaFree(B_fp16);
    cudaFree(C_fp16);
  }

  return verification_passed ? cudaSuccess : cudaErrorUnknown;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 程序入口点
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void PrintUsage(const char *program_name) {
  std::cout << "\n使用方法: " << program_name << " [M] [N] [K] [alpha] [beta]\n"
            << std::endl;
  std::cout << "参数说明:" << std::endl;
  std::cout << "  M      : 矩阵A的行数 (默认: 1024)" << std::endl;
  std::cout << "  N      : 矩阵B的列数 (默认: 1024)" << std::endl;
  std::cout << "  K      : 矩阵A的列数/矩阵B的行数 (默认: 1024)" << std::endl;
  std::cout << "  alpha  : 标量乘子alpha (默认: 1.0)" << std::endl;
  std::cout << "  beta   : 标量乘子beta (默认: 0.0)" << std::endl;
  std::cout << "\n功能:" << std::endl;
  std::cout << "  • 对比FP32 SIMT核心 vs Tensor Core性能" << std::endl;
  std::cout << "  • 测试FP16和混合精度GEMM" << std::endl;
  std::cout << "  • 自动检测GPU Tensor Core支持" << std::endl;
  std::cout << "  • 提供详细的性能分析和优化建议" << std::endl;
  std::cout << "\n注意:" << std::endl;
  std::cout << "  - Tensor Core需要SM_70+架构支持" << std::endl;
  std::cout << "  - 建议矩阵维度为8的倍数以获得最佳Tensor Core性能"
            << std::endl;
  std::cout << "  - FP16计算可能存在精度损失" << std::endl;
  std::cout << std::endl;
}

int main(int argc, const char *argv[]) {
  if (argc > 1) {
    std::string arg1(argv[1]);
    if (arg1 == "--help" || arg1 == "-h" || arg1 == "help") {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  // 解析命令行参数
  int problem[3] = {1024, 1024, 1024};

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(argv[i]);
    int value;
    if (!(ss >> value) || value <= 0) {
      std::cerr << "错误: 无效的矩阵维度参数 '" << argv[i] << "'" << std::endl;
      PrintUsage(argv[0]);
      return -1;
    }
    problem[i - 1] = value;
  }

  float scalars[2] = {1.0f, 0.0f};

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(argv[i]);
    float value;
    if (!(ss >> value)) {
      std::cerr << "错误: 无效的标量参数 '" << argv[i] << "'" << std::endl;
      PrintUsage(argv[0]);
      return -1;
    }
    scalars[i - 4] = value;
  }

  // 检查矩阵维度是否适合Tensor Core
  if (problem[0] % 8 != 0 || problem[1] % 8 != 0 || problem[2] % 8 != 0) {
    std::cout << "建议: 为获得最佳Tensor Core性能，建议矩阵维度为8的倍数"
              << std::endl;
  }

  std::cout << "=== Tensor Core GEMM性能测试 ===" << std::endl;
  std::cout << "矩阵维度: M=" << problem[0] << ", N=" << problem[1]
            << ", K=" << problem[2] << std::endl;
  std::cout << "标量参数: alpha=" << scalars[0] << ", beta=" << scalars[1]
            << std::endl;

  cudaError_t result = TestGemmComparison(problem[0], problem[1], problem[2],
                                          scalars[0], scalars[1]);

  if (result == cudaSuccess) {
    std::cout << "\n✓ 所有测试通过！" << std::endl;
  } else {
    std::cout << "\n✗ 测试失败！" << std::endl;
  }

  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
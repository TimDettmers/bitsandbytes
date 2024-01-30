// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <float.h>
#include <ops.cuh>

#ifndef kernels
#define kernels

//template <int QUANT_TYPE, typename INP_TYPE, typename COMP_TYPE, typename OUT_TYPE>__global__ void kMatmul_inference_4bit(INP_TYPE *A, unsigned char *B, OUT_TYPE *out, int lda, int ldb, int rowsA, int colsA, int colsB);

template<typename T>__global__ void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n);

__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n);
__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n);

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE> __global__ void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);
template<typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE> __global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPreconditionOptimizer32bit2State(T* g, T* p,
                float* state1, float* state2, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n);

template<typename T, int OPTIMIZER>
__global__ void kOptimizer32bit2State(T* g, T* p,
                float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPreconditionOptimizer32bit1State(T* g, T* p,
                float* state1, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n);

template<typename T, int OPTIMIZER>
__global__ void kOptimizer32bit1State(T* g, T* p,
                float* state1,  float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

template<typename T, int OPTIMIZER>
__global__ void
kPreconditionOptimizerStatic8bit1State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1,
                float* max1, float* new_max1,
                const float weight_decay,
                const float gnorm_scale, const int n);


template<typename T, int OPTIMIZER>
__global__ void
kOptimizerStatic8bit1State(T* p, T* const g, unsigned char* state1,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* max1, float* new_max1,
                float weight_decay, const float gnorm_scale, const int n);



template<typename T, int OPTIMIZER>
__global__ void
kPreconditionOptimizerStatic8bit2State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                const float gnorm_scale, const int n);


template<typename T, int OPTIMIZER>
__global__ void
kOptimizerStatic8bit2State(T* p, T* const g, unsigned char* state1, unsigned char* state2,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, const float gnorm_scale, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH> __global__ void kOptimizerStatic8bit2StateBlockwise(
		T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2,
                const float beta1, const float beta2, const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, const bool skip_zeros, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH> __global__ void kOptimizerStatic8bit1StateBlockwise(
		T* p, T* __restrict__ const g, unsigned char* state1,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* absmax1,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n);


template<typename T, int BLOCK_SIZE, int NUM_VALS> __global__ void kPercentileClipping(T * __restrict__ g, float *gnorm_vec, int step, const int n);


__global__ void kHistogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, const int maxidx1, const int n);


template <typename T, int SPMM_ITEMS, int BITS> __global__ void kspmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, T *B, half *out,  float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB);

template <int ITEMS_PER_THREAD, int SUBTILE_ROWS, int THREADS>__global__ void kdequant_mm_int32_fp16(
  int *__restrict__ const A, float *__restrict__ const rowStats, float *__restrict__ const colStats,
  half *out, float* newRowStats, float* newcolStats, half * __restrict__ const bias, const int numRows, const int numCols, const int tileCols, const int n);

template<typename T, int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int SPARSE_DECOMP> __global__ void kgetColRowStats(T * __restrict__ A, float *rowStats, float *colStats, int * nnz_count_row, float nnz_threshold, int rows, int cols, int tiledRows, int tiledCols);
template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int SPARSE_DECOMP> __global__ void kDoubleRowColQuant(half *__restrict__ const A, float *__restrict__ const rowStats, float * __restrict__ const colStats, int8_t *out_col_normed, int8_t *out_row_normed, int *rowidx, int *colidx, half *val, int * __restrict__ nnz_block_ptr, float threshold, int rows, int cols, int tiledCols);

template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int TRANSPOSE, int FORMAT> __global__ void kTransformRowToFormat(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols);

template <int FORMAT> __global__ void kExtractOutliers(char *A, int *idx, char *out, int idx_size, int rowsA, int colsA, int tiledRowsA, int tiledColsA);

template <typename T, int BITS, int THREADS> __global__ void gemm_device(int M, int N, int K, T * __restrict__ const A,  T* B,  T * out,  int lda, int ldb, int ldc);
template <typename T, int THREADS> __global__ void kgemm_4bit_inference(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize);
template <typename T, int THREADS, int BITS> __global__ void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize);

template <typename T, int FUNC> __global__ void kfunc(T *A, T *B, T value, long n);

#endif

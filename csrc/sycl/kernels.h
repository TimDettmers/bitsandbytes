// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>
#include "ops.h"

#ifndef kernels
#define kernels

#pragma once

//================typedefs===================================

typedef sycl::local_accessor<uint8_t, 1> sycl_la;
typedef sycl::accessor<int, 1> sycl_dacc;
typedef sycl::accessor<float, 1> sycl_dacc_float;
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;
typedef sycl::accessor<char, 1> sycl_dacc_char;

//===========================================================
//template <int QUANT_TYPE, typename INP_TYPE, typename COMP_TYPE, typename OUT_TYPE>__global__ void kMatmul_inference_4bit(INP_TYPE *A, unsigned char *B, OUT_TYPE *out, int lda, int ldb, int rowsA, int colsA, int colsB);

template<typename T> extern SYCL_EXTERNAL void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_float &dacc_code);

extern SYCL_EXTERNAL void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n,
               const sycl::nd_item<3> &item_ct1, float* smem_code, const sycl_la &tacc, const sycl_dacc_float &dacc_A,
               const sycl_dacc_uc &dacc_out, const sycl_dacc_float &dacc_code);
               
extern SYCL_EXTERNAL void kDequantize(float *code, unsigned char *A, float *out, const int n,
                 const sycl::nd_item<3> &item_ct1, float *smem_code);

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE> extern SYCL_EXTERNAL void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n,
                        const sycl::nd_item<3> &item_ct1,
                        float *smem_code,
                        float *smem_absmax_value,
                        const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_A,
                        const sycl_dacc_float &dacc_rand, const sycl_dacc_uc &dacc_out,
                        const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);
                        
//=========================k-dequant blockwise ======================                        
template<typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE> extern SYCL_EXTERNAL void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl_dacc_uc &dacc_A,const sycl::accessor<T, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

//====================32 bit headers=============================

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
extern SYCL_EXTERNAL void kPreconditionOptimizer32bit2State(T* g, T* p,
                float* state1, float* state2, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl_dacc_float &dacc_state1,const sycl_dacc_float &dacc_state2,const sycl::accessor<T, 1> &dacc_g, const sycl_dacc_float &dacc_unorm);

template<typename T, int OPTIMIZER>
extern SYCL_EXTERNAL void kOptimizer32bit2State(T* g, T* p,
                float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_g,const sycl::accessor<T, 1> &dacc_p,const sycl_dacc_float &dacc_state1,const  sycl_dacc_float &dacc_state2, const sycl_dacc_float &dacc_unorm);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
extern SYCL_EXTERNAL void kPreconditionOptimizer32bit1State(T* g, T* p,
                float* state1, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_g,const sycl_dacc_float &dacc_state1,
                const sycl_dacc_float &dacc_unorm);

template<typename T, int OPTIMIZER>
extern SYCL_EXTERNAL void kOptimizer32bit1State(T* g, T* p,
                float* state1,  float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_g,const sycl::accessor<T, 1> &dacc_p,const
                sycl_dacc_float &dacc_state1, const sycl_dacc_float &dacc_unorm);


//==============================8 bit headers==========================
template<typename T, int OPTIMIZER>
extern SYCL_EXTERNAL void
kPreconditionOptimizerStatic8bit1State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1,
                float* max1, float* new_max1,
                const float weight_decay,
                const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,
                float *smem_quantiles1,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, const sycl_dacc_uc &dacc_state1,
                const sycl_dacc_float &dacc_unorm, const sycl_dacc_float &dacc_quantiles1, 
                const sycl_dacc_float &dacc_max1, const sycl_dacc_float &dacc_new_max1);


template<typename T, int OPTIMIZER>
extern SYCL_EXTERNAL void
kOptimizerStatic8bit1State(T* p, T* const g, unsigned char* state1,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* max1, float* new_max1,
                float weight_decay, const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1, float *smem_quantiles1,
                const sycl_la &tacc,
                const sycl::accessor<T, 1> &dacc_g, const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1,
                const sycl_dacc_float &dacc_unorm, const sycl_dacc_float &dacc_quantiles1, 
                const sycl_dacc_float &dacc_max1, const sycl_dacc_float &dacc_new_max1);



template<typename T, int OPTIMIZER>
extern SYCL_EXTERNAL void
kPreconditionOptimizerStatic8bit2State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1, 
                float *smem_quantiles1, float *smem_quantiles2,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2,
                const sycl_dacc_float &dacc_unorm, const sycl_dacc_float &dacc_quantiles1, const sycl_dacc_float &dacc_quantiles2,
                const sycl_dacc_float &dacc_max1, const sycl_dacc_float &dacc_max2, const sycl_dacc_float &dacc_new_max1, 
                const sycl_dacc_float &dacc_new_max2);


template<typename T, int OPTIMIZER>
extern SYCL_EXTERNAL void
kOptimizerStatic8bit2State(T* p, T* const g, unsigned char* state1, unsigned char* state2,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1, float *smem_quantiles1,
                float *smem_quantiles2,const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2, const sycl_dacc_float &dacc_unorm,
                const sycl_dacc_float &dacc_quantiles1, const sycl_dacc_float &dacc_quantiles2,
                const sycl_dacc_float &dacc_max1, const sycl_dacc_float &dacc_max2, const sycl_dacc_float &dacc_new_max1, 
                const sycl_dacc_float &dacc_new_max2);
                
//====================8 bit blockwise=========================

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH> extern SYCL_EXTERNAL void kOptimizerStatic8bit2StateBlockwise(
		T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* absmax1, float* absmax2,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,
                sycl::local_accessor<float, 2> smem_quantiles1,
                sycl::local_accessor<float, 2> smem_quantiles2,
                float *smem_exchange1, float *smem_exchange2,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g,
                const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2,
                const sycl_dacc_float &dacc_quantiles1, const sycl_dacc_float &dacc_quantiles2,
                const sycl_dacc_float &dacc_absmax1, const sycl_dacc_float &dacc_absmax2);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH> extern SYCL_EXTERNAL void kOptimizerStatic8bit1StateBlockwise(
		T* p, T* __restrict__ const g, unsigned char* state1,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* absmax1,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,
                sycl::local_accessor<float, 2> smem_quantiles1,
                float *smem_exchange1,
                const sycl_la &tacc,
                const sycl::accessor<T, 1> &dacc_g,
                const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1,
                const sycl_dacc_float &dacc_quantiles1,
                const sycl_dacc_float &dacc_absmax1);

//=======================percentile clipping============================
template<typename T, int BLOCK_SIZE, int NUM_VALS> extern SYCL_EXTERNAL void kPercentileClipping(T * __restrict__ g, float *gnorm_vec, int step, const int n,const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, float *dacc_gnorm_vec);


//===============histogram========================

extern SYCL_EXTERNAL void kHistogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, const int maxidx1, const int n,
                            const sycl::nd_item<3> &item_ct1, const sycl_dacc_float &dacc_histogram, const sycl_dacc &dacc_index1, 
                            const sycl_dacc &dacc_index2, const sycl_dacc_float &dacc_src);

//====================spm=======================
template <typename T, int SPMM_ITEMS, int BITS>
extern SYCL_EXTERNAL void kspmm_coo_very_sparse_naive(int *max_count, int *max_idx,
                                 int *offset_rowidx, int *rowidx, int *colidx,
                                 sycl::half *values, T *B, sycl::half *out,
                                 float *__restrict__ const dequant_stats,
                                 int nnz, int rowsA, int rowsB, int colsB,
                                 const sycl::nd_item<3> &item_ct1,
                                 sycl::half *smem_dequant_stats,
                                 const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx,
                                 const sycl_dacc &dacc_offset_rowidx, 
                                 const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, 
                                 const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<T, 1> &dacc_B, 
                                 const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);

//=====================mm dequant ====================================
template <int ITEMS_PER_THREAD, int SUBTILE_ROWS, int THREADS>
extern SYCL_EXTERNAL void kdequant_mm_int32_fp16(
    int *__restrict__ const A, float *__restrict__ const rowStats,
    float *__restrict__ const colStats, sycl::half *out, float *newRowStats,
    float *newcolStats, sycl::half *__restrict__ const bias, const int numRows,
    const int numCols, const int tileCols, const int n,
    const sycl::nd_item<3> &item_ct1, float *smem_rowStats, const sycl_la &tacc, const sycl_dacc &dacc_A,
    const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, const sycl::accessor<sycl::half, 1> &dacc_out, 
    const sycl::accessor<sycl::half, 1> &dacc_bias 
);
//==================k row col stats=====================

template<typename T, int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int SPARSE_DECOMP> extern SYCL_EXTERNAL void kgetColRowStats(T * __restrict__ A, float *rowStats, float *colStats, int * nnz_count_row, float nnz_threshold, int rows, int cols, int tiledRows, int tiledCols,
 const sycl::nd_item<3> &item_ct1, float *smem_row_absmax_values, int *smem_row_nnz_values, const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, const sycl_dacc &dacc_nnz_count_row);

//===========================double row col quant===================
template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS,
          int SPARSE_DECOMP> extern SYCL_EXTERNAL void kDoubleRowColQuant(sycl::half *__restrict__ const A,
                        float *__restrict__ const rowStats,
                        float *__restrict__ const colStats,
                        char *out_col_normed, char *out_row_normed, int *rowidx,
                        int *colidx, sycl::half *val,
                        int *__restrict__ nnz_block_ptr, float threshold,
                        int rows, int cols, int tiledCols,
                        const sycl::nd_item<3> &item_ct1,
                        float *smem_row_stats, unsigned int *smem_nnz_row_idx,
                        const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_A,
                        const sycl_dacc_char &dacc_out_col_normed, const sycl_dacc_char &dacc_out_row_normed,
                        const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, 
                        const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, 
                        const sycl::accessor<sycl::half, 1> &dacc_val, const sycl_dacc &dacc_nnz_block_ptr);

//==============================k transfrom row col=====================
template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int TRANSPOSE, int FORMAT> extern SYCL_EXTERNAL void kTransformRowToFormat(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols, const sycl::nd_item<3> &item_ct1, char *smem_data,
const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

//========================k extract outliers=========================
template <int FORMAT> extern SYCL_EXTERNAL  void kExtractOutliers(char *A, int *idx, char *out, int idx_size, int rowsA, int colsA, int tiledRowsA, int tiledColsA, const sycl::nd_item<3> &item_ct1, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out, const sycl_dacc &dacc_idx);

//=========================gemm device============================
template <typename T, int BITS, int THREADS> extern SYCL_EXTERNAL void gemm_device(int M, int N, int K, T * __restrict__ const A,  T* B,  T * out,  int lda, int ldb, int ldc, const sycl::nd_item<3> &item_ct1, T *smem_A, T *smem_B, const sycl::accessor<T, 1> &dacc_A, const sycl::accessor<T, 1> &dacc_B, const sycl::accessor<T, 1> &dacc_out);

//=========================gemm 4 bit inf================================
template <typename T, int THREADS>  extern SYCL_EXTERNAL void kgemm_4bit_inference(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize, const sycl::nd_item<3> &item_ct1,  T *smem_A, unsigned char *smem_B,
 T *smem_C, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out, const sycl::accessor<T, 1> &dacc_A,
  const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out);
 
//====================gemm 4 bit naive inf============================
template <typename T, int THREADS, int BITS> extern SYCL_EXTERNAL  void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize, const sycl::nd_item<3> &item_ct1, T *quant_map, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out, const sycl_dacc_float &dacc_absmax, const sycl_dacc_float &dacc_datatype);

template <typename T, int FUNC> extern SYCL_EXTERNAL void kfunc(T *A, T *B, T value, long n,
                                           const sycl::nd_item<3> &item_ct1);

#endif

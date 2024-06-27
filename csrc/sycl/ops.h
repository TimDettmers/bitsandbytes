// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#ifndef ops_H
#define ops_H

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include <dpct/blas_utils.hpp>

#include <dpct/sparse_utils.hpp>

#include <vector>
#include <functional>
#include <dpct/dpl_utils.hpp>


#define THREADS_PER_BLOCKS (512)

inline void checkCudaStatus(int status) {
    /*
    DPCT1000:93: Error handling if-stmt was detected but could not be rewritten.
    */
    if (status != 0) {
        /*
        DPCT1009:94: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        printf(
            "cuda API failed with status %d: %s\n", status,
            "cudaGetErrorString is not supported" /*cudaGetErrorString(status)*/);
        /*
        DPCT1001:92: The statement could not be removed.
        */
        throw std::logic_error("cuda API failed");
    }
}

inline int checkCublasStatus(int status) {
    if (status != 0) {
        printf("cuBLAS API failed with status %d\n", status);
        //throw std::logic_error("cuBLAS API failed");
        return 1;
    }
    return 0;
}

typedef enum Operations_t
{
	ksmul = 0,
} Operations_t;

typedef enum Optimizer_t
{
	ADAM = 0,
	MOMENTUM = 1,
  RMSPROP = 2,
  LARS = 3,
  ADAGRAD = 4,
  LION = 5,
} Optimizer_t;

typedef enum Transform_t
{
	ROW = 0,
	COL = 1,
  COL32 = 2,
  COL_TURING = 3,
  COL_AMPERE = 4,
} Transform_t;

typedef enum DataType_t
{
	General8bit = 0,
	FP4 = 1,
  NF4 = 2,
} DataType_t;

typedef enum Funcs_t
{
	FILL = 0,
	ARANGE = 1,
	_MUL = 2,
} Funcs_t;

class Context
{
    public:
                                dpct::queue_ptr m_handle;

                                Context()
				{
                                        dpct::queue_ptr handle;
                                        handle = &dpct::get_default_queue();
                                        m_handle = handle;
				}

};

class ContextLt
{
    public:
                                dpct::queue_ptr m_handle;

                                ContextLt()
				{
                                        dpct::queue_ptr handle;
                                        handle = &dpct::get_default_queue();
                                        m_handle = handle;
				}

};

class ContextCusparse
{
    public:
                                sycl::queue *m_handle;

                                ContextCusparse()
				{
                                        sycl::queue *handle;
                                        handle = &dpct::get_default_queue();
                                        m_handle = handle;
				}

};


template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n);

void quantize(float *code, float *A, unsigned char *out, int n);
void dequantize(float *code, unsigned char *A, float *out, int n);
template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int block_size, const int n);

template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2, float eps, float weight_decay,
                int step, float lr, const float gnorm_scale, bool skip_zeros, int n);

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g, unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr,
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n);

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale,
								bool skip_zeros, int n);

template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n);

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n);

void gemmex(Context * context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc);
void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
                    long long int strideA, long long int strideB, long long int strideC, int batchCount);


template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(dpct::queue_ptr ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(dpct::queue_ptr ltHandle, T *A, T *out, int dim1, int dim2);
void cutlass_igemm(bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc);
void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats,
                           sycl::half *out, float *newRowStats,
                           float *newcolStats, sycl::half *bias, int numRows,
                           int numCols);
void getColRowStats(sycl::half *A, float *rowStats, float *colStats,
                    int *nnz_count_row, float nnz_threshold, int rows,
                    int cols);
void doubleRowColQuant(sycl::half *A, float *rowStats, float *colStats,
                       char *out_col_normed, char *out_row_normed, int *rowidx,
                       int *colidx, sycl::half *val, int *nnz_block_ptr,
                       float threshold, int rows, int cols);

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols);

void spmm_coo(sycl::queue *handle, int *A_rowidx, int *A_colidx,
              sycl::half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols,
              int ldb, sycl::half *B, int ldc, sycl::half *C,
              bool transposed_B);

template <typename T, int BITS>
void spmm_coo_very_sparse_naive(int *max_count, int *max_idx,
                                int *offset_rowidx, int *rowidx, int *colidx,
                                sycl::half *values, T *B, sycl::half *out,
                                float *dequant_stats, int nnz_rows, int nnz,
                                int rowsA, int rowsB, int colsB);

template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols);

void matmul4bite(sycl::half *A, unsigned char *B, sycl::half *out, int lda,
                 int ldb, int rowsA, int colsA, int colsB);

template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits);
template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize);
template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize);

template <typename T, int FUNC> void func(T *A, T *B, T value, long n);

#endif

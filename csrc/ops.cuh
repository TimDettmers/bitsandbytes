// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


#ifndef ops_H
#define ops_H

#include <stdio.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <vector>
#include <functional>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#if USE_CUDA_WRAPPER

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef const char* (*cudaGetErrorString_t)(cudaError_t err);
typedef const char* (*cusparseGetErrorString_t)(cusparseStatus_t status);
typedef cusparseStatus_t (*cusparseCreate_t)(cusparseHandle_t* handle);
typedef cublasStatus_t (*cublasCreate_v2_t)(cublasHandle_t* handle);
typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t* lightHandle);
//typedef cudaError_t (*cudaMallocManaged_t)(void **devPtr, size_t size, unsigned int flags);
//typedef cudaError_t (*cudaMemPrefetchAsync_t)(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
//typedef cudaError_t (*cudaDeviceGetAttribute_t)(int *value, enum cudaDeviceAttr attr, int device);

typedef cusparseStatus_t (*cusparseCreateCoo_t)(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 cooRowInd,
                  void*                 cooColInd,
                  void*                 cooValues,
                  cusparseIndexType_t   cooIdxType,
                  cusparseIndexBase_t   idxBase,
                  cudaDataType          valueType);

typedef cusparseStatus_t (*cusparseCreateDnMat_t)(cusparseDnMatDescr_t* dnMatDescr,
                    int64_t               rows,
                    int64_t               cols,
                    int64_t               ld,
                    void*                 values,
                    cudaDataType          valueType,
                    cusparseOrder_t       order);

typedef cusparseStatus_t (*cusparseSpMM_bufferSize_t)(cusparseHandle_t     handle,
                        cusparseOperation_t  opA,
                        cusparseOperation_t  opB,
                        const void*          alpha,
                        cusparseSpMatDescr_t matA,
                        cusparseDnMatDescr_t matB,
                        const void*          beta,
                        cusparseDnMatDescr_t matC,
                        cudaDataType         computeType,
                        cusparseSpMMAlg_t    alg,
                        size_t*              bufferSize);

typedef cusparseStatus_t (*cusparseSpMM_t)(cusparseHandle_t     handle,
             cusparseOperation_t  opA,
             cusparseOperation_t  opB,
             const void*          alpha,
             cusparseSpMatDescr_t matA,
             cusparseDnMatDescr_t matB,
             const void*          beta,
             cusparseDnMatDescr_t matC,
             cudaDataType         computeType,
             cusparseSpMMAlg_t    alg,
             void*                externalBuffer);

typedef cusparseStatus_t (*cusparseDestroySpMat_t)(cusparseSpMatDescr_t spMatDescr);
typedef cusparseStatus_t (*cusparseDestroyDnMat_t)(cusparseDnMatDescr_t dnMatDescr);

typedef cudaError_t (*cudaMemset_t)(void *devPtr, int value, size_t count);
typedef cudaError_t (*cudaMalloc_t)(void **devPtr, size_t size);
typedef cudaError_t (*cudaFree_t)(void *devPtr);
typedef cudaError_t (*cudaPeekAtLastError_t)(void);

typedef cublasStatus_t (*cublasGemmEx_t)(cublasHandle_t handle,
                                                   cublasOperation_t transa,
                                                   cublasOperation_t transb,
                                                   int m,
                                                   int n,
                                                   int k,
                                                   const void* alpha, /* host or device pointer */
                                                   const void* A,
                                                   cudaDataType Atype,
                                                   int lda,
                                                   const void* B,
                                                   cudaDataType Btype,
                                                   int ldb,
                                                   const void* beta, /* host or device pointer */
                                                   void* C,
                                                   cudaDataType Ctype,
                                                   int ldc,
                                                   cublasComputeType_t computeType,
                                                   cublasGemmAlgo_t algo);

typedef cublasStatus_t (*cublasGemmStridedBatchedEx_t)(cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const void* alpha, /* host or device pointer */
                                                                 const void* A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long long int strideA, /* purposely signed */
                                                                 const void* B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const void* beta, /* host or device pointer */
                                                                 void* C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount,
                                                                 cublasComputeType_t computeType,
                                                                 cublasGemmAlgo_t algo);

typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(  //
    cublasLtMatrixLayout_t* matLayout,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld);

typedef cublasStatus_t (*cublasLtMatrixLayoutSetAttribute_t)(  //
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* buf,
    size_t sizeInBytes);

typedef cublasStatus_t (*cublasLtMatrixTransform_t)(cublasLtHandle_t lightHandle,
                                                    cublasLtMatrixTransformDesc_t transformDesc,
                                                    const void* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cublasLtMatrixLayout_t Adesc,
                                                    const void* beta, /* host or device pointer */
                                                    const void* B,
                                                    cublasLtMatrixLayout_t Bdesc,
                                                    void* C,
                                                    cublasLtMatrixLayout_t Cdesc,
                                                    cudaStream_t stream);

typedef cublasStatus_t (*cublasLtMatrixTransformDescCreate_t)(cublasLtMatrixTransformDesc_t* transformDesc,
                                                              cudaDataType scaleType);

typedef cublasStatus_t (*cublasLtMatrixTransformDescSetAttribute_t)(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);

typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(cublasLtMatrixLayout_t matLayout);

typedef cublasStatus_t (*cublasLtMatrixTransformDescDestroy_t)(cublasLtMatrixTransformDesc_t transformDesc);

typedef cublasStatus_t (*cublasLtMatmul_t)(cublasLtHandle_t lightHandle,
                                           cublasLtMatmulDesc_t computeDesc,
                                           const void* alpha, /* host or device pointer */
                                           const void* A,
                                           cublasLtMatrixLayout_t Adesc,
                                           const void* B,
                                           cublasLtMatrixLayout_t Bdesc,
                                           const void* beta, /* host or device pointer */
                                           const void* C,
                                           cublasLtMatrixLayout_t Cdesc,
                                           void* D,
                                           cublasLtMatrixLayout_t Ddesc,
                                           const cublasLtMatmulAlgo_t* algo,
                                           void* workspace,
                                           size_t workspaceSizeInBytes,
                                           cudaStream_t stream);

typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t* matmulDesc,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType);

typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t matmulDesc);

typedef cublasStatus_t (*cublasLtMatmulDescSetAttribute_t)(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);


/* externs */
extern cudaGetErrorString_t _cudaGetErrorString;
extern cusparseGetErrorString_t _cusparseGetErrorString;
//extern cudaMallocManaged_t _cudaMallocManaged;
//extern cudaMemPrefetchAsync_t _cudaMemPrefetchAsync;
//extern cudaDeviceGetAttribute_t _cudaDeviceGetAttribute;

extern cusparseCreate_t _cusparseCreate;
extern cublasCreate_v2_t _cublasCreate_v2;
extern cublasLtCreate_t _cublasLtCreate;

extern cusparseDestroySpMat_t _cusparseDestroySpMat;
extern cusparseDestroyDnMat_t _cusparseDestroyDnMat;
extern cusparseCreateCoo_t _cusparseCreateCoo;
extern cusparseSpMM_t _cusparseSpMM;
extern cusparseSpMM_bufferSize_t _cusparseSpMM_bufferSize;
extern cusparseCreateDnMat_t _cusparseCreateDnMat;

extern cudaMemset_t _cudaMemset;
extern cudaMalloc_t _cudaMalloc;
extern cudaFree_t _cudaFree;
extern cudaPeekAtLastError_t _cudaPeekAtLastError;

extern cublasGemmEx_t _cublasGemmEx;
extern cublasGemmStridedBatchedEx_t _cublasGemmStridedBatchedEx;

extern cublasLtMatrixLayoutCreate_t _cublasLtMatrixLayoutCreate;
extern cublasLtMatrixLayoutSetAttribute_t _cublasLtMatrixLayoutSetAttribute;
extern cublasLtMatrixTransform_t _cublasLtMatrixTransform;
extern cublasLtMatrixTransformDescCreate_t _cublasLtMatrixTransformDescCreate;
extern cublasLtMatrixTransformDescSetAttribute_t _cublasLtMatrixTransformDescSetAttribute;
extern cublasLtMatrixLayoutDestroy_t _cublasLtMatrixLayoutDestroy;
extern cublasLtMatrixTransformDescDestroy_t _cublasLtMatrixTransformDescDestroy;
extern cublasLtMatmul_t _cublasLtMatmul;
extern cublasLtMatmulDescCreate_t _cublasLtMatmulDescCreate;
extern cublasLtMatmulDescDestroy_t _cublasLtMatmulDescDestroy;
extern cublasLtMatmulDescSetAttribute_t _cublasLtMatmulDescSetAttribute;


#define cudaGetErrorString _cudaGetErrorString
#define cusparseGetErrorString _cusparseGetErrorString
#define cusparseCreate _cusparseCreate
#define cublasCreate_v2 _cublasCreate_v2
#define cublasLtCreate _cublasLtCreate

#define cudaMemset _cudaMemset
#define cudaMalloc _cudaMalloc
#define cudaFree _cudaFree
#define cudaPeekAtLastError _cudaPeekAtLastError

#define cusparseCreateCoo _cusparseCreateCoo
#define cusparseDestroySpMat _cusparseDestroySpMat
#define cusparseDestroyDnMat _cusparseDestroyDnMat
#define cusparseSpMM _cusparseSpMM
#define cusparseSpMM_bufferSize _cusparseSpMM_bufferSize
#define cusparseCreateDnMat _cusparseCreateDnMat

#define cublasGemmEx _cublasGemmEx
#define cublasGemmStridedBatchedEx _cublasGemmStridedBatchedEx
#define cublasLtMatrixLayoutCreate _cublasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutSetAttribute _cublasLtMatrixLayoutSetAttribute
#define cublasLtMatrixTransform _cublasLtMatrixTransform
#define cublasLtMatrixTransformDescCreate _cublasLtMatrixTransformDescCreate

#define cublasLtMatrixTransformDescSetAttribute _cublasLtMatrixTransformDescSetAttribute
#define cublasLtMatrixLayoutDestroy _cublasLtMatrixLayoutDestroy
#define cublasLtMatrixTransformDescDestroy _cublasLtMatrixTransformDescDestroy
#define cublasLtMatmul _cublasLtMatmul
#define cublasLtMatmulDescCreate _cublasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy _cublasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute _cublasLtMatmulDescSetAttribute

#endif /* USE_CUDA_WRAPPER */

#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#define THREADS_PER_BLOCKS (512)

#define CHECK_CUSPARSE(value) {                      \
  cusparseStatus_t _m_cudaStat = value;                    \
  if (_m_cudaStat != CUSPARSE_STATUS_SUCCESS) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cusparseGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }


#define THREADS_PER_BLOCKS (512)


inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline int checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
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
				cublasHandle_t m_handle;

				Context()
				{
					cublasHandle_t handle;
					cublasCreate_v2(&handle);
					m_handle = handle;
				}

};

class ContextLt
{
    public:
				cublasLtHandle_t m_handle;

				ContextLt()
				{
					cublasLtHandle_t handle;
					cublasLtCreate(&handle);
					m_handle = handle;
				}

};

class ContextCusparse
{
    public:
				cusparseHandle_t m_handle;

				ContextCusparse()
				{
					cusparseHandle_t handle;
					cusparseCreate(&handle);
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


template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(cublasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2);
void cutlass_igemm(bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc);
void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, half* bias, int numRows, int numCols);
void getColRowStats(half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols);
void doubleRowColQuant(half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed,
                       int *rowidx, int *colidx, half *val, int *nnz_block_ptr, float threshold, int rows, int cols);

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols);

void spmm_coo(cusparseHandle_t handle, int *A_rowidx, int *A_colidx, half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, half *B, int ldc, half* C, bool transposed_B);

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, T *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols);

void matmul4bite(half *A, unsigned char *B, half*out, int lda, int ldb, int rowsA, int colsA, int colsB);

template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits);
template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize);
template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize);

template <typename T, int FUNC> void func(T *A, T *B, T value, long n);

#endif

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#if BUILD_CUDA
#include <ops.cuh>
#endif
#include <cpu_ops.h>

// We cannot call templated code from C, so we wrap the template in a C compatible call here if necessary.
// We use macro functions to expand all the different optimizers. Looks ugly, and is ugly, but its better than to
// maintain all that boilerplate
//===================================================================================
//                               UNMANGLED CALLS
//===================================================================================

#if BUILD_CUDA
void estimateQuantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles<float>(A, code, offset, n); }
void estimateQuantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles<half>(A, code, offset, n); }


//void gemm_host_fp32(int M, int N, int K, float * A,  float* B,  float * out,  int lda, int ldb, int ldc)
//{ gemm_host<float>(M, N, K, A, B, out, lda, ldb, ldc, 32); }
void gemm_host_fp16(int M, int N, int K, half * A,  half* B,  half * out,  int lda, int ldb, int ldc)
{ gemm_host<half>(M, N, K, A, B, out, lda, ldb, ldc, 16); }

void gemm_4bit_inference(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize)
{ gemm_4bit_inference<half>(m, n, k, A, B, absmax,  out, lda, ldb, ldc, blocksize); }

void gemm_4bit_inference_naive_fp16(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize)
{ gemm_4bit_inference_naive<half, 16>(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

void gemm_4bit_inference_naive_bf16(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize)
{ gemm_4bit_inference_naive<__nv_bfloat16, 16>(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

void gemm_4bit_inference_naive_fp32(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize)
{ gemm_4bit_inference_naive<float, 32>(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

#define MAKE_ELEMENTWISE_FUNC(fname, type_name, ctype, FUNC) \
void fname##_##type_name(ctype *A, ctype *B, ctype value, long n){ func<ctype, FUNC>(A, B, value, n); } \

MAKE_ELEMENTWISE_FUNC(fill, fp32, float, FILL)
MAKE_ELEMENTWISE_FUNC(fill, uint8, unsigned char, FILL)
MAKE_ELEMENTWISE_FUNC(arange, fp32, float, ARANGE)
MAKE_ELEMENTWISE_FUNC(_mul, fp32, float, _MUL)


#define MAKE_FUNC32(fname, oname, gtype, gbits) \
void fname##32bit_grad_##gbits(gtype *g, gtype *p, \
               float* state1, float* state2, float *unorm, float max_unorm, float param_norm, \
               const float beta1, const float beta2, const float eps, const float weight_decay, \
               const int step, const float lr, float gnorm_scale, bool skip_zeros, const int n) \
{ optimizer32bit<gtype, oname>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n); } \

MAKE_FUNC32(momentum, MOMENTUM, float, 32)
MAKE_FUNC32(momentum, MOMENTUM, half, 16)
MAKE_FUNC32(adam, ADAM, float, fp32)
MAKE_FUNC32(adam, ADAM, half, fp16)
MAKE_FUNC32(adam, ADAM, __nv_bfloat16, bf16)
MAKE_FUNC32(rmsprop, RMSPROP, float, 32)
MAKE_FUNC32(rmsprop, RMSPROP, half, 16)
MAKE_FUNC32(lion, LION, float, fp32)
MAKE_FUNC32(lion, LION, half, fp16)
MAKE_FUNC32(lion, LION, __nv_bfloat16, bf16)
MAKE_FUNC32(adagrad, ADAGRAD, float, 32)
MAKE_FUNC32(adagrad, ADAGRAD, half, 16)

#define MAKE_FUNC8(fname, oname, gtype, gbits) \
void fname##_static_8bit_grad_##gbits(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
								float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, float gnorm_scale, int n) \
{  \
	optimizerStatic8bit<gtype, oname>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, \
			                                  quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n); \
} \

MAKE_FUNC8(adam, ADAM, float, 32)
MAKE_FUNC8(adam, ADAM, half, 16)
MAKE_FUNC8(momentum, MOMENTUM, float, 32)
MAKE_FUNC8(momentum, MOMENTUM, half, 16)
MAKE_FUNC8(rmsprop, RMSPROP, float, 32)
MAKE_FUNC8(rmsprop, RMSPROP, half, 16)
MAKE_FUNC8(lion, LION, float, 32)
MAKE_FUNC8(lion, LION, half, 16)

#define MAKE_BLOCKWISE8(fname, optim_name, gtype, gbits) \
void fname##_8bit_blockwise_grad_##gbits(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)\
{	optimizerStatic8bitBlockwise<gtype, optim_name>(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n); }\

MAKE_BLOCKWISE8(adam, ADAM, half, fp16)
MAKE_BLOCKWISE8(adam, ADAM, float, fp32)
MAKE_BLOCKWISE8(momentum, MOMENTUM, half, fp16)
MAKE_BLOCKWISE8(momentum, MOMENTUM, float, fp32)
MAKE_BLOCKWISE8(rmsprop, RMSPROP, half, fp16)
MAKE_BLOCKWISE8(rmsprop, RMSPROP, float, fp32)
MAKE_BLOCKWISE8(adagrad, ADAGRAD, half, fp16)
MAKE_BLOCKWISE8(adagrad, ADAGRAD, float, fp32)
MAKE_BLOCKWISE8(adam, ADAM, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(lion, LION, half, fp16)
MAKE_BLOCKWISE8(lion, LION, float, fp32)
MAKE_BLOCKWISE8(lion, LION, __nv_bfloat16, bf16)


void percentileClipping_g32(float * g, float *gnorm_vec, int step, const int n){ percentileClipping<float>(g, gnorm_vec, step, n); }
void percentileClipping_g16(half * g, float *gnorm_vec, int step, const int n){ percentileClipping<half>(g, gnorm_vec, step, n); }

void quantizeBlockwise_fp16(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<half, 0, General8bit>(code, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_fp16_fp4(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<half, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_fp16_nf4(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<half, 0, NF4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }

void quantizeBlockwise_bf16(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<__nv_bfloat16, 0, General8bit>(code, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_bf16_fp4(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<__nv_bfloat16, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_bf16_nf4(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<__nv_bfloat16, 0, NF4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }

void quantizeBlockwise_fp32(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<float, 0, General8bit>(code, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_fp32_fp4(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<float, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }
void quantizeBlockwise_fp32_nf4(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise<float, 0, NF4>(NULL, A, absmax, out, NULL, 0, blocksize, n); }

void dequantizeBlockwise_fp16(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise<half, General8bit>(code, A, absmax, out, blocksize, n); } \
void dequantizeBlockwise_fp16_fp4(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise<half, FP4>(NULL, A, absmax, out, blocksize, n); } \
void dequantizeBlockwise_fp16_nf4(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise<half, NF4>(NULL, A, absmax, out, blocksize, n); } \

void dequantizeBlockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise<float, General8bit>(code, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_fp32_fp4(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise<float, FP4>(NULL, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_fp32_nf4(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise<float, NF4>(NULL, A, absmax, out, blocksize, n); }

void dequantizeBlockwise_bf16(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise<__nv_bfloat16, General8bit>(code, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_bf16_fp4(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise<__nv_bfloat16, FP4>(NULL, A, absmax, out, blocksize, n); }
void dequantizeBlockwise_bf16_nf4(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise<__nv_bfloat16, NF4>(NULL, A, absmax, out, blocksize, n); }


#define MAKE_FUNC_TRANSFORM(fbits, fsrc, ftrgt, ftranspose, dtype, src, target, transpose, bits) \
void transform_##fbits##_##fsrc##_to_##ftrgt##_##ftranspose(cublasLtHandle_t ltHandle, dtype *A, dtype *out, int dim1, int dim2) \
{ \
	transform<dtype, src, target, transpose, bits>(ltHandle, A, out, dim1, dim2); \
} \

MAKE_FUNC_TRANSFORM(8, row, col, n, int8_t, ROW, COL, false, 8);
MAKE_FUNC_TRANSFORM(8, row, row, n, int8_t, ROW, ROW, false, 8);
MAKE_FUNC_TRANSFORM(8, row, col32, n, int8_t, ROW, COL32, false, 8);
MAKE_FUNC_TRANSFORM(32, row, col32, n, int32_t, ROW, COL32, false, 32);
MAKE_FUNC_TRANSFORM(8, row, col_turing, n, int8_t, ROW, COL_TURING, false, 8);
MAKE_FUNC_TRANSFORM(8, row, col_ampere, n, int8_t, ROW, COL_AMPERE, false, 8);
MAKE_FUNC_TRANSFORM(8, col32, row, n, int8_t, COL32, ROW, false, 8);
MAKE_FUNC_TRANSFORM(32, col32, row, n, int32_t, COL32, ROW, false, 32);

void transform_row2col32(char * A, char *out, int rows, int cols){ transformRowToFormat<COL32, 0>(A, out, rows, cols); }
void transform_row2col32T(char * A, char *out, int rows, int cols){ transformRowToFormat<COL32, 1>(A, out, rows, cols); }
void transform_row2turing(char * A, char *out, int rows, int cols){ transformRowToFormat<COL_TURING, 0>(A, out, rows, cols); }
void transform_row2turingT(char * A, char *out, int rows, int cols){ transformRowToFormat<COL_TURING, 1>(A, out, rows, cols); }
void transform_row2ampere(char * A, char *out, int rows, int cols){ transformRowToFormat<COL_AMPERE, 0>(A, out, rows, cols); }
void transform_row2ampereT(char * A, char *out, int rows, int cols){ transformRowToFormat<COL_AMPERE, 1>(A, out, rows, cols); }

void extractOutliers_turing(char * A, int *idx, char *out, int idx_size, int rows, int cols){ extractOutliers<COL_TURING>(A, idx, out, idx_size, rows, cols); }
void extractOutliers_ampere(char * A, int *idx, char *out, int idx_size, int rows, int cols){ extractOutliers<COL_AMPERE>(A, idx, out, idx_size, rows, cols); }

 int igemmlt_turing_32(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_TURING, 32, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

 int igemmlt_turing_8(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_TURING, 8, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

 int igemmlt_turing_8_rowscale(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_TURING, 8, 1>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

 int igemmlt_ampere_32(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_AMPERE, 32, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

 int igemmlt_ampere_8(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_AMPERE, 8, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

 int igemmlt_ampere_8_rowscale(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_AMPERE, 8, 1>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

void spmm_coo_very_sparse_naive_fp16(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, half *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{ spmm_coo_very_sparse_naive<half, 16>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB, colsB); }

void spmm_coo_very_sparse_naive_int8(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, signed char *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{ spmm_coo_very_sparse_naive<signed char, 8>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB, colsB); }
#endif

extern "C"
{
#if BUILD_CUDA
	void cestimate_quantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles_fp32(A, code, offset, n); }
	void cestimate_quantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles_fp16(A, code, offset, n); }
	void cquantize(float *code, float *A, unsigned char *out, int n){ quantize(code, A, out, n); }
	void cdequantize(float *code, unsigned char *A, float *out, int n){ dequantize(code, A, out, n); }

  void cdequantize_blockwise_fp16_fp4(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_fp16(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise_fp16(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_fp16_nf4(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise_fp16_nf4(code, A, absmax, out, blocksize, n); }

  void cquantize_blockwise_fp16(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp16(code, A, absmax, out, blocksize, n); }
  void cquantize_blockwise_fp16_fp4(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n); }
  void cquantize_blockwise_fp16_nf4(float * code, half *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp16_nf4(code, A, absmax, out, blocksize, n); }

  void cquantize_blockwise_fp32(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp32(code, A, absmax, out, blocksize, n); }
  void cquantize_blockwise_fp32_fp4(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n); }
  void cquantize_blockwise_fp32_nf4(float * code, float *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_fp32_nf4(code, A, absmax, out, blocksize, n); }

  void cdequantize_blockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise_fp32(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_fp32_fp4(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_fp32_nf4(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise_fp32_nf4(code, A, absmax, out, blocksize, n); }

  void cquantize_blockwise_bf16(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_bf16(code, A, absmax, out, blocksize, n); }
  void cquantize_blockwise_bf16_fp4(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n); }
  void cquantize_blockwise_bf16_nf4(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, int blocksize, const int n){ quantizeBlockwise_bf16_nf4(code, A, absmax, out, blocksize, n); }

  void cdequantize_blockwise_bf16(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise_bf16(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_bf16_fp4(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_bf16_nf4(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n){ dequantizeBlockwise_bf16_nf4(code, A, absmax, out, blocksize, n); }

	#define MAKE_CFUNC32(name, gtype, gbits) \
	void c##name##32bit_grad_##gbits(gtype *g, gtype *p, \
								 float* state1, float* state2, float *unorm, float max_unorm, float param_norm, \
								 const float beta1, const float beta2, const float eps, const float weight_decay, \
								 const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n) \
	{ name##32bit_grad_##gbits(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n); } \

	MAKE_CFUNC32(adam, float, fp32)
	MAKE_CFUNC32(adam, half, fp16)
	MAKE_CFUNC32(adam, __nv_bfloat16, bf16)
	MAKE_CFUNC32(momentum, float, 32)
	MAKE_CFUNC32(momentum, half, 16)
	MAKE_CFUNC32(rmsprop, float, 32)
	MAKE_CFUNC32(rmsprop, half, 16)
	MAKE_CFUNC32(lion, float, fp32)
	MAKE_CFUNC32(lion, half, fp16)
	MAKE_CFUNC32(lion, __nv_bfloat16, bf16)
	MAKE_CFUNC32(adagrad, float, 32)
	MAKE_CFUNC32(adagrad, half, 16)

	#define MAKE_CFUNC8(name, gtype, gbits) \
	void c##name##_static_8bit_grad_##gbits(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, float gnorm_scale, int n) \
  {  \
	    name##_static_8bit_grad_##gbits(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, \
			                                 quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n); \
  } \

	MAKE_CFUNC8(adam, float, 32)
	MAKE_CFUNC8(adam, half, 16)
	MAKE_CFUNC8(momentum, float, 32)
	MAKE_CFUNC8(momentum, half, 16)
	MAKE_CFUNC8(rmsprop, float, 32)
	MAKE_CFUNC8(rmsprop, half, 16)
	MAKE_CFUNC8(lion, float, 32)
	MAKE_CFUNC8(lion, half, 16)

  #define MAKE_CBLOCKWISE8(fname, optim_name, gtype, gbits) \
  void c##fname##_8bit_blockwise_grad_##gbits(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n) \
  {	fname##_8bit_blockwise_grad_##gbits(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n); } \

	MAKE_CBLOCKWISE8(adam, ADAM, half, fp16)
	MAKE_CBLOCKWISE8(adam, ADAM, float, fp32)
	MAKE_CBLOCKWISE8(momentum, MOMENTUM, half, fp16)
	MAKE_CBLOCKWISE8(momentum, MOMENTUM, float, fp32)
	MAKE_CBLOCKWISE8(rmsprop, RMSPROP, half, fp16)
	MAKE_CBLOCKWISE8(rmsprop, RMSPROP, float, fp32)
	MAKE_CBLOCKWISE8(adagrad, ADAGRAD, half, fp16)
	MAKE_CBLOCKWISE8(adagrad, ADAGRAD, float, fp32)
	MAKE_CBLOCKWISE8(adam, ADAM, __nv_bfloat16, bf16)
	MAKE_CBLOCKWISE8(lion, LION, half, fp16)
	MAKE_CBLOCKWISE8(lion, LION, float, fp32)
	MAKE_CBLOCKWISE8(lion, LION, __nv_bfloat16, bf16)

	void cpercentile_clipping_g32(float * g, float *gnorm_vec, int step, const int n){ percentileClipping_g32(g, gnorm_vec, step, n); }
	void cpercentile_clipping_g16(half * g, float *gnorm_vec, int step, const int n){ percentileClipping_g16(g, gnorm_vec, step, n); }
	void chistogram_scatter_add_2d(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n){ histogramScatterAdd2D(histogram, index1, index2, src, maxidx1, n); }

	void cigemm(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
	{ gemmex(context, transposeA, transposeB, m, n, k, A, B, C, lda, ldb, ldc); }
	void cbatched_igemm(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
			               long strideA, long strideB, long strideC, int batchCount)
	{ strided_gemmex(context, transposeA, transposeB, m, n, k, A, B, C, lda, ldb, ldc, strideA, strideB, strideC, batchCount); }

	Context *get_context(){ return new Context(); }
	ContextCusparse *get_cusparse(){ return new ContextCusparse(); }

	int cigemmlt_turing_32(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_turing_32((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }
	//{ (cublasLtHandle_t)context->m_handle; return 0; }
	//{ return 0; }//igemmlt_turing_32((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

	int cigemmlt_turing_8(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_turing_8((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

	int cigemmlt_turing_8_rowscale(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_turing_8_rowscale((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

	int cigemmlt_ampere_32(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_ampere_32((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

	int cigemmlt_ampere_8_rowscale(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_ampere_8_rowscale((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

	int cigemmlt_ampere_8(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_ampere_8((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

  #define MAKE_FUNC_CTRANSFORM(fbits, fsrc, ftrgt, ftranspose, dtype, src, target, transpose, bits) \
	void ctransform_##fbits##_##fsrc##_to_##ftrgt##_##ftranspose(Context *context, dtype *A, dtype *out, int dim1, int dim2) \
	{ \
		transform_##fbits##_##fsrc##_to_##ftrgt##_##ftranspose((cublasLtHandle_t) context->m_handle, A, out, dim1, dim2); \
	} \

	MAKE_FUNC_CTRANSFORM(8, row, col, n, int8_t, ROW, COL, false, 8)
	MAKE_FUNC_CTRANSFORM(8, row, row, n, int8_t, ROW, ROW, false, 8)
	MAKE_FUNC_CTRANSFORM(8, row, col32, n, int8_t, ROW, COL32, false, 8)
	MAKE_FUNC_CTRANSFORM(32, row, col32, n, int32_t, ROW, COL32, false, 32)
	MAKE_FUNC_CTRANSFORM(8, row, col_turing, n, int8_t, ROW, COL_TURING, false, 8)
	MAKE_FUNC_CTRANSFORM(8, row, col_ampere, n, int8_t, ROW, COL_AMPERE, false, 8)
	MAKE_FUNC_CTRANSFORM(8, col32, row, n, int8_t, COL32, ROW, false, 8)
	MAKE_FUNC_CTRANSFORM(32, col32, row, n, int32_t, COL32, ROW, false, 32)

	void cdequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, half* bias, int numRows, int numCols)
	{ dequant_mm_int32_fp16(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols); }
	void cget_col_row_stats(half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
	{ getColRowStats(A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols); }

  void cdouble_rowcol_quant(half * A, float *rowStats, float *colStats, int8_t *out_col_normed, int8_t *out_row_normed, int *rowidx, int *colidx, half *val, int *nnz_row_ptr, float threshold, int rows, int cols)
	{ doubleRowColQuant(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_row_ptr, threshold, rows, cols); }

	void ctransform_row2col32(char * A, char *out, int rows, int cols)
	{ transform_row2col32(A, out, rows, cols); }

	void ctransform_row2col32T(char * A, char *out, int rows, int cols)
	{ transform_row2col32T(A, out, rows, cols); }

	void ctransform_row2turing(char * A, char *out, int rows, int cols)
	{ transform_row2turing(A, out, rows, cols); }

	void ctransform_row2turingT(char * A, char *out, int rows, int cols)
	{ transform_row2turingT(A, out, rows, cols); }

	void ctransform_row2ampere(char * A, char *out, int rows, int cols)
	{ transform_row2ampere(A, out, rows, cols); }

	void ctransform_row2ampereT(char * A, char *out, int rows, int cols)
	{ transform_row2ampereT(A, out, rows, cols); }

	void cspmm_coo(ContextCusparse *context, int *A_rowidx, int *A_colidx, half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, half *B, int ldc, half* C, bool transposed_B)
  { spmm_coo((cusparseHandle_t) context->m_handle, A_rowidx, A_colidx, A_vals, A_nnz, A_rows, A_cols, B_cols, ldb, B, ldc, C, transposed_B); }

	void cspmm_coo_very_sparse_naive_fp16(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, half *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
	{ spmm_coo_very_sparse_naive_fp16(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB, colsB); }

	void cspmm_coo_very_sparse_naive_int8(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, signed char *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
	{ spmm_coo_very_sparse_naive_int8(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB, colsB); }

	void cextractOutliers_turing(char * A, int *idx, char *out, int idx_size, int rows, int cols){ extractOutliers_turing(A, idx, out, idx_size, rows, cols); }
	void cextractOutliers_ampere(char * A, int *idx, char *out, int idx_size, int rows, int cols){ extractOutliers_ampere(A, idx, out, idx_size, rows, cols); }

	//void cgemm_host_fp32(int M, int N, int K, float * A,  float* B,  float * out,  int lda, int ldb, int ldc)
	//{ gemm_host_fp32(M, N, K, A, B, out, lda, ldb, ldc); }

	void cgemm_host_fp16(int M, int N, int K, half * A,  half* B,  half * out,  int lda, int ldb, int ldc)
	{ gemm_host_fp16(M, N, K, A, B, out, lda, ldb, ldc); }

	void cgemm_4bit_inference(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize)
	{ gemm_4bit_inference(m, n, k, A, B, absmax, out, lda, ldb, ldc, blocksize); }

	void *cget_managed_ptr(size_t bytes)
	{
		void *ptr;
		CUDA_CHECK_RETURN(cudaMallocManaged(&ptr, bytes, cudaMemAttachHost));
		CUDA_CHECK_RETURN(cudaPeekAtLastError());

		return ptr;
	}

	void cprefetch(void *ptr, size_t bytes, int device)
	{

		int hasPrefetch = 0;
		CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&hasPrefetch, cudaDevAttrConcurrentManagedAccess, device)); // 40ns overhead
		if (hasPrefetch == 0) return;
 
		CUDA_CHECK_RETURN(cudaMemPrefetchAsync(ptr, bytes, device, 0));
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}

  #define CMAKE_ELEMENTWISE_FUNC(fname, type_name, ctype, FUNC) \
	void c##fname##_##type_name(ctype *A, ctype *B, ctype value, long n){ fname##_##type_name(A, B, value, n); } \

	CMAKE_ELEMENTWISE_FUNC(fill, fp32, float, FILL)
	CMAKE_ELEMENTWISE_FUNC(fill, uint8, unsigned char, FILL)
	CMAKE_ELEMENTWISE_FUNC(arange, fp32, float, ARANGE)
	CMAKE_ELEMENTWISE_FUNC(_mul, fp32, float, _MUL)

	void cgemm_4bit_inference_naive_fp16(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize)
	{ gemm_4bit_inference_naive_fp16(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

	void cgemm_4bit_inference_naive_bf16(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize)
	{ gemm_4bit_inference_naive_bf16(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

	void cgemm_4bit_inference_naive_fp32(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize)
	{ gemm_4bit_inference_naive_fp32(m, n, k, A, B, absmax,  datatype, out, lda, ldb, ldc, blocksize); }

#endif
	void cquantize_blockwise_cpu_fp32(float *code, float *A, float *absmax, unsigned char *out, long long blocksize, long long n){ quantize_cpu(code, A, absmax, out, blocksize, n); }
	void cdequantize_blockwise_cpu_fp32(float *code, unsigned char *A, float *absmax, float *out, long long blocksize, long long n){ dequantize_cpu(code, A, absmax, out, blocksize, n); }
}

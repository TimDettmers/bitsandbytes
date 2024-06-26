// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ops.cuh>
#include <kernels.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <BinSearch.h>
#include <cassert>
#include <common.h>

#define ERR_NOT_IMPLEMENTED 100


using namespace BinSearch;
using std::cout;
using std::endl;

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  int threads = 512;
  int num_blocks = n/threads;
  num_blocks = n % threads == 0 ? num_blocks : num_blocks + 1;
  kHistogramScatterAdd2D<<<num_blocks, 512>>>(histogram, index1, index2, src, maxidx1, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
	CUDA_CHECK_RETURN(cudaMemset(code, 0, 256*sizeof(float)));
  kEstimateQuantiles<T><<<num_blocks, 512>>>(A, code, offset, std::numeric_limits<T>::max(), n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void quantize(float *code, float *A, unsigned char *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  kQuantize<<<num_blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  kDequantize<<<num_blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;

  if(blocksize == 4096)
    kQuantizeBlockwise<T, 4096, 4, STOCHASTIC, DATA_TYPE><<<num_blocks, 1024>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 2048)
    kQuantizeBlockwise<T, 2048, 4, 0, DATA_TYPE><<<num_blocks, 512>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 1024)
    kQuantizeBlockwise<T, 1024, 4, 0, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 512)
    kQuantizeBlockwise<T, 512, 2, 0, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 256)
    kQuantizeBlockwise<T, 256, 2, 0, DATA_TYPE><<<num_blocks, 128>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 128)
    kQuantizeBlockwise<T, 128, 2, 0, DATA_TYPE><<<num_blocks, 64>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 64)
    kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE><<<num_blocks, 32>>>(code, A, absmax, out, rand, rand_offset, n);


  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;

  if(DATA_TYPE > 0)
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize/2, n);
  else
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize, n);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


//void matmul4bite(half *A, unsigned char *B, half*out, int lda, int ldb, int rowsA, int colsA, int colsB)
//{
//	int num_blocks = (colsB+32-1)/32;
//	kMatmul_inference_4bit<NF4, half, half, half><<<num_blocks, 256>>>(A, B, out, lda, ldb, rowsA, colsA, colsB);
//  CUDA_CHECK_RETURN(cudaPeekAtLastError());
//}


template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p, T* return_updates,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
        kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
      }
			kOptimizer32bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(g, p, return_updates, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
      if(max_unorm > 0.0f)
			{
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
				kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
			}

			kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(g, p, return_updates, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
    case LION:
      // in lion, the momentum update after the parameter update
      kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(g, p, return_updates, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());

      if(max_unorm > 0.0f)
      {
        CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
        kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
      }
      break;
	}
}

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g, T* return_updates,
                unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr,
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;

  if(max_unorm > 0.0f){ CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float))); }

	switch(OPTIMIZER)
	{
		case ADAM:
			CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1*sizeof(float)));
			CUDA_CHECK_RETURN(cudaMemset(new_max2, 0, 1*sizeof(float)));
			kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER><<<num_blocks, 256>>>(p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			kOptimizerStatic8bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(p, g, return_updates, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1*sizeof(float)));
			kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 256>>>(p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			kOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(p, g, return_updates, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr,
																														quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
    case LION:
      // in lion, the momentum update happens after the parameter update
      kOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(p, g, return_updates, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr,
                                                            quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());

      CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1*sizeof(float)));
      kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 256>>>(p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      break;
		default:
			break;
	}
}

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g, T* return_updates,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
{

	int num_blocks = 0;
	switch(OPTIMIZER)
	{
		case ADAM:
			num_blocks = n/BLOCKSIZE_2STATE;
			num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
			kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE><<<num_blocks, BLOCKSIZE_2STATE/NUM_2STATE>>>(p, g, return_updates, state1, state2, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
		case RMSPROP:
    case ADAGRAD:
    case LION:
			num_blocks = n/BLOCKSIZE_1STATE;
			num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
			kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE><<<num_blocks, BLOCKSIZE_1STATE/NUM_1STATE>>>(p, g, return_updates, state1, beta1, beta2, eps, step, lr,
																														quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
	}
}



template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
  int num_blocks = n/2048;
  num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
	CUDA_CHECK_RETURN(cudaMemset(&gnorm_vec[step % 100], 0, 1*sizeof(float)));
  kPercentileClipping<T, 2048, 4><<<num_blocks, 512>>>(g, gnorm_vec, step, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	cublasStatus_t status;

			status = cublasGemmEx(context->m_handle,
					transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
					transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
					m, n,	k,
					alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, beta,
					C, CUDA_R_32I, ldc,
          CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	cublasStatus_t status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

			status = cublasGemmStridedBatchedEx(context->m_handle,
					transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
					transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
					m, n,	k,
					alpha, A, CUDA_R_8I, lda, (long long int)strideA, B, CUDA_R_8I, ldb, (long long int)strideB, beta,
					C, CUDA_R_32I, ldc, (long long int)strideC, batchCount,
          CUDA_R_32I, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


#ifdef NO_CUBLASLT
#else
template<int ORDER> cublasLtOrder_t get_order()
{
	switch(ORDER)
	{
		case ROW:
      return CUBLASLT_ORDER_ROW;
			break;
    case COL:
      return CUBLASLT_ORDER_COL;
      break;
    case COL32:
      return CUBLASLT_ORDER_COL32;
      break;
    case COL_TURING:
      return CUBLASLT_ORDER_COL4_4R2_8C;
      break;
    case COL_AMPERE:
      return CUBLASLT_ORDER_COL32_2R_4R4;
      break;
		default:
			break;
  }

	return CUBLASLT_ORDER_ROW;
}

template cublasLtOrder_t get_order<ROW>();
template cublasLtOrder_t get_order<COL>();
template cublasLtOrder_t get_order<COL32>();
template cublasLtOrder_t get_order<COL_TURING>();
template cublasLtOrder_t get_order<COL_AMPERE>();
#endif


template<int ORDER> int get_leading_dim(int dim1, int dim2)
{
	switch(ORDER)
	{
		case ROW:
      return dim2;
			break;
    case COL:
      return dim1;
      break;
    case COL32:
      // 32*row tiles
      return dim1*32;
      break;
    case COL_TURING:
      return 32*roundoff(dim1, 8);
      break;
    case COL_AMPERE:
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
		default:
			return 0;
			break;
  }
}

template int get_leading_dim<ROW>(int dim1, int dim2);
template int get_leading_dim<COL>(int dim1, int dim2);
template int get_leading_dim<COL32>(int dim1, int dim2);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(cublasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2)
{
#ifdef NO_CUBLASLT
#else
  cublasLtOrder_t orderA = get_order<SRC>();
  cublasLtOrder_t orderOut = get_order<TARGET>();
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);

  cublasLtMatrixLayout_t A_desc = NULL, out_desc = NULL;
  cublasLtMatrixTransformDesc_t A2Out_desc = NULL;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  float transformAlpha = 1.0f, transformBeta = 0.0f;


  if(DTYPE == 8)
  {
    checkCublasStatus(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8I, dim1, dim2, ldA));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&out_desc, CUDA_R_8I, dim1, dim2, ldOut));
  }
  else if(DTYPE == 32)
  {
    checkCublasStatus(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32I, dim1, dim2, ldA));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&out_desc, CUDA_R_32I, dim1, dim2, ldOut));
  }
  else
  {
    printf("ERROR WRONG TYPE FOR TRANSFORM: %i\n", DTYPE);
  }

  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(out_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderOut, sizeof(orderOut)));

  checkCublasStatus(cublasLtMatrixTransformDescCreate(&A2Out_desc, CUDA_R_32F));

  if(transpose){ checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(A2Out_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose))); }

  checkCublasStatus(cublasLtMatrixTransform(ltHandle, A2Out_desc, &transformAlpha, A, A_desc, &transformBeta, NULL, NULL, out, out_desc, 0));

  if (A_desc) checkCublasStatus(cublasLtMatrixLayoutDestroy(A_desc));
  if (out_desc) checkCublasStatus(cublasLtMatrixLayoutDestroy(out_desc));
  if (A2Out_desc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(A2Out_desc));
#endif
}

template void transform<int8_t, ROW, COL, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
{
#ifdef NO_CUBLASLT
	return ERR_NOT_IMPLEMENTED;
#else
    int has_error = 0;
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasLtPointerMode_t alphaVec = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t col_turing = CUBLASLT_ORDER_COL4_4R2_8C;
    cublasLtOrder_t col_ampere = CUBLASLT_ORDER_COL32_2R_4R4;

    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, ldb));

    has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    if(FORMATB == COL_TURING)
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_turing, sizeof(col_turing)));
    else
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_ampere, sizeof(col_ampere)));

    if(DTYPE_OUT == 32)
    {
      has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
      has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      int alpha = 1, beta = 0;
      has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int32_t*)C, Cdesc, (int32_t*)C, Cdesc, NULL, NULL, 0, 0));
    }
    else
    {
      has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F));
      has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, ldc));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      if(!SCALE_ROWS)
      {
        float alpha = 1.0f, beta = 0.0f;
        has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
      else
      {
        has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &alphaVec, sizeof(alphaVec)));
        has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
    }


    if (Cdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) has_error |= checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if(has_error == 1)
      printf("error detected");

    return has_error;
#endif // NO_CUBLASLT
}

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, half *bias, int numRows, int numCols)
{
  int threads = 512;
  int tileCols = fill_up_to_nearest_multiple(numCols, 32);
  int n = numRows*tileCols;
  int subtile_rows = 128;
  int tilesize = 32*subtile_rows;
  int num_blocks = numRows/subtile_rows;
  num_blocks += (numRows % subtile_rows == 0) ? 0 : 1;
  num_blocks = num_blocks*(tileCols/32);
  assert(threads <= tilesize);

  kdequant_mm_int32_fp16<4, 128, 512><<<num_blocks, threads>>>(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16
void getColRowStats(half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
{
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
	int row_tiles = (tiledRows/STATS_ROWS);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  if(nnz_threshold == 0.0)
    kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0><<<num_blocks, STATS_THREADS>>>(A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols);
  else if(nnz_threshold != 0.0)
    kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 1><<<num_blocks, STATS_THREADS>>>(A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

}

void doubleRowColQuant(half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, half *val, int *nnz_block_ptr, float threshold, int rows, int cols)
{
  int threads = 64;
  int items_per_thread = 4;
  int tile_cols = threads*items_per_thread;
  int tile_rows = 16;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;


  if(threshold > 0.0f)
    kDoubleRowColQuant<64, 4, 16, 64*4, 1><<<num_blocks, threads>>>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols);
  else
    kDoubleRowColQuant<64, 4, 16, 64*4, 0><<<num_blocks, threads>>>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  int threads = 256;
  int items_per_thread = 8;
  // we load 128 column values per warp
  int tile_cols = 32*items_per_thread;
  int tile_rows = 32;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  int outCols = fill_up_to_nearest_multiple(cols, 32);
  int outRows = fill_up_to_nearest_multiple(rows, 32);
  if(FORMAT == COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 32);
    else
      outRows = fill_up_to_nearest_multiple(rows, 32);
  }
  else
  {
    if(TRANSPOSE)
    {
      outCols = fill_up_to_nearest_multiple(rows, 32);
      outRows = cols;
    }
  }

  kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT><<<num_blocks, threads>>>(A, out, rows, cols, tiledCols, outRows, outCols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void spmm_coo(cusparseHandle_t handle, int *A_rowidx, int *A_colidx, half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, half *B, int ldc, half* C, bool transposed_B)
{

#ifdef NO_CUBLASLT
#else

    cusparseSpMatDescr_t descA;
    cusparseDnMatDescr_t descB, descC;

    float alpha = 1.0f;
    float beta = 0.0f;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUSPARSE( cusparseCreateCoo(&descA, A_rows, A_cols, A_nnz,
                                      A_rowidx, A_colidx, A_vals,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) );
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&descC, A_rows, B_cols, ldc, C,
                                        CUDA_R_16F, CUSPARSE_ORDER_ROW) );
    // Create dense matrix B
    if(transposed_B)
    {
      int tmp = A_cols;
      A_cols = B_cols;
      B_cols = tmp;
    }

    CHECK_CUSPARSE( cusparseCreateDnMat(&descB, A_cols, B_cols, ldb, B,
                                        CUDA_R_16F, CUSPARSE_ORDER_ROW) );
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 transposed_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, descA, descB, &beta, descC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    CUDA_CHECK_RETURN( cudaMalloc(&dBuffer, bufferSize) );

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 transposed_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, descA, descB, &beta, descC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(descA) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(descB) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(descC) );
    CUDA_CHECK_RETURN( cudaFree(dBuffer) );
#endif
}

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, T *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{

  kspmm_coo_very_sparse_naive<T, 8, BITS><<<nnz_rows, 256>>>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 256;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;

	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  kExtractOutliers<FORMAT><<<num_blocks, threads>>>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}




template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits)
{

	int num_blocks = (m+31)/32;

	//cout << num_blocks << endl;
	//cout << lda << endl;
	//cout << ldb << endl;
	//cout << ldc << endl;

	//cout << m << endl;
	//cout << n << endl;
	//cout << k << endl;
  //if(bits == 32)
    //gemm_device<T, 32, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 32, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
  if(bits == 16)
    //gemm_device<T, 16, 256><<< num_blocks, 256, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    gemm_device<T, 16, 160><<< num_blocks, 160, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 96><<< num_blocks, 96, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 64><<< num_blocks, 64, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
}

template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+31)/32;

	//cout << num_blocks << endl;
	//cout << lda << endl;
	//cout << ldb << endl;
	//cout << ldc << endl;

	//cout << m << endl;
	//cout << n << endl;
	//cout << k << endl;
  kgemm_4bit_inference<T, 96><<< num_blocks, 96, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
  //kgemm_4bit_inference<T, 256><<< num_blocks, 256, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
  //kgemm_4bit_inference<T, 160><<< num_blocks, 160, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
  //kgemm_4bit_inference<T, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
}

template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+3)/4;

  kgemm_4bit_inference_naive<T, 128, BITS><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B, absmax, datatype, out, lda, ldb, ldc, blocksize);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T, int FUNC> void func(T *A, T *B, T value, long n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  blocks = blocks > 65535 ? 65535 : blocks;
  kfunc<T, FUNC><<<blocks, 512>>>(A, B, value, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void func<float, FILL>(float *A, float *B, float value, long n);
template void func<unsigned char, FILL>(unsigned char *A, unsigned char *B, unsigned char value, long n);
template void func<float, ARANGE>(float *A, float *B, float value, long n);
template void func<float, _MUL>(float *A, float *B, float value, long n);

template void gemm_4bit_inference<half>(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<half, 16>(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<__nv_bfloat16, 16>(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<float, 32>(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize);

//template void gemm_host<float>(int m, int n, int k, float * A,  float* B,  float * out,  int lda, int ldb, int ldc, int bits);
template void gemm_host<half>(int m, int n, int k, half * A,  half* B,  half * out,  int lda, int ldb, int ldc, int bits);
template void extractOutliers<COL_TURING>(char * A, int *idx, char *out, int idx_size, int rows, int cols);
template void extractOutliers<COL_AMPERE>(char * A, int *idx, char *out, int idx_size, int rows, int cols);

template void spmm_coo_very_sparse_naive<half, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, half *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);
template void spmm_coo_very_sparse_naive<signed char, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, signed char *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

template int igemmlt<COL_TURING, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template void transformRowToFormat<COL32, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL32, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 1>(char * A, char *out, int rows, int cols);

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<half, 1, General8bit>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, General8bit>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, FP4>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, NF4>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, FP4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, NF4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 1, General8bit>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, General8bit>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, FP4>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, NF4>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);

template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<half, General8bit>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<half, FP4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<half, NF4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, FP4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, NF4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, gtype* return_updates, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(ADAM, __nv_bfloat16)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(LION, half)
MAKE_optimizer32bit(LION, float)
MAKE_optimizer32bit(LION, __nv_bfloat16)
MAKE_optimizer32bit(ADAGRAD, half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bit(name, gtype) \
template void optimizerStatic8bit<gtype, name>(gtype* p, gtype* g, gtype* return_updates, \
                unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale, int n); \

MAKE_optimizerStatic8bit(ADAM, half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, half)
MAKE_optimizerStatic8bit(RMSPROP, float)
MAKE_optimizerStatic8bit(LION, half)
MAKE_optimizerStatic8bit(LION, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, gtype* return_updates, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, LION);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);

MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADAM);

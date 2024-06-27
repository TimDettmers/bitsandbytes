// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include "ops.h"
#include "kernels.h"
#include <limits>
#include <BinSearch.h>
#include <cassert>
#include <common.h>
#include <dpct/lib_common_utils.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>


#define ERR_NOT_IMPLEMENTED 100

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

#define THREADS_ESTIMATE 512
#define NUM_ESTIMATE 8
#define BLOCK_ESTIMATE 4096
#define NUM_PER_THREAD 4

using namespace dnnl;

typedef sycl::ext::oneapi::bfloat16 bf16;

using namespace BinSearch;
using std::cout;
using std::endl;

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

//================================histogram 2d==============================================

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  
  int threads = 512;
  int num_blocks = n/threads;
  num_blocks = n % threads == 0 ? num_blocks : num_blocks + 1;
  int size = NUM_BLOCK;
	
  
  sycl::buffer<float, 1> buff_histogram(histogram,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_index1(index1,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_index2(index2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_src(src,sycl::range<1>(size));
  
  
  {
  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
    [&](sycl::handler &cgh) {
    
     sycl::accessor dacc_histogram(buff_histogram, cgh, sycl::read_write);
     sycl::accessor dacc_index1(buff_index1, cgh, sycl::read_write);
     sycl::accessor dacc_index2(buff_index2, cgh, sycl::read_write);
     sycl::accessor dacc_src(buff_src, cgh, sycl::read_write);
     
    
    cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kHistogramScatterAdd2D(histogram, index1, index2, src, maxidx1, n, item_ct1, dacc_histogram, dacc_index1, dacc_index2, dacc_src);
      });
    });
  }

}
//============================estimate quantiles===============================
template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  std::memset(code, 0, 256*sizeof(float));
  //DPCT_CHECK_ERROR(q_ct1.memset(code, 0, 256*sizeof(float)).wait());
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  
  sycl::buffer<T, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
        using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
        //using group_radix_sort = dpct::group::radix_sort<int, NUM_ESTIMATE>;
        size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
        sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), cgh);
        sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
        sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          
        
        auto std_numeric_limits_T_max_ct3 = std::numeric_limits<T>::max();

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
          [=](sycl::nd_item<3> item_ct1) {
            kEstimateQuantiles<T>(A, code, offset, std_numeric_limits_T_max_ct3, n, item_ct1, tacc, dacc_A, dacc_code);
            
          });
      });
  }
  
}

//============================k quantize ===============================
void quantize(float *code, float *A, unsigned char *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  sycl::buffer<float, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
      size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
      sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
      sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
      sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
      sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
      
      //__shared__ vars
      sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
      
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
        [=](sycl::nd_item<3> item_ct1) {
          kQuantize(code, A, out, n, item_ct1, smem_code_acc_ct1.get_pointer(), tacc, dacc_A, dacc_out, dacc_code);
        });
    });
  }
  
}

//============================k dequantize===============================
void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  unsigned char *buff_A;
  float *buff_out;
  *((void **)&buff_A) = sycl::malloc_device(size, dev_ct1, ctx);
  *((void **)&buff_out) = sycl::malloc_device(size, dev_ct1, ctx);
  q_ct1.memcpy((void*)(buff_out), (void*)(out), NUM_BLOCK);
  q_ct1.memcpy((void*)(buff_A), (void*)(A), NUM_BLOCK);

  
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
      //__shared__ vars
      sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
      
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
        [=](sycl::nd_item<3> item_ct1) {
          kDequantize(code, buff_A, buff_out, n, item_ct1, smem_code_acc_ct1.get_pointer());
        });
    });
   
   
   } 
  //back memcpy
  q_ct1.memcpy((void *)(out), (void*)(buff_out), NUM_BLOCK);
  q_ct1.memcpy((void*)(A), (void*)(buff_A), NUM_BLOCK); 
  
}

//============================quantize blockwise===============================

template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  sycl::context ctx = q_ct1.get_context();
  int size= NUM_BLOCK;
  for(int i=0; i< NUM_BLOCK; i++){ out[i]=out[(DATA_TYPE > 0) ? i/2 : i];};
  
  
  sycl::buffer<T, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_rand(rand,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax(absmax,sycl::range<1>(size));
  
  
  
  if(blocksize == 4096)
    
    
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          
           using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          
          //__shared__ vars for funtions
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 4096, 4, STOCHASTIC, 0>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 2048)
    
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 2048, 4, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 1024)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 1024, 4, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 512)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
         
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 512, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 256)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);

          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 256, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 128)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 128, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 64)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
           sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
      
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  
}


//============================k dequantize blockwise===============================
template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  sycl::buffer<unsigned char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax(absmax,sycl::range<1>(size));
  
  
  
    
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
      [&](sycl::handler &cgh){
      
              using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
              sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
              
              sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
              sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
              sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
              sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
  
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (n+tile_size-1)/tile_size) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
        [=](sycl::nd_item<3> item_ct1) {
          if(DATA_TYPE > 0){
          kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, A, absmax, out, blocksize/2, n, item_ct1, tacc, dacc_A, dacc_out, dacc_code, dacc_absmax); }
          else{
          kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, A, absmax, out, blocksize, n, item_ct1, tacc, dacc_A, dacc_out, dacc_code, dacc_absmax);
          }
        });
        
      });
 
}


//void matmul4bite(half *A, unsigned char *B, half*out, int lda, int ldb, int rowsA, int colsA, int colsB)
//{
//	int num_blocks = (colsB+32-1)/32;
//	kMatmul_inference_4bit<NF4, half, half, half><<<num_blocks, 256>>>(A, B, out, lda, ldb, rowsA, colsA, colsB);
//  CUDA_CHECK_RETURN(cudaPeekAtLastError());
//}



//============================32 bit optimizer===============================
template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
 try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  int size= NUM_BLOCK;  
   
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_state2(state2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_unorm(unorm, sycl::range<1>(size));
  
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{
				std::memset(unorm, 0, 1*sizeof(float));
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
       
        {
          dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              
              using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
              using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              
              
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
              sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
              
              sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
              sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
              sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
              sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
              
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_state1, dacc_state2, dacc_g, dacc_unorm);
                });
            });
        }
        
      }
			
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
            
            using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
            using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
             
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
            sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
            sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
            sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
            sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);   
            sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
                
           
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit2State<T, OPTIMIZER>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_state2, dacc_unorm);
			        });
			    });
			}
      
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
      if(max_unorm > 0.0f)
			{
        std::memset(unorm, 0, 1*sizeof(float));
				//DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
				
				{
				  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
				  q_ct1.submit(
				    [&](sycl::handler &cgh) {
                                  
             using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
                                
				      
		         cgh.parallel_for(
				        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
				        [=](sycl::nd_item<3> item_ct1) {
				          kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_g, dacc_state1, dacc_unorm);
				        });
				    });
				}
      }  

			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
             using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
       
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
         
			       cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit1State<T, OPTIMIZER>(g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_unorm);
			        });
			    });
			}
      
			break;
    case LION:
      // in lion, the momentum update after the parameter update
      
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
              using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
       
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
              
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizer32bit1State<T, OPTIMIZER>(g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_unorm);
              });
          });
      }
     

      if(max_unorm > 0.0f)
      {
        std::memset(unorm, 0, 1*sizeof(float));
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
        
        {
          dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
                         
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_g, dacc_state1, dacc_unorm);
                });
            });
        }
        
      }
      break;
	}
  
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}



//============================8 bit optimizer===============================

#define NUM8BIT 16
#define NUM_THREADS 256
#define NUM_PER_BLOCK 4096


template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g,
                unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr,
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
 try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state2(state2,sycl::range<1>(size));
  
  sycl::buffer<float, 1> buff_quantiles1(quantiles1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_quantiles2(quantiles2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_max1(max1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_max2(max2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_new_max1(new_max1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_new_max2(new_max2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_unorm(unorm,sycl::range<1>(size));
  

  if(max_unorm > 0.0f){ 
  std::memset(unorm, 0, 1*sizeof(float)); }

	switch(OPTIMIZER)
	{
		case ADAM:
      std::memset(new_max1, 0, 1*sizeof(float));
      std::memset(new_max2, 0, 1*sizeof(float));
			
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait());
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max2, 0, 1*sizeof(float)).wait());
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
                                              
             using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
             
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles2(buff_quantiles2, cgh, sycl::read_write);
             sycl::accessor dacc_max1(buff_max1, cgh, sycl::read_write);
             sycl::accessor dacc_max2(buff_max2, cgh, sycl::read_write);
             sycl::accessor dacc_new_max1(buff_new_max1, cgh, sycl::read_write);
             sycl::accessor dacc_new_max2(buff_new_max2, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
                         
             
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);        
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
              

			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER>(p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), smem_quantiles2_acc_ct1.get_pointer(), tacc, dacc_g, dacc_state1, dacc_state2, dacc_unorm, dacc_quantiles1, dacc_quantiles2, dacc_max1, dacc_max2, dacc_new_max1 , dacc_new_max2);
			        });
			    });
			}
			
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
                    
            using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
             
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles2(buff_quantiles2, cgh, sycl::read_write);
             sycl::accessor dacc_max1(buff_max1, cgh, sycl::read_write);
             sycl::accessor dacc_max2(buff_max2, cgh, sycl::read_write);
             sycl::accessor dacc_new_max1(buff_new_max1, cgh, sycl::read_write);
             sycl::accessor dacc_new_max2(buff_new_max2, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);

            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);        
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit2State<T, OPTIMIZER>(p, g, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), smem_quantiles2_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1, dacc_state2, dacc_unorm, dacc_quantiles1, dacc_quantiles2, dacc_max1, dacc_max2, dacc_new_max1 , dacc_new_max2);
			        });
			    });
			}
			
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			
      std::memset(new_max1, 0, 1*sizeof(float));
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait());
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
             using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_max1(buff_max1, cgh, sycl::read_write);
             sycl::accessor dacc_new_max1(buff_new_max1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
             
             //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);                   
       
            cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_state1, dacc_unorm, dacc_quantiles1, dacc_max1, dacc_new_max1);
			        });
			    });
			}
			
			
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
             
             using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_max1(buff_max1, cgh, sycl::read_write);
             sycl::accessor dacc_new_max1(buff_new_max1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
             
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1,smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1, dacc_unorm, dacc_quantiles1, dacc_max1, dacc_new_max1);
			        });
			    });
			}
		
			break;
    case LION:
      
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write); 
             
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_max1(buff_max1, cgh, sycl::read_write);
             sycl::accessor dacc_new_max1(buff_new_max1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
                        
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
            
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1, dacc_unorm, dacc_quantiles1, dacc_max1, dacc_new_max1);
              });
          });
      }
       std::memset(new_max1, 0, 1*sizeof(float));
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait());
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_max1(buff_max1, cgh, sycl::read_write);
             sycl::accessor dacc_new_max1(buff_new_max1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
             
             
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
            
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
              [=](sycl::nd_item<3> item_ct1) {
                kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_state1, dacc_unorm, dacc_quantiles1, dacc_max1, dacc_new_max1);
              });
          });
      }
     
      break;
		default:
			break;
	}
 
}catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


//============================8 bit blockwise optimizer===============================

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
 try {
    
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = BLOCKSIZE_2STATE;
  
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state2(state2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_quantiles1(quantiles1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_quantiles2(quantiles2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax1(absmax1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax2(absmax2,sycl::range<1>(size));
  
   
	switch(OPTIMIZER)
	{
		case ADAM:
			num_blocks = n/BLOCKSIZE_2STATE;
			num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			     
            using group_load = dpct::group::workgroup_load<NUM_2STATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles2(buff_quantiles2, cgh, sycl::read_write);
             sycl::accessor dacc_absmax1(buff_absmax1, cgh, sycl::read_write);
             sycl::accessor dacc_absmax2(buff_absmax2, cgh, sycl::read_write);
            
            //__shared__ vars
            sycl::local_accessor<float, 2> smem_quantiles1_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);        
			      sycl::local_accessor<float, 2> smem_quantiles2_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);
			      sycl::local_accessor<float, 1> smem_exchange1_acc_ct1(sycl::range<1>(1), cgh);
			      sycl::local_accessor<float, 1> smem_exchange2_acc_ct1(sycl::range<1>(1), cgh);
			      
     
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_2STATE/NUM_2STATE), sycl::range<3>(1, 1, BLOCKSIZE_2STATE/NUM_2STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE>(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n,item_ct1,  smem_quantiles1_acc_ct1, smem_quantiles2_acc_ct1,smem_exchange1_acc_ct1.get_pointer(), smem_exchange2_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1, dacc_state2, dacc_quantiles1, dacc_quantiles2, dacc_absmax1, dacc_absmax2);
			        });
			    });
			}
		
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
		case LION:
			num_blocks = n/BLOCKSIZE_1STATE;
			num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
            using group_load = dpct::group::workgroup_load<NUM_1STATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_absmax1(buff_absmax1, cgh, sycl::read_write);
             
             
            //__shared__ vars
            sycl::local_accessor<float, 2> smem_quantiles1_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);
			      sycl::local_accessor<float, 1> smem_exchange1_acc_ct1(sycl::range<1>(1), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE), sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE>(p, g, state1, beta1, beta2, eps, step, lr, quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n, item_ct1, smem_quantiles1_acc_ct1, smem_exchange1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1,  dacc_quantiles1, dacc_absmax1);
			        });
			    });
			}
			
		break;
	}

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//============================percentile clipping===============================

template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::context ctx = q_ct1.get_context();
    
    int num_blocks = n/2048;
    num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
    int size = NUM_BLOCK;
  
    sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
    std::memset(&gnorm_vec[step % 100], 0, 1*sizeof(float));
    sycl::buffer<float, 1> buff_gnorm_vec(gnorm_vec, sycl::range<1>(size));
  
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
        
         using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
         size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
         sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
         sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
         sycl::accessor dacc_gnorm_vec(buff_gnorm_vec, cgh, sycl::read_write);
           
          //sycl::local_accessor<float, 1> dacc_gnorm_vec(sycl::range<1>(size), cgh);
                   
      cgh.parallel_for(    
       sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
      [=](sycl::nd_item<3> item_ct1) {
        kPercentileClipping<T, 2048, 4>(g, gnorm_vec, step, n, item_ct1, tacc, dacc_g, dacc_gnorm_vec.get_pointer());
      });
    });
  }
 
}

//==========================dequant mm int 32 fp16==========================

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, sycl::half *out, float* newRowStats, float* newcolStats, sycl::half *bias, int numRows, int numCols)
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
  int size = NUM_BLOCK;
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();

  
  sycl::buffer<int, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_rowStats (rowStats, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_colStats (colStats, sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_out (out, sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_bias (bias, sycl::range<1>(size));
  
  
  
  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
		[&](sycl::handler &cgh) {
            
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);  
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          
          sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
          sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_bias(buff_bias, cgh, sycl::read_write);
          
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_rowStats_acc_ct1(sycl::range<1>(256), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE), sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
  kdequant_mm_int32_fp16<4, 128, 512>(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n, item_ct1,smem_rowStats_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rowStats, dacc_colStats, dacc_out, dacc_bias);
           });
  
  });
  
}

//========================GEMM============================

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
 try {
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	int status;

   dpct::gemm(*context->m_handle, transposeA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, transposeB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, m, n, k, alpha, A, dpct::library_data_t::real_int8, lda, B, dpct::library_data_t::real_int8, ldb, beta, C, dpct::library_data_t::real_int32, ldc, dpct::library_data_t::real_int32);


}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
 try {
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	int status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

   dpct::gemm_batch(*context->m_handle, transposeA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, transposeB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, m, n, k, alpha, A, dpct::library_data_t::real_int8, lda, (long long int)strideA, B, dpct::library_data_t::real_int8, ldb, (long long int)strideB, beta, C, dpct::library_data_t::real_int32, ldc, (long long int)strideC, batchCount, dpct::library_data_t::real_int32);

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


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

//=================================transform GEMM==============================

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform( T *A, T *out, int dim1, int dim2)
{

  using namespace dnnl;
  using tag = memory::format_tag;
  using dt = memory::data_type;
  void *Aout;
  auto dev = sycl::device(sycl::gpu_selector_v);
  auto ctx = sycl::context(dev);
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);
  int ldAOut = get_leading_dim<TARGET>(dim1, dim2);
  
  dnnl::engine engine = sycl_interop::make_engine(dev, ctx);
  // column major 
  const memory::dims a_strides = memory::dims {1, ldA};
  const auto a_md = DTYPE ==32 ? memory::desc({dim1, dim2}, dt::s32, a_strides) : memory::desc({dim1, dim2}, dt::s8, a_strides);
  const memory::dims out_strides = memory::dims {ldOut, 1};
  const auto out_md = DTYPE ==32 ? memory::desc({dim1, dim2}, dt::s32, out_strides) : memory::desc({dim1, dim2}, dt::s8, out_strides);
  const memory::dims aout_strides = memory::dims {ldAOut, 1};
  const auto aout_md = DTYPE == 32 ? memory::desc({dim1, dim2}, dt::s32, aout_strides) : memory::desc({dim1, dim2}, dt::s8, aout_strides);
  
  //memory align
  memory a_mem(a_md, engine, A);
  memory out_mem(out_md, engine, out);
  memory aout_mem(aout_md, engine, Aout);
  
  //create dnnl stream
  auto q_ct1 = sycl::queue(ctx, dev);
  dnnl::stream stream = sycl_interop::make_stream(engine, q_ct1);
  
  primitive_attr attr;
  
  auto matmul_pd = matmul::primitive_desc(engine, a_md, out_md, aout_md, attr);
  auto matmul_prim = matmul(matmul_pd);
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, out_mem});
  matmul_args.insert({DNNL_ARG_DST, aout_mem});

  matmul_prim.execute(stream, matmul_args);
  stream.wait();

}


template void transform<int8_t, ROW, COL, false, 8>(int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>( int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>( int32_t *A, int32_t *out, int dim1, int dim2);


//========================igemmlt============================================

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
 try {
    
    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto dev = sycl::device(sycl::gpu_selector_v);
    auto ctx = sycl::context(dev);
    
    dnnl::engine engine = sycl_interop::make_engine(dev, ctx);
    // column major 
    const memory::dims a_strides = memory::dims {1, lda};
    const auto a_md = memory::desc({m, k}, dt::s8, a_strides);
    const memory::dims b_strides = memory::dims {ldb, 1};
    const auto b_md = memory::desc({k, n}, dt::s8, b_strides);
    const memory::dims c_strides = memory::dims {ldc, 1};
    const auto c_md = DTYPE_OUT == 32 ? memory::desc({m, n}, dt::s32, c_strides) : memory::desc({m, n}, dt::s8, c_strides);
    
    //memory align
    memory a_mem(a_md, engine);
    memory b_mem(b_md, engine);
    memory c_mem(c_md, engine);
    memory scales_C_mem({{1}, dt::f32, {1}}, engine, row_scale);
    
    //create dnnl stream
    auto q_ct1 = sycl::queue(ctx, dev);
    dnnl::stream stream = sycl_interop::make_stream(engine, q_ct1);
    
    primitive_attr attr;
    if (SCALE_ROWS) {
        attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 1 << 1);
    }
    
    auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md, attr);
    auto matmul_prim = matmul(matmul_pd);
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});

    if (SCALE_ROWS) {
      matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_C_mem});
    }
    matmul_prim.execute(stream, matmul_args);
    stream.wait();

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


template int igemmlt<COL_TURING, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

//===========================gemm_host============================================

template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits)
{

	int num_blocks = (m+31)/32;
 
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_B (B, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out (out, sycl::range<1>(size));
  
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
    {
      dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
      dpct::get_in_order_queue().submit(
        [&](sycl::handler &cgh) {
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(224/*8*16 + (2*16*(batch_size_warps-1))*/), cgh);
          sycl::local_accessor<T, 1> smem_B_acc_ct1(sycl::range<1>(4192/*2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))*/), cgh);

          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 160), sycl::range<3>(1, 1, 160)), 
            [=](sycl::nd_item<3> item_ct1) {
              gemm_device<T, 16, 160>(m, n, k, A, B, out, lda, ldb, ldc, item_ct1, smem_A_acc_ct1.get_pointer(), smem_B_acc_ct1.get_pointer(),
              dacc_A, dacc_B, dacc_out);
            });
        });
    }
    //gemm_device<T, 16, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 96><<< num_blocks, 96, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 64><<< num_blocks, 64, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
   
  
}


//============================gemm 4bit inference ================================

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
 
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_B (B, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out (out, sycl::range<1>(size));
 
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        
        sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
        sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
        sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);  
        
        //__shared__ vars
        sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(176/*8*16 + (16*(batch_size_warps-1))*/), cgh);
        sycl::local_accessor<unsigned char, 1> smem_B_acc_ct1(sycl::range<1>(4192/*2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))*/), cgh);
        sycl::local_accessor<T, 1> smem_C_acc_ct1(sycl::range<1>(8*32), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 96), sycl::range<3>(1, 1, 96)), 
          [=](sycl::nd_item<3> item_ct1) {
            kgemm_4bit_inference<T, 96>(m, n, k, A, B, absmax, out, lda, ldb, ldc, blocksize, item_ct1, smem_A_acc_ct1.get_pointer(), smem_B_acc_ct1.get_pointer(), smem_C_acc_ct1.get_pointer(), dacc_A, dacc_B, dacc_out);
          });
      });
  }
  
}


//============================gemm 4 bit inference naive =================

template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+3)/4;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_B (B, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out (out, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax(absmax, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_datatype(datatype, sycl::range<1>(size));
  
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        
        sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
        sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
        sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);  
        sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
        sycl::accessor dacc_datatype(buff_datatype, cgh, sycl::read_write);
        sycl::local_accessor<T, 1> quant_map_acc_ct1(sycl::range<1>(16), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            kgemm_4bit_inference_naive<T, 128, BITS>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, item_ct1, quant_map_acc_ct1.get_pointer(), dacc_A, dacc_B, dacc_out, dacc_absmax, dacc_datatype);
          });
      });
  }
 
}
//================================spm coo==================================

void spmm_coo(int *A_rowidx, int *A_colidx, sycl::half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, sycl::half *B, int ldc, sycl::half* C, bool transposed_B)
{ 

  try{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  
    dpct::sparse::sparse_matrix_desc_t descA;
    std::shared_ptr<dpct::sparse::dense_matrix_desc> descB, descC;

    float alpha = 1.0f;
    float beta = 0.0f;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    
    // Create dense matrix C
    
    descC = std::make_shared<dpct::sparse::dense_matrix_desc>(A_rows, B_cols, ldc, C, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major);
    // Create dense matrix B
    if(transposed_B)
    {
      int tmp = A_cols;
      A_cols = B_cols;
      B_cols = tmp;
    }

    
    descB = std::make_shared<dpct::sparse::dense_matrix_desc>(A_cols, B_cols, ldb, B, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major);
    // allocate an external buffer if needed
    
    bufferSize = 0;
    
    dBuffer = (void *)sycl::malloc_device(bufferSize, q_ct1);

    
    dpct::sparse::spmm(q_ct1, oneapi::mkl::transpose::nontrans, transposed_B ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, &alpha, descA, descB, &beta, descC, dpct::library_data_t::real_float);
    // destroy matrix/vector descriptors
    descA.reset();
    descB.reset();
    descC.reset();
    sycl::free(dBuffer, q_ct1);
    
  }
  catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
  }

}

//===============================spm _coo _very _sparse=========================

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, T *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int size = NUM_BLOCK;
  
  sycl::buffer<int, 1> buff_max_count(max_count,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_max_idx(max_idx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_offset_rowidx(offset_rowidx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_rowidx(rowidx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_colidx(colidx,sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_values(values,sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_B(B, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_dequant_stats(dequant_stats,sycl::range<1>(size));
  

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
        
         sycl::accessor dacc_max_count(buff_max_count, cgh, sycl::read_write);
         sycl::accessor dacc_max_idx(buff_max_idx, cgh, sycl::read_write);
         sycl::accessor dacc_offset_rowidx(buff_offset_rowidx, cgh, sycl::read_write);
         sycl::accessor dacc_colidx(buff_colidx, cgh, sycl::read_write);
         sycl::accessor dacc_rowidx(buff_rowidx, cgh, sycl::read_write);
         sycl::accessor dacc_values(buff_values, cgh, sycl::read_write);
         sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
         sycl::accessor dacc_dequant_stats(buff_dequant_stats, cgh, sycl::read_write);
         sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
         
        
        //smem
        sycl::local_accessor<sycl::half, 1> smem_dequant_stats_acc_ct1(sycl::range<1>(2048/*SMEM_SIZE*/), cgh);
   
        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, nnz_rows) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
          [=](sycl::nd_item<3> item_ct1) {
            kspmm_coo_very_sparse_naive<T, 8, BITS>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB, item_ct1, smem_dequant_stats_acc_ct1.get_pointer(), dacc_max_count, dacc_max_idx, dacc_offset_rowidx, dacc_rowidx, dacc_colidx, dacc_values, dacc_B, dacc_out, dacc_dequant_stats);
          });
      });
  }
  
}

//======================================non gemm 2d quants============================================

//===========================Row col stats=================================

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16
void getColRowStats(sycl::half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
	int row_tiles = (tiledRows/STATS_ROWS);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;
  
  int size = NUM_BLOCK;
  
  sycl::buffer<sycl::half, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_nnz_count_row(nnz_count_row,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_rowStats(rowStats,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_colStats(colStats,sycl::range<1>(size));
  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
            sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
            sycl::accessor dacc_nnz_count_row(buff_nnz_count_row, cgh, sycl::read_write);
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_absmax_values_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<int, 1> smem_row_nnz_values_acc_ct1(sycl::range<1>(256), cgh);
                        
                        
       cgh.parallel_for(      
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
            [=](sycl::nd_item<3> item_ct1) {
            if(nnz_threshold == 0.0){
              kgetColRowStats<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats,
               nnz_count_row,    nnz_threshold, rows, cols, tiledRows, tiledCols,item_ct1, 
               smem_row_absmax_values_acc_ct1.get_pointer(), smem_row_nnz_values_acc_ct1.get_pointer(), tacc, 
               dacc_A, dacc_rowStats, dacc_colStats, dacc_nnz_count_row);
               }
            else if(nnz_threshold != 0.0){
              kgetColRowStats<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats,
             nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols,item_ct1, 
             smem_row_absmax_values_acc_ct1.get_pointer(),smem_row_nnz_values_acc_ct1.get_pointer(), 
             tacc, dacc_A, dacc_rowStats, dacc_colStats, dacc_nnz_count_row);
            }
            });
       });

}


//===================================double row col quant======================

void doubleRowColQuant(sycl::half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, sycl::half *val, int *nnz_block_ptr, float threshold, int rows, int cols)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<sycl::half, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out_col_normed(out_col_normed,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out_row_normed(out_row_normed,sycl::range<1>(size));
  
  sycl::buffer<float, 1> buff_rowStats(rowStats,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_colStats(colStats,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_rowidx(rowidx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_colidx(colidx,sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_val(val,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_nnz_block_ptr(nnz_block_ptr,sycl::range<1>(size));
  
  
  
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
  num_blocks = row_tiles * col_tiles;


  if(threshold > 0.0f)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            sycl::accessor dacc_out_col_normed(buff_out_col_normed, cgh, sycl::read_write);
            sycl::accessor dacc_out_row_normed(buff_out_row_normed, cgh, sycl::read_write);

            sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
            sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
            sycl::accessor dacc_rowidx(buff_rowidx, cgh, sycl::read_write);
            sycl::accessor dacc_colidx(buff_colidx, cgh, sycl::read_write);
            sycl::accessor dacc_val(buff_val, cgh, sycl::read_write);
            sycl::accessor dacc_nnz_block_ptr(buff_nnz_block_ptr, cgh, sycl::read_write);


            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_stats_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<unsigned int, 1> smem_nnz_row_idx_acc_ct1(sycl::range<1>(256), cgh);
                        
        cgh.parallel_for(      
           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
           [=](sycl::nd_item<3> item_ct1) {
          
                kDoubleRowColQuant<STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols, item_ct1, smem_row_stats_acc_ct1.get_pointer(), smem_nnz_row_idx_acc_ct1.get_pointer(), tacc, dacc_A, dacc_out_col_normed, dacc_out_row_normed, dacc_rowStats, dacc_colStats, dacc_rowidx, dacc_colidx, dacc_val, dacc_nnz_block_ptr);
          });
      });
    }
  else
    {
  
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            sycl::accessor dacc_out_col_normed(buff_out_col_normed, cgh, sycl::read_write);
            sycl::accessor dacc_out_row_normed(buff_out_row_normed, cgh, sycl::read_write);
            
            sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
            sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
            sycl::accessor dacc_rowidx(buff_rowidx, cgh, sycl::read_write);
            sycl::accessor dacc_colidx(buff_colidx, cgh, sycl::read_write);
            sycl::accessor dacc_val(buff_val, cgh, sycl::read_write);
            sycl::accessor dacc_nnz_block_ptr(buff_nnz_block_ptr, cgh, sycl::read_write);

            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_stats_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<unsigned int, 1> smem_nnz_row_idx_acc_ct1(sycl::range<1>(256), cgh);
            
                        
        cgh.parallel_for(      
           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
           [=](sycl::nd_item<3> item_ct1) {
          
                kDoubleRowColQuant<STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols,item_ct1, smem_row_stats_acc_ct1.get_pointer(), smem_nnz_row_idx_acc_ct1.get_pointer(),  tacc, dacc_A, dacc_out_col_normed, dacc_out_row_normed, dacc_rowStats, dacc_colStats, dacc_rowidx, dacc_colidx, dacc_val, dacc_nnz_block_ptr);
          });
      });
  
  }
  
}
//========================== transform row to format================================
template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  
  
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
  num_blocks = row_tiles * col_tiles;

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

  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
     
     sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
     sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
            
      
    //__shared__ vars
      sycl::local_accessor<char, 1> smem_data_acc_ct1(sycl::range<1>(32*33*8), cgh);

      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
        [=](sycl::nd_item<3> item_ct1) {
          kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT>(A, out, rows, cols, tiledCols, outRows, outCols, item_ct1, smem_data_acc_ct1.get_pointer(), dacc_A, dacc_out);
        });
    });
  
}

//===========================extract outliers===========================

template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 512;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;
  int size = NUM_BLOCK;
	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_idx(idx,sycl::range<1>(size));
  
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
      sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
     sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
     sycl::accessor dacc_idx(buff_idx, cgh, sycl::read_write);
     
    
    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
      [=](sycl::nd_item<3> item_ct1) {
           kExtractOutliers<FORMAT>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols, item_ct1, dacc_A, dacc_out, dacc_idx);
    });
   });
 
}

//==================================func===========================

template <typename T, int FUNC> void func(T *A, T *B, T value, long n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  blocks = blocks > 65535 ? 65535 : blocks;
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
  cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kfunc<T, FUNC>(A, B, value, n, item_ct1);
    });
  });
  
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void func<float, FILL>(float *A, float *B, float value, long n);
template void func<unsigned char, FILL>(unsigned char *A, unsigned char *B, unsigned char value, long n);
template void func<float, ARANGE>(float *A, float *B, float value, long n);
template void func<float, _MUL>(float *A, float *B, float value, long n);

template void gemm_4bit_inference<sycl::half>(int m, int n, int k, sycl::half * A,  unsigned char* B,  float *absmax, sycl::half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<sycl::half, 16>(int m, int n, int k, sycl::half * A,  unsigned char* B,  float *absmax, float *datatype, sycl::half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<bf16, 16>(int m, int n, int k, bf16 * A,  unsigned char* B,  float *absmax, float *datatype, bf16 * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<float, 32>(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize);

//template void gemm_host<float>(int m, int n, int k, float * A,  float* B,  float * out,  int lda, int ldb, int ldc, int bits);
template void gemm_host<sycl::half>(int m, int n, int k, sycl::half * A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc, int bits);
template void extractOutliers<COL_TURING>(char * A, int *idx, char *out, int idx_size, int rows, int cols);
template void extractOutliers<COL_AMPERE>(char * A, int *idx, char *out, int idx_size, int rows, int cols);

template void spmm_coo_very_sparse_naive<sycl::half, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);
template void spmm_coo_very_sparse_naive<signed char, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

template int igemmlt<COL_TURING, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template void transformRowToFormat<COL32, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL32, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 1>(char * A, char *out, int rows, int cols);

template void estimateQuantiles(sycl::half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<sycl::half, 1, General8bit>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<sycl::half, 0, General8bit>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<sycl::half, 0, FP4>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<sycl::half, 0, NF4>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, FP4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, NF4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 1, General8bit>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 0, General8bit>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 0, FP4>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 0, NF4>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);

template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<sycl::half, General8bit>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n);
template void dequantizeBlockwise<sycl::half, FP4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n);
template void dequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n);
template void dequantizeBlockwise<bf16, General8bit>(float *code, unsigned char *A, float *absmax, bf16 *out, int blocksize, const int n);
template void dequantizeBlockwise<bf16, FP4>(float *code, unsigned char *A, float *absmax, bf16 *out, int blocksize, const int n);
template void dequantizeBlockwise<bf16, NF4>(float *code, unsigned char *A, float *absmax, bf16 *out, int blocksize, const int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, sycl::half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(ADAM, bf16)
MAKE_optimizer32bit(MOMENTUM, sycl::half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, sycl::half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(LION, sycl::half)
MAKE_optimizer32bit(LION, float)
MAKE_optimizer32bit(LION, bf16)
MAKE_optimizer32bit(ADAGRAD, sycl::half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bit(name, gtype) \
template void optimizerStatic8bit<gtype, name>(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale, int n); \

MAKE_optimizerStatic8bit(ADAM, sycl::half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, sycl::half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, sycl::half)
MAKE_optimizerStatic8bit(RMSPROP, float)
MAKE_optimizerStatic8bit(LION, sycl::half)
MAKE_optimizerStatic8bit(LION, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(sycl::half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(sycl::half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(sycl::half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(sycl::half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(bf16, LION);
MAKE_optimizerStatic8bitBlockwise(sycl::half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(sycl::half * g, float *gnorm_vec, int step, const int n);

MAKE_optimizerStatic8bitBlockwise(bf16, ADAM);

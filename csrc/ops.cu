// Copyright (c) Facebook, Inc. and its affiliates. 
//   
// This source code is licensed under the MIT license found in the 
// LICENSE file in the root directory of this source tree.

#include <ops.cuh>
#include <kernels.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>


using std::cout;
using std::endl;

#define CPU_BLOCK_SIZE 4096

static const float powerTable8[8] = {0.0f, 1.0f, 1e1f, 1e2f, 1e3f, 1e4f, 1e5f, 1e6f};
static const float scaleTable[8] = {142.22222222222f, 71.1111111111f, 35.55555555555f, 17.7777777777f, 8.8888888888f, 4.444444444444f, 2.22222222222f, 1.11111111111f};
__forceinline__ unsigned char cQuantizeDynamic(float x)
{
    if(x == 0.0f){ return 0; }
    if(x > 0.996485f){ return 1; }
    if(x < -0.996485){ return 129; }

    unsigned char out = 0;
    float absx = fabs(x);
    int exp10 = abs(floor(log10f(absx)));
    float frac = absx*powerTable8[exp10];

    // if exp = 1 then there are 6 bits for the linear quantization -> normalized by 2^6-1 (7-exp10)
    // if exp = 2 then there are 5 bits for the linear quantization -> normalized by 2^5-1
    // Reasoning: We get a number between [0.1, 1.0]
    // We first shift it so zero (frac-0.1) to put it in [0, 0.9]
    // Now we divide the interval in 2^bits sections by dividing through 0.9/2^bits
    // However, we end up directly on the quantization bins, e.g. for 2^1 we have:
    // [0..0.45...0.9]/(0.9/2) = [0..1..2]
    // so if we want to round to [0..1] we need to subtract 0.5 to center
    // the values in the middle between quantization bins
    //float base = powf(2.0f, 7-exp10);
    //float inc = 0.9f/(base);
    // 1/inc =  base/0.9f = 2.0^(7-exp10)/0.9
    // for exp10 0..7: scaleTable = [142.2222, 71.11111, 35.55556, 17.77778, 8.888889, 4.444444, 2.222222, 1.111111]
    int frac_int = round(((frac-0.1f))*scaleTable[exp10]-0.5f);

    out |= signbit(x) << 7;
    out |= 1 << (7-exp10);
    out += frac_int;

    if(out == 1) 
      return 0;
    else 
      return out;
}

struct quantize_block_args
{
  float *A;
  unsigned char *cA;
  float *absmax;
  unsigned char *cout;
  float *fout;
  int block_end;
  int block_idx;
};

void *quantize_block(void *arguments)
{
  // 1. find absmax in block
  // 2. divide input value by absmax to normalize into [-1.0, 1.0]
  // 3. do bitwise encoding
  // 4. store index

  struct quantize_block_args *args = (quantize_block_args*)arguments;

  // 1. find absmax in block
  float absmax_block = -FLT_MAX;
  for (int i = args->block_idx; i < args->block_end; i++)
    absmax_block = fmax(absmax_block, fabs(args->A[i]));

  args->absmax[args->block_idx/CPU_BLOCK_SIZE] = absmax_block;

  for (int i = args->block_idx; i < args->block_end; i++)
  {
    // 2. divide input value by absmax to normalize into [-1.0, 1.0]
    float normed_value = args->A[i]/absmax_block;

    // 3. do binary search to find the closest value
    // 4. store index
    args->cout[i] = cQuantizeDynamic(normed_value);
  }

  return NULL;
}

void quantize_cpu(float *A, float *absmax, unsigned char *out, int n)
{

  int num_blocks = n/CPU_BLOCK_SIZE;
  num_blocks += n % CPU_BLOCK_SIZE == 0 ? 0 : 1;

  pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t)*num_blocks);
  struct quantize_block_args **args = (quantize_block_args**)malloc(num_blocks*sizeof(quantize_block_args*));

  for(int i = 0; i < num_blocks; i++)
    args[i] = (quantize_block_args*)malloc(sizeof(quantize_block_args));

  for(int block_idx = 0; block_idx < n; block_idx+=CPU_BLOCK_SIZE)
  {
    int valid_items = n-block_idx >= CPU_BLOCK_SIZE ? CPU_BLOCK_SIZE : n - block_idx;
    int block_end = block_idx + valid_items;

    struct quantize_block_args *arg = args[block_idx/CPU_BLOCK_SIZE];
    arg->A = A;
    arg->absmax = absmax;
    arg->cout = out;
    arg->block_end = block_end;
    arg->block_idx = block_idx;
 
    pthread_create(&threads[block_idx/CPU_BLOCK_SIZE], NULL, &quantize_block, (void *)arg);
  }

  for(int i = 0; i < num_blocks; i++)
    int err = pthread_join(threads[i], NULL);

  free(threads);
  for(int i = 0; i < num_blocks; i++)
    free(args[i]);
  free(args);
}

void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, int n)
{
  for(int block_idx = 0; block_idx < n; block_idx+=CPU_BLOCK_SIZE)
  {
    int valid_items = n-block_idx >= CPU_BLOCK_SIZE ? CPU_BLOCK_SIZE : n - block_idx;
    int block_end = block_idx + valid_items;
    for (int i = block_idx; i < block_end; i++)
      out[i] = code[A[i]]*absmax[block_idx/CPU_BLOCK_SIZE];
  }
}

template <typename T, int FUNC> void func(T *A, T value, long n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  blocks = blocks > 65535 ? 65535 : blocks;
  kfunc<T, FUNC><<<blocks, 512>>>(A, value, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  kHistogramScatterAdd2D<<<blocks, 512>>>(histogram, index1, index2, src, maxidx1, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
	CUDA_CHECK_RETURN(cudaMemset(code, 0, 256*sizeof(float)));
  kEstimateQuantiles<T><<<blocks, 512>>>(A, code, offset, std::numeric_limits<T>::max(), n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

#define ITEMS 4
template <typename T, int STOCHASTIC> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, const int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
  kQuantizeBlockwise<T, 4096, ITEMS, STOCHASTIC><<<blocks, 4096/ITEMS>>>(code, A, absmax, out, rand, rand_offset, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T, int BLOCK_SIZE> void quantizeBlockwiseDynamic(T *A, float *absmax, unsigned char *out, int n)
{
  int blocks = n/BLOCK_SIZE;
  blocks = n % BLOCK_SIZE == 0 ? blocks : blocks + 1;
  kQuantizeBlockwiseDynamic<T, BLOCK_SIZE, ITEMS><<<blocks, BLOCK_SIZE/ITEMS>>>(A, absmax, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T, int BLOCK_SIZE> void dequantizeBlockwiseDynamic(unsigned char *A, float *absmax, T *out, int n)
{
  int blocks = n/BLOCK_SIZE;
  blocks = n % BLOCK_SIZE == 0 ? blocks : blocks + 1;
  kDequantizeBlockwiseDynamic<T, BLOCK_SIZE, 512, ITEMS><<<blocks, BLOCK_SIZE/ITEMS>>>(A, absmax, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  int blocks = n/blocksize;
  blocks = n % blocksize == 0 ? blocks : blocks + 1;
  if(blocksize == 4096)
    kDequantizeBlockwise<T, 4096, 1024, 4><<<blocks, 4096/4>>>(code, A, absmax, out, n);
  else if(blocksize == 2048)
    kDequantizeBlockwise<T, 2048, 512, 4><<<blocks, 2048/4>>>(code, A, absmax, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p, 
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{ 
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
        kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<blocks, 512>>>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
      }
			kOptimizer32bit2State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:

      if(max_unorm > 0.0f)
			{ 
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
				kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8><<<blocks, 512>>>(g, p, state1, unorm, beta1, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
			}

			kOptimizer32bit1State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, unorm, max_unorm, param_norm, beta1, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
	}
}

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
{

	int blocks = 0;
	switch(OPTIMIZER)
	{
		case ADAM:
			blocks = n/BLOCKSIZE_2STATE;
			blocks = n % BLOCKSIZE_2STATE == 0 ? blocks : blocks + 1;
			kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE><<<blocks, BLOCKSIZE_2STATE/NUM_2STATE>>>(p, g, state1, state2, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
		case RMSPROP:
    case ADAGRAD:
			blocks = n/BLOCKSIZE_1STATE;
			blocks = n % BLOCKSIZE_1STATE == 0 ? blocks : blocks + 1;
			kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE><<<blocks, BLOCKSIZE_1STATE/NUM_1STATE>>>(p, g, state1, beta1, beta2, eps, step, lr,
																														quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
	}
}



template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
  int blocks = n/2048;
  blocks = n % 2048 == 0 ? blocks : blocks + 1;
	CUDA_CHECK_RETURN(cudaMemset(&gnorm_vec[step % 100], 0, 1*sizeof(float)));
  kPercentileClipping<T, 2048, 4><<<blocks, 512>>>(g, gnorm_vec, step, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void func<float, FILL>(float *A, float value, long n);
template void func<unsigned char, FILL>(unsigned char *A, unsigned char value, long n);
template void func<float, ARANGE>(float *A, float value, long n);

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<half, 0>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void quantizeBlockwise<float, 0>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void quantizeBlockwise<half, 1>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void quantizeBlockwise<float, 1>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void dequantizeBlockwise<half>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<float>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);

template void quantizeBlockwiseDynamic<float, 2048>(float *A, float *absmax, unsigned char *out, const int n);
template void quantizeBlockwiseDynamic<float, 4096>(float *A, float *absmax, unsigned char *out, const int n);
template void dequantizeBlockwiseDynamic<float, 2048>(unsigned char *A, float *absmax, float *out, int n);
template void dequantizeBlockwiseDynamic<float, 4096>(unsigned char *A, float *absmax, float *out, int n);

template void quantizeBlockwiseDynamic<half, 2048>(half *A, float *absmax, unsigned char *out, const int n);
template void quantizeBlockwiseDynamic<half, 4096>(half *A, float *absmax, unsigned char *out, const int n);
template void dequantizeBlockwiseDynamic<half, 2048>(unsigned char *A, float *absmax, half *out, int n);
template void dequantizeBlockwiseDynamic<half, 4096>(unsigned char *A, float *absmax, half *out, int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(ADAGRAD, half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);

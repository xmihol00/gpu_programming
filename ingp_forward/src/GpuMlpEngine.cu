#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include "Utility.cuh"
#include <iomanip>

using namespace std;

// set to true to compile all kernels, otherwise only the best are compiled
#define COMPILE_ALL true

// workaround for half precision arithmetic on CC < 5.3
#if __CUDA_ARCH__ >= 530
	#define hadd __hadd
	#define hmul __hmul
	#define hfma __hfma
	#define hrelu __hfma_relu
#else 
	__device__ __forceinline__ half hadd(const half a, const half b)
	{
		return __float2half(__half2float(a) + __half2float(b));
	}

	__device__ __forceinline__ half hmul(const half a, const half b)
	{
		return __float2half(__half2float(a) * __half2float(b));
	}
	
	__device__ __forceinline__ half hfma(const half a, const half b, const half c)
	{
		return __float2half(__half2float(a) * __half2float(b) + __half2float(c));
	}	

	__device__ __forceinline__ half hrelu(const half a, const half b, const half c)
	{
		return __float2half(__half2float(a) * __half2float(b) + __half2float(c) > 0 ? __half2float(a) * __half2float(b) + __half2float(c) : 0);
	}
#endif


void allocateMemory(float *&d_data, uint32_t size)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_data, size * sizeof(float)));
}

void allocateAndSetWeights(vector<float> weights, float *&d_weights)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_weights, weights.size() * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void allocateAndSetHalfWeights(vector<half> weights, half *&d_weights)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_weights, weights.size() * sizeof(half)));
	HANDLE_ERROR(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(half), cudaMemcpyHostToDevice));
}

void freeMemoryOnDevice(initializer_list<void *> d_memoryList)
{
	for (void *d_memory : d_memoryList)
	{
		if (d_memory != nullptr)
		{
			HANDLE_ERROR(cudaFree(d_memory));
		}
	}
}

#if COMPILE_ALL
__global__ void matMulKernel(const float* inputs, float *weights, float* outputs, uint16_t rows, uint16_t elements)
{
	uint32_t columnIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    	
    float *weightsThread = weights + rowIndex * elements;
    const float *inputsThread = inputs + columnIndex * elements;
    
    float accumulator = 0;
    for (uint16_t i = 0; i < elements; i++)
    {
        accumulator += inputsThread[i] * weightsThread[i];
    }
	
    outputs[columnIndex * rows + rowIndex] = accumulator;
}

__global__ void reluKernel(float* inputs, float* outputs)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	outputs[index] = inputs[index] > 0 ? inputs[index] : 0;
}

__global__ void fusedGlobalMemoryKernel(const float* inputs, float* outputs, float* intermidiateOutputsL0, 
										float* intermidiateOutputsL1, float *weightsL0, float *weightsL1, 
										float *weightsL2)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 6;
	uint32_t rowIndex = threadId & 63;
		
	float *weightsThread = weightsL0 + rowIndex * 71;
	const float *inputsThread = inputs + columnIndex * 71;
	float accumulator = 0;
	#pragma unroll
	for (uint8_t i = 0; i < 71; i++)
	{
		accumulator += inputsThread[i] * weightsThread[i];
	}
	intermidiateOutputsL0[columnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
	__syncthreads();

	weightsThread = weightsL1 + rowIndex * 64;
	inputsThread = intermidiateOutputsL0 + columnIndex * 64;
	accumulator = 0;
	#pragma unroll
	for (uint8_t i = 0; i < 64; i++)
	{
		accumulator += inputsThread[i] * weightsThread[i];
	}
	intermidiateOutputsL1[columnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
	__syncthreads();

	if (rowIndex >= 4)
	{
		return;
	}

	weightsThread = weightsL2 + rowIndex * 64;
	inputsThread = intermidiateOutputsL1 + columnIndex * 64;
	accumulator = 0;
	#pragma unroll
	for (uint8_t i = 0; i < 64; i++)
	{
		accumulator += inputsThread[i] * weightsThread[i];
	}
	outputs[columnIndex * 4 + rowIndex] = accumulator;
}

template <uint8_t inputsPerBlock>
__global__ void fusedSharedMemKernel(const float* inputs, float* outputs, float *weightsL0, float *weightsL1, float *weightsL2)
{
	static_assert(inputsPerBlock == 4, "inputsPerBlock must be equal to 4");

	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 6;
	uint32_t rowIndex = threadId & 63;
	uint8_t sharedColumnIndex = columnIndex & (inputsPerBlock - 1);

	__shared__ float sharedInputs[71 * inputsPerBlock];
	__shared__ float intermidiateOutputsL0[64 * inputsPerBlock];
	__shared__ float intermidiateOutputsL1[64 * inputsPerBlock];
	__shared__ float sharedWeights[64 * 71];
	float *offsetedSharedWeights = sharedWeights;

	inputs += inputsPerBlock * 71 * (columnIndex >> 2);
	sharedInputs[threadIdx.x] = inputs[threadIdx.x];
	if (threadIdx.x < inputsPerBlock * 7)
	{
		sharedInputs[inputsPerBlock * 64 + threadIdx.x] = inputs[inputsPerBlock * 64 + threadIdx.x];
	}

	#pragma unroll
	for (uint8_t i = 0; i < inputsPerBlock; i++)
	{
		reinterpret_cast<float4 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float4 *>(weightsL0)[threadIdx.x];
		offsetedSharedWeights += 64 * 4 * inputsPerBlock;
		weightsL0 += 64 * 4 * inputsPerBlock;
	}
	if (threadIdx.x < 224)
	{
		reinterpret_cast<float2 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float2 *>(weightsL0)[threadIdx.x];
	}
	__syncthreads();

	float *weightsThread = sharedWeights + rowIndex * 71;
	const float *inputsThread = sharedInputs + sharedColumnIndex * 71;
	float accumulator = 0;
	#pragma unroll
	for (uint8_t i = 0; i < 71; i++)
	{
		accumulator += inputsThread[i] * weightsThread[i];
	}
	intermidiateOutputsL0[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
	__syncthreads();
	
	offsetedSharedWeights = sharedWeights;
	#pragma unroll
	for (uint8_t i = 0; i < inputsPerBlock; i++)
	{
		reinterpret_cast<float4 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float4 *>(weightsL1)[threadIdx.x];
		offsetedSharedWeights += 64 * 4 * inputsPerBlock;
		weightsL1 += 64 * 4 * inputsPerBlock;
	}
	__syncthreads();

	weightsThread = sharedWeights + rowIndex * 64;
	inputsThread = intermidiateOutputsL0 + sharedColumnIndex * 64;
	accumulator = 0;
	#pragma unroll
	for (uint8_t i = 0; i < 64; i++)
	{
		accumulator += inputsThread[i] * weightsThread[i];
	}
	intermidiateOutputsL1[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
	__syncthreads();

	sharedWeights[threadIdx.x] = weightsL2[threadIdx.x];
	__syncthreads();
	
	if (rowIndex >= 4)
	{
		return;
	}

	weightsThread = sharedWeights + rowIndex * 64;
	inputsThread = intermidiateOutputsL1 + sharedColumnIndex * 64;
	accumulator = 0;
	#pragma unroll
	for (uint8_t i = 0; i < 64; i++)
	{
		accumulator += inputsThread[i] * weightsThread[i];
	}
	outputs[columnIndex * 4 + rowIndex] = accumulator;
}

template <uint8_t inputsPerBlock, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void wholeNetInSharedMemKernel(const float* inputs, float* outputs, float *weightsL0, float *weightsL1, float *weightsL2)
{
	static_assert(inputsPerBlock == 4, "inputsPerBlock must be equal to 4");

	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 6;
	uint32_t rowIndex = threadId & 63;
	uint8_t sharedColumnIndex = columnIndex & (inputsPerBlock - 1);

	__shared__ float sharedInputs[71 * inputsPerBlock];
	__shared__ float intermidiateOutputs[64 * inputsPerBlock];
	__shared__ float sharedWeightsL0[64 * 71];
	__shared__ float sharedWeightsL1[64 * 64];
	__shared__ float sharedWeightsL2[4 * 64];
	float *offsetedSharedWeights = sharedWeightsL0;

	inputs += inputsPerBlock * 71 * (columnIndex >> 2);
	outputs += 4 * columnIndex;
	
	#pragma unroll
	for (uint8_t i = 0; i < inputsPerBlock; i++)
	{
		reinterpret_cast<float4 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float4 *>(weightsL0)[threadIdx.x];
		offsetedSharedWeights += 64 * 4 * inputsPerBlock;
		weightsL0 += 64 * 4 * inputsPerBlock;
	}
	if (threadIdx.x < 224)
	{
		reinterpret_cast<float2 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float2 *>(weightsL0)[threadIdx.x];
	}

	offsetedSharedWeights = sharedWeightsL1;
	#pragma unroll
	for (uint8_t i = 0; i < inputsPerBlock; i++)
	{
		reinterpret_cast<float4 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float4 *>(weightsL1)[threadIdx.x];
		offsetedSharedWeights += 64 * 4 * inputsPerBlock;
		weightsL1 += 64 * 4 * inputsPerBlock;
	}

	sharedWeightsL2[threadIdx.x] = weightsL2[threadIdx.x];
	
	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		sharedInputs[threadIdx.x] = inputs[threadIdx.x];
		if (threadIdx.x < inputsPerBlock * 7)
		{
			sharedInputs[inputsPerBlock * 64 + threadIdx.x] = inputs[inputsPerBlock * 64 + threadIdx.x];
		}
		__syncthreads();

		float *weightsThread = sharedWeightsL0 + rowIndex * 71;
		float *inputsThread = sharedInputs + sharedColumnIndex * 71;
		float accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 71; j++)
		{
			accumulator += inputsThread[j] * weightsThread[j];
		}
		intermidiateOutputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		__syncthreads();

		weightsThread = sharedWeightsL1 + rowIndex * 64;
		inputsThread = intermidiateOutputs + sharedColumnIndex * 64;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 64; j++)
		{
			accumulator += inputsThread[j] * weightsThread[j];
		}
		sharedInputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		__syncthreads();

		if (rowIndex < 4)
		{
			weightsThread = sharedWeightsL2 + rowIndex * 64;
			inputsThread = sharedInputs + sharedColumnIndex * 64;
			accumulator = 0;
			#pragma unroll
			for (uint8_t j = 0; j < 64; j++)
			{
				accumulator += inputsThread[j] * weightsThread[j];
			}
			outputs[rowIndex] = accumulator;
		}

		__syncthreads();
		inputs += 71 * 4 * numberOfBlocks;
		outputs += 4 * 4 * numberOfBlocks;
	}
}

template <uint8_t inputsPerBlock, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void netInSharedMemAndRegsKernel(const float* inputs, float* outputs, float *weightsL0, float *weightsL1, float *weightsL2)
{
	static_assert(inputsPerBlock == 4, "inputsPerBlock must be equal to 4");

	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 6;
	uint32_t rowIndex = threadId & 63;
	uint8_t sharedColumnIndex = columnIndex & (inputsPerBlock - 1);

	__shared__ float sharedInputs[71 * inputsPerBlock];
	__shared__ float intermidiateOutputs[64 * inputsPerBlock];
	__shared__ float sharedWeightsL0[64 * 71];
	__shared__ float sharedWeightsL2[4 * 64];
	float weightsL1Regs[64];
	float *offsetedSharedWeights = sharedWeightsL0;

	inputs += inputsPerBlock * 71 * (columnIndex >> 2);
	outputs += 4 * columnIndex;
	
	#pragma unroll
	for (uint8_t i = 0; i < inputsPerBlock; i++)
	{
		reinterpret_cast<float4 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float4 *>(weightsL0)[threadIdx.x];
		offsetedSharedWeights += 64 * 4 * inputsPerBlock;
		weightsL0 += 64 * 4 * inputsPerBlock;
	}
	if (threadIdx.x < 224)
	{
		reinterpret_cast<float2 *>(offsetedSharedWeights)[threadIdx.x] = reinterpret_cast<float2 *>(weightsL0)[threadIdx.x];
	}

	offsetedSharedWeights = weightsL1 + rowIndex * 64;
	#pragma unroll
	for (uint8_t i = 0; i < 64; i++)
	{
		weightsL1Regs[i] = offsetedSharedWeights[i];
	}

	sharedWeightsL2[threadIdx.x] = weightsL2[threadIdx.x];
	
	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		sharedInputs[threadIdx.x] = inputs[threadIdx.x];
		if (threadIdx.x < inputsPerBlock * 7)
		{
			sharedInputs[inputsPerBlock * 64 + threadIdx.x] = inputs[inputsPerBlock * 64 + threadIdx.x];
		}
		__syncthreads();

		float *weightsThread = sharedWeightsL0 + rowIndex * 71;
		float *inputsThread = sharedInputs + sharedColumnIndex * 71;
		float accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 71; j++)
		{
			accumulator += inputsThread[j] * weightsThread[j];
		}
		intermidiateOutputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		__syncthreads();

		inputsThread = intermidiateOutputs + sharedColumnIndex * 64;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 64; j++)
		{
			accumulator += inputsThread[j] * weightsL1Regs[j];
		}
		sharedInputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		__syncthreads();

		if (rowIndex < 4)
		{
			weightsThread = sharedWeightsL2 + rowIndex * 64;
			inputsThread = sharedInputs + sharedColumnIndex * 64;
			accumulator = 0;
			#pragma unroll
			for (uint8_t j = 0; j < 64; j++)
			{
				accumulator += inputsThread[j] * weightsThread[j];
			}
			outputs[rowIndex] = accumulator;
		}

		__syncthreads();
		inputs += 71 * inputsPerBlock * numberOfBlocks;
		outputs += 4 * inputsPerBlock * numberOfBlocks;
	}
}

template <uint8_t inputsPerBlock, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void lastLayerCoopKernel(const float* inputs, float* outputs, float *weightsL0, float *weightsL1, float *weightsL2)
{
	static_assert(inputsPerBlock == 4, "inputsPerBlock must be equal to 4");

	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 6;
	uint32_t rowIndex = threadId & 63;
	uint8_t sharedColumnIndex = columnIndex & (inputsPerBlock - 1);
	uint8_t L2WeightsOffset = (threadIdx.x & 31) * 4 + (threadIdx.x >> 7) * 128;
	uint8_t L2InputsOffset = (((threadIdx.x >> 5) - (threadIdx.x >> 7) * 4) << 6) + (threadIdx.x & 15) * 4;

	__shared__ float sharedInputs[71 * 4];
	__shared__ float intermidiateOutputs[64 * 4];
	__shared__ float sharedWeightsL0[64 * 71];
	float weightsL1Regs[64];
	float weightsL2Regs[4];
	float *offsetedWeights = sharedWeightsL0;

	inputs += inputsPerBlock * 71 * (columnIndex >> 2);
	uint8_t offset = (threadIdx.x >> 4) - 8 * (threadIdx.x >> 7);
	outputs += 4 * inputsPerBlock * (columnIndex >> 2) + offset + (offset & 6) + 2 * (threadIdx.x >> 7);

	#pragma unroll
	for (uint8_t i = 0; i < inputsPerBlock; i++)
	{
		reinterpret_cast<float4 *>(offsetedWeights)[threadIdx.x] = reinterpret_cast<float4 *>(weightsL0)[threadIdx.x];
		offsetedWeights += 64 * 4 * inputsPerBlock;
		weightsL0 += 64 * 4 * inputsPerBlock;
	}
	if (threadIdx.x < 224)
	{
		reinterpret_cast<float2 *>(offsetedWeights)[threadIdx.x] = reinterpret_cast<float2 *>(weightsL0)[threadIdx.x];
	}

	offsetedWeights = weightsL1 + rowIndex * 64;
	#pragma unroll
	for (uint8_t i = 0; i < 64; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL2 + L2WeightsOffset;
	#pragma unroll
	for (uint8_t i = 0; i < 4; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i];
	}
	
	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		sharedInputs[threadIdx.x] = inputs[threadIdx.x];
		if (threadIdx.x < inputsPerBlock * 7)
		{
			sharedInputs[inputsPerBlock * 64 + threadIdx.x] = inputs[inputsPerBlock * 64 + threadIdx.x];
		}
		__syncthreads();

		float *weightsThread = sharedWeightsL0 + rowIndex * 71;
		float *inputsThread = sharedInputs + sharedColumnIndex * 71;
		float accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 71; j++)
		{
			accumulator += inputsThread[j] * weightsThread[j];
		}
		intermidiateOutputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		__syncthreads();

		inputsThread = intermidiateOutputs + sharedColumnIndex * 64;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 64; j++)
		{
			accumulator += inputsThread[j] * weightsL1Regs[j];
		}
		sharedInputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		__syncthreads();

		inputsThread = sharedInputs + L2InputsOffset;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 4; j++)
		{
			accumulator += inputsThread[j] * weightsL2Regs[j];
		}
		#pragma unroll
		for (uint8_t j = 8; j > 0; j >>= 1)
		{
			accumulator += __shfl_xor_sync(0xffffffff, accumulator, j);
		}

		if ((threadIdx.x & 15) == 0)
		{
			outputs[0] = accumulator;
		}
		__syncthreads();

		inputs += 71 * 4 * numberOfBlocks;
		outputs += 4 * 4 * numberOfBlocks;
	}
}

template <uint8_t inputsPerBlock, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void wholeNetInRegsKernel(const float* inputs, float* outputs, float *weightsL0, float *weightsL1, float *weightsL2)
{
	static_assert(inputsPerBlock == 2, "inputsPerBlock must be equal to 2");
		
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t L2InputsOffset = (threadIdx.x >> 7) * 64 + (threadIdx.x & 31) * 2;

	__shared__ float sharedInputs[72 * inputsPerBlock];
	__shared__ float intermidiateOutputs[64 * inputsPerBlock];
	float weightsL0Regs[36];
	float weightsL1Regs[32];
	float weightsL2Regs[2];

	inputs += inputsPerBlock * 71 * (columnIndex >> 1);
	outputs += inputsPerBlock * 4 * (columnIndex >> 1) + warpId;

	float *offsetedWeights = weightsL0 + rowIndex * 72 + oddThread * 36;
	#pragma unroll
	for (uint8_t i = 0; i < 36; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL1 + rowIndex * 64 + oddThread * 32;
	#pragma unroll
	for (uint8_t i = 0; i < 32; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 127) * 2;
	#pragma unroll
	for (uint8_t i = 0; i < 2; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i];
	}

	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{	
		if (threadIdx.x < 71)
		{
			sharedInputs[threadIdx.x] = inputs[threadIdx.x];
		}
		else if (threadIdx.x >= 72 && threadIdx.x < 143)
		{
			sharedInputs[threadIdx.x] = inputs[threadIdx.x - 1];
		}
		__syncthreads();

		float *inputsThread = sharedInputs + sharedColumnIndex * 72 + oddThread * 36;
		float accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 36; j++)
		{
			accumulator += inputsThread[j] * weightsL0Regs[j];
		}
		accumulator += __shfl_xor_sync(0xffffffff, accumulator, 1);
		if (oddThread)
		{
			intermidiateOutputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;	
		}
		__syncthreads();

		inputsThread = intermidiateOutputs + sharedColumnIndex * 64 + oddThread * 32;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 32; j++)
		{
			accumulator += inputsThread[j] * weightsL1Regs[j];
		}
		accumulator += __shfl_xor_sync(0xffffffff, accumulator, 1);
		if (oddThread)
		{
			sharedInputs[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
		}
		__syncthreads();

		inputsThread = sharedInputs + L2InputsOffset;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 2; j++)
		{
			accumulator += inputsThread[j] * weightsL2Regs[j];
		}
		#pragma unroll
		for (uint8_t j = 16; j > 0; j >>= 1)
		{
			accumulator += __shfl_xor_sync(0xffffffff, accumulator, j);
		}

		if ((threadIdx.x & 31) == 0)
		{
			outputs[0] = accumulator;
		}

		__syncthreads();
		inputs += 71 * inputsPerBlock * numberOfBlocks;
		outputs += 4 * inputsPerBlock * numberOfBlocks;
	}
}

template <uint8_t inputsPerBlock, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void wholeNetInRegsHalfKernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	// 'hadd', 'hmul', 'hfma', 'hrelu' - workaround for CC < 5.3, see top of the file

	static_assert(inputsPerBlock == 2, "inputsPerBlock must be equal to 2");
		
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x; // global thread id across all blocks
	uint32_t columnIndex = threadId >> 7;  // column index in the input matrix for 4 warps (2 columns per block of 256 threads)
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1; // row index in the input matrix, 2 susbsequent threads process 1 row
	uint8_t sharedColumnIndex = threadIdx.x >> 7; // column index to the shared memory for 4 warps
	// thread offset in the shared memory for inputs to the last layer, warps 0-3 access the first column [0, 2, 4, ..., 62], 
	// warps 4-7 access the second column [64, 66, 68, ..., 126]
	uint8_t L2InputsOffset = (threadIdx.x >> 7) * 64 + (threadIdx.x & 31) * 2;

	__shared__ half sharedInputs[72 * inputsPerBlock];
	__shared__ half intermidiateOutputs[64 * inputsPerBlock];
	// even threads access the first 36 weights of a row, odd threads access the second 36 weights
	half weightsL0Regs[36];
	half weightsL1Regs[32];
	// whole warp computes 1 output, each thread holds 2 weights in a given row
	half weightsL2Regs[2];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += inputsPerBlock * 72 * (columnIndex >> 1); // all threads in the whole block have the same input offset
	outputs += inputsPerBlock * 4 * (columnIndex >> 1) + warpId; // all threads in a warp have the same output offset

	// population of the registers with weights
	half *offsetedWeights = weightsL0 + rowIndex * 72 + oddThread * 36;
	#pragma unroll
	for (uint8_t i = 0; i < 36; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL1 + rowIndex * 64 + oddThread * 32;
	#pragma unroll
	for (uint8_t i = 0; i < 32; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 127) * 2;
	#pragma unroll
	for (uint8_t i = 0; i < 2; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i];
	}

	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{	
		// fill in the shared memory with inputs
		if (threadIdx.x < 144)
		{
			sharedInputs[threadIdx.x] = inputs[threadIdx.x];
		}
		__syncthreads(); // make sure inputs are loaded

		// layer 0
		half *inputsThread = sharedInputs + sharedColumnIndex * 72 + oddThread * 36; // move to correct row and column
		half accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 36; j++) // TODO: bank conflicts?
		{
			accumulator = hfma(inputsThread[j], weightsL0Regs[j], accumulator);
		}
		// sum the partial results from neighbouring threads
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (oddThread)
		{
			intermidiateOutputs[sharedColumnIndex * 64 + rowIndex] = hrelu(accumulator, one, zero);
		}
		__syncthreads(); // make sure layer 0 is computed for all inputs

		// layer 1
		inputsThread = intermidiateOutputs + sharedColumnIndex * 64 + oddThread * 32;
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 32; j++)
		{
			accumulator = hfma(inputsThread[j], weightsL1Regs[j], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (oddThread)
		{
			sharedInputs[sharedColumnIndex * 64 + rowIndex] = hrelu(accumulator, one, zero);
		}
		__syncthreads(); // make sure layer 1 is computed for all inputs

		// layer 2
		inputsThread = sharedInputs + L2InputsOffset; // move to correct row and column
		accumulator = 0;
		#pragma unroll
		for (uint8_t j = 0; j < 2; j++)
		{
			accumulator = hfma(inputsThread[j], weightsL2Regs[j], accumulator);
		}
		// tree reduction sum
		#pragma unroll
		for (uint8_t j = 16; j > 0; j >>= 1)
		{
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, j));
		}

		if ((threadIdx.x & 31) == 0) // 1 st thread in a warp writes the result
		{
			outputs[0] = __half2float(accumulator);
		}
		__syncthreads(); // make sure 'sharedInputs' is not overwritten

		// move to the next block of inputs and outputs
		inputs += 72 * inputsPerBlock * numberOfBlocks;
		outputs += 4 * inputsPerBlock * numberOfBlocks;
	}
}

template <uint8_t doubleColumnsPerIteration, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void reducedBankConflictsKernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	static_assert(doubleColumnsPerIteration == 8 || doubleColumnsPerIteration == 16, "doubleColumnsPerIteration must be equal to 8 or 16");
		
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t L2InputsOffset = (threadIdx.x >> 7) * 64 + (threadIdx.x & 31) * 2;
	uint8_t halfWarpId = warpId >> 1;

	__shared__ half sharedInputs[144 * doubleColumnsPerIteration];
	__shared__ half intermidiateOutputsL0[128 * doubleColumnsPerIteration];
	__shared__ half intermidiateOutputsL1[128 * doubleColumnsPerIteration];
	half weightsL0Regs[36];
	half weightsL1Regs[32];
	half weightsL2Regs[2];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += doubleColumnsPerIteration * 144 * (columnIndex >> 1);
	outputs += doubleColumnsPerIteration * 8 * (columnIndex >> 1) + warpId;

	half *offsetedWeights = weightsL0 + rowIndex * 72 + oddThread * 36;
	#pragma unroll
	for (uint8_t i = 0; i < 36; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL1 + rowIndex * 64 + oddThread * 32;
	#pragma unroll
	for (uint8_t i = 0; i < 32; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 127) * 2;
	#pragma unroll
	for (uint8_t i = 0; i < 2; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i];
	}

	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		half *inputsThread = sharedInputs;
		reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
		inputsThread += 256 * 8;
		inputs += 256 * 8;
		inputsThread[threadIdx.x] = inputs[threadIdx.x];
		inputs += 256;
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < doubleColumnsPerIteration; j++)
		{
			uint8_t offset = (halfWarpId & (doubleColumnsPerIteration - 1));
			// each thread accesses different 36 elements at each iteration
			inputsThread = sharedInputs + offset * 72 + oddThread * 36 + sharedColumnIndex * 72 * doubleColumnsPerIteration; 
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 36; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * doubleColumnsPerIteration] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < doubleColumnsPerIteration; j++)
		{
			uint16_t offset = (halfWarpId & (doubleColumnsPerIteration - 1)) * 64 + sharedColumnIndex * 64 * doubleColumnsPerIteration;
			inputsThread = intermidiateOutputsL0 + offset + oddThread * 32;
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 32; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL1Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL1[offset + rowIndex] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		inputsThread = intermidiateOutputsL1 + L2InputsOffset;
		#pragma unroll
		for (uint8_t j = 0; j < doubleColumnsPerIteration; j++)
		{
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 2; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL2Regs[k], accumulator);
			}
			#pragma unroll
			for (uint8_t k = 16; k > 0; k >>= 1)
			{
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, k));
			}

			if ((threadIdx.x & 31) == 0)
			{
				outputs[0] = __half2float(accumulator);
			}
			inputsThread += 128;
			outputs += 8;
		}
		
		inputs += 144 * doubleColumnsPerIteration * (numberOfBlocks - 1);
		outputs += 8 * doubleColumnsPerIteration * (numberOfBlocks - 1);
	}
}

template <uint8_t doubleColumnsPerIteration, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void coalescedWeightsReadsKernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	static_assert(doubleColumnsPerIteration == 8 || doubleColumnsPerIteration == 16, "doubleColumnsPerIteration must be equal to 8 or 16");
		
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t L2InputsOffset = (threadIdx.x >> 7) * 64 + (threadIdx.x & 31) * 2;
	uint8_t halfWarpId = warpId >> 1;

	__shared__ half sharedInputs[144 * doubleColumnsPerIteration];
	__shared__ half intermidiateOutputsL0[128 * doubleColumnsPerIteration];
	__shared__ half intermidiateOutputsL1[128 * doubleColumnsPerIteration];
	half weightsL0Regs[36];
	half weightsL1Regs[32];
	half weightsL2Regs[2];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += doubleColumnsPerIteration * 144 * (columnIndex >> 1);
	outputs += doubleColumnsPerIteration * 8 * (columnIndex >> 1) + warpId;

	half *offsetedWeights = weightsL0 + rowIndex + oddThread * 36 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 36; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL1 + rowIndex + oddThread * 32 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 32; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 127);
	#pragma unroll
	for (uint16_t i = 0; i < 2; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i * 128];
	}

	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		half *inputsThread = sharedInputs;
		reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
		inputsThread += 256 * 8;
		inputs += 256 * 8;
		inputsThread[threadIdx.x] = inputs[threadIdx.x];
		inputs += 256;
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < doubleColumnsPerIteration; j++)
		{
			uint8_t offset = (halfWarpId & (doubleColumnsPerIteration - 1));
			// each thread accesses different 36 elements at each iteration
			inputsThread = sharedInputs + offset * 72 + oddThread * 36 + sharedColumnIndex * 72 * doubleColumnsPerIteration; 
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 36; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * doubleColumnsPerIteration] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < doubleColumnsPerIteration; j++)
		{
			uint16_t offset = (halfWarpId & (doubleColumnsPerIteration - 1)) * 64 + sharedColumnIndex * 64 * doubleColumnsPerIteration;
			inputsThread = intermidiateOutputsL0 + offset + oddThread * 32;
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 32; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL1Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL1[offset + rowIndex] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		inputsThread = intermidiateOutputsL1 + L2InputsOffset;
		#pragma unroll
		for (uint8_t j = 0; j < doubleColumnsPerIteration; j++)
		{
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 2; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL2Regs[k], accumulator);
			}
			#pragma unroll
			for (uint8_t k = 16; k > 0; k >>= 1)
			{
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, k));
			}

			if ((threadIdx.x & 31) == 0)
			{
				outputs[0] = __half2float(accumulator);
			}
			inputsThread += 128;
			outputs += 8;
		}
		
		inputs += 144 * doubleColumnsPerIteration * (numberOfBlocks - 1);
		outputs += 8 * doubleColumnsPerIteration * (numberOfBlocks - 1);
	}
}

template <uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void inputsLoadHidingKernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 7;
	uint8_t warpId = threadIdx.x >> 4;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint16_t L2InputsOffset = ((threadIdx.x >> 6) - 2) * 64 + (threadIdx.x & 15) * 4;
	uint8_t halfWarpId = warpId >> 1;

	__shared__ half sharedInputs[72 * 32];
	__shared__ half intermidiateOutputsL0[64 * 32];
	__shared__ half intermidiateOutputsL1[64 * 32];
	half weightsL0Regs[36];
	half weightsL1Regs[32];
	half weightsL2Regs[4];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += 32 * 72 * (columnIndex >> 1);
	outputs += 32 * 4 * (columnIndex >> 1) + warpId - 8;

	half *offsetedWeights = weightsL0 + rowIndex + oddThread * 36 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 36; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL1 + rowIndex + oddThread * 32 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 32; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 63);
	#pragma unroll
	for (uint16_t i = 0; i < 4; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i * 64];
	}

	half *inputsThread = sharedInputs;
	reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
	inputsThread += 256 * 8;
	inputs += 256 * 8;
	inputsThread[threadIdx.x] = inputs[threadIdx.x];
	inputs += 256;

	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < 16; j++)
		{
			uint8_t offset = (halfWarpId & (15));
			// each thread accesses different 36 elements at each iteration
			inputsThread = sharedInputs + offset * 72 + oddThread * 36 + sharedColumnIndex * 72 * 16; 
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 36; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * 16] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < 16; j++)
		{
			uint16_t offset = (halfWarpId & (15)) * 64 + sharedColumnIndex * 64 * 16;
			inputsThread = intermidiateOutputsL0 + offset + oddThread * 32;
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 32; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL1Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL1[offset + rowIndex] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		if (threadIdx.x >= 128)
		{
			inputsThread = intermidiateOutputsL1 + L2InputsOffset;
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				half accumulator = 0;
				#pragma unroll
				for (uint8_t k = 0; k < 4; k++)
				{
					accumulator = hfma(inputsThread[k], weightsL2Regs[k], accumulator);
				}
				#pragma unroll
				for (uint8_t k = 8; k > 0; k >>= 1)
				{
					accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, k));
				}

				if ((threadIdx.x & 15) == 0)
				{
					outputs[0] = __half2float(accumulator);
				}
				inputsThread += 128;
				outputs += 8;
			}

			outputs += 8 * 16 * (numberOfBlocks - 1);
		}
		else if (i < iterationsPerThread - 1)
		{
			inputsThread = sharedInputs;
			inputs += 72 * 32 * (numberOfBlocks - 1);
			
			reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
			inputsThread += 128 * 8;
			inputs += 128 * 8;
			reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
			inputsThread += 128 * 8;
			inputs += 128 * 8;
			reinterpret_cast<float*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float*>(inputs)[threadIdx.x];
			inputs += 128 * 2;
		}
	}
}

template <uint8_t columnsPerIteration, uint16_t iterationsPerThread, uint16_t numberOfBlocks>
__global__ void betterOccupancyKernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	static_assert(columnsPerIteration == 16, "columnsPerIteration must be equal to 8 or 16");
		
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 8;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 3;
	uint8_t rowIndex = threadIdx.x >> 2;
	uint16_t L2InputsOffset = (threadIdx.x >> 4) * 64 + oddThread * 16;
	uint8_t halfWarpId = warpId >> 1;

	__shared__ half sharedInputs[72 * columnsPerIteration];
	__shared__ half intermidiateOutputs[64 * columnsPerIteration];
	half weightsL0Regs[18];
	half weightsL1Regs[16];
	half weightsL2Regs[16];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += columnsPerIteration * 72 * columnIndex;
	outputs += columnsPerIteration * 4 * columnIndex + ((threadIdx.x & 15) >> 2) + (threadIdx.x >> 4) * 4;

	half *offsetedWeights = weightsL0 + rowIndex + oddThread * 18 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 18; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL1 + rowIndex + oddThread * 16 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 16; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 31);
	#pragma unroll
	for (uint16_t i = 0; i < 16; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i * 32];
	}

	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		half *inputsThread = sharedInputs;
		reinterpret_cast<float2*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float2*>(inputs)[threadIdx.x];
		inputsThread += 256 * 4;
		inputs += 256 * 4;
		if (threadIdx.x < 128)
		{
			inputsThread[threadIdx.x] = inputs[threadIdx.x];
		}
		inputs += 128;
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < columnsPerIteration; j++)
		{
			uint8_t offset = (halfWarpId & (columnsPerIteration - 1));
			// each thread accesses different 36 elements at each iteration
			inputsThread = sharedInputs + offset * 72 + oddThread * 18; 
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 18; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 2));
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (!oddThread)
			{
				intermidiateOutputs[offset * 64 + rowIndex] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < columnsPerIteration; j++)
		{
			uint16_t offset = (halfWarpId & (columnsPerIteration - 1)) * 64;
			inputsThread = intermidiateOutputs + offset + oddThread * 16;
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 16; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL1Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 2));
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (!oddThread)
			{
				sharedInputs[offset + rowIndex] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		inputsThread = sharedInputs + L2InputsOffset;
		half accumulator = 0;
		#pragma unroll
		for (uint8_t k = 0; k < 16; k++)
		{
			accumulator = hfma(inputsThread[k], weightsL2Regs[k], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 2));
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		
		if ((threadIdx.x & 3) == 0)
		{
			outputs[0] = __half2float(accumulator);
		}

		inputs += 72 * columnsPerIteration * (numberOfBlocks - 1);
		outputs += 4 * columnsPerIteration * numberOfBlocks;
		__syncthreads();
	}
}
#endif 

__global__ void best5x5Kernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	const uint8_t columnsPerThread = 16;
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t halfWarpId = warpId >> 1;

	__shared__ half sharedInputs[72 * 2 * columnsPerThread];
	__shared__ half intermidiateOutputs[64 * 2 * columnsPerThread];
	half weightsRegs[36];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += columnsPerThread * 2 * 72 * (columnIndex >> 1);
	outputs += columnsPerThread * 2 * 4 * (columnIndex >> 1) + (threadIdx.x >> 1);

	half *inputsThread = sharedInputs;
	reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
	inputsThread += 256 * 8;
	inputs += 256 * 8;
	inputsThread[threadIdx.x] = inputs[threadIdx.x];
	__syncthreads();

	half *offsetedWeights = weightsL0 + rowIndex + oddThread * 36 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 36; i++)
	{
		weightsRegs[i] = offsetedWeights[i * 64];
	}

	#pragma unroll
	for (uint8_t j = 0; j < columnsPerThread; j++)
	{
		uint8_t offset = (halfWarpId & (columnsPerThread - 1));
		// each thread accesses different 36 elements at each iteration
		inputsThread = sharedInputs + offset * 72 + oddThread * 36 + sharedColumnIndex * 72 * columnsPerThread; 
		half accumulator = 0;
		#pragma unroll
		for (uint8_t k = 0; k < 36; k++)
		{
			accumulator = hfma(inputsThread[k], weightsRegs[k], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (oddThread)
		{
			intermidiateOutputs[offset * 64 + rowIndex + sharedColumnIndex * 64 * columnsPerThread] = hrelu(accumulator, one, zero);
		}
		halfWarpId++;
	}
	__syncthreads();

	offsetedWeights = weightsL1 + rowIndex + oddThread * 32 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 32; i++)
	{
		weightsRegs[i] = offsetedWeights[i * 64];
	}

	#pragma unroll
	for (uint8_t j = 0; j < columnsPerThread; j++)
	{
		uint16_t offset = (halfWarpId & (columnsPerThread - 1)) * 64 + sharedColumnIndex * 64 * columnsPerThread;
		inputsThread = intermidiateOutputs + offset + oddThread * 32;
		half accumulator = 0;
		#pragma unroll
		for (uint8_t k = 0; k < 32; k++)
		{
			accumulator = hfma(inputsThread[k], weightsRegs[k], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (oddThread)
		{
			sharedInputs[offset + rowIndex] = hrelu(accumulator, one, zero);
		}
		halfWarpId++;
	}
	__syncthreads();

	offsetedWeights = weightsL2 + (threadIdx.x & 31);
	#pragma unroll
	for (uint16_t i = 0; i < 32; i++)
	{
		weightsRegs[i] = offsetedWeights[i * 32];
	}

	inputsThread = sharedInputs + (threadIdx.x >> 3) * 64 + (threadIdx.x & 1) * 32;
	half accumulator = 0;
	#pragma unroll
	for (uint8_t k = 0; k < 32; k++)
	{
		accumulator = hfma(inputsThread[k], weightsRegs[k], accumulator);
	}
	accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
	
	if (threadIdx.x & 1)
	{
		outputs[0] = __half2float(accumulator);
	}
}

__global__ void best100x100Kernel(const half* inputs, float* outputs, half *weightsL0, half *weightsL1, half *weightsL2)
{
	const uint16_t numberOfBlocks = 4096;
	const uint16_t columnsPerThread = 16;
	const uint16_t iterationsPerThread = 16;

	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t columnIndex = threadId >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint16_t L2InputsOffset = (threadIdx.x >> 5) * 64 + (threadIdx.x & 7) * 8;
	uint8_t halfWarpId = warpId >> 1;

	__shared__ half sharedInputs[72 * 2 * columnsPerThread];
	__shared__ half intermidiateOutputsL0[64 * 2 * columnsPerThread];
	__shared__ half intermidiateOutputsL1[64 * 2 * columnsPerThread];
	half weightsL0Regs[36];
	half weightsL1Regs[32];
	half weightsL2Regs[8];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	inputs += columnsPerThread * 2 * 72 * (columnIndex >> 1);
	outputs += columnsPerThread * 2 * 4 * (columnIndex >> 1) + (threadIdx.x >> 3);

	half *offsetedWeights = weightsL0 + rowIndex + oddThread * 36 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 36; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL1 + rowIndex + oddThread * 32 * 64;
	#pragma unroll
	for (uint16_t i = 0; i < 32; i++)
	{
		weightsL1Regs[i] = offsetedWeights[i * 64];
	}

	offsetedWeights = weightsL2 + (threadIdx.x & 31);
	#pragma unroll
	for (uint16_t i = 0; i < 8; i++)
	{
		weightsL2Regs[i] = offsetedWeights[i * 32];
	}

	#pragma unroll
	for (uint16_t i = 0; i < iterationsPerThread; i++)
	{
		half *inputsThread = sharedInputs;
		reinterpret_cast<float4*>(inputsThread)[threadIdx.x] = reinterpret_cast<const float4*>(inputs)[threadIdx.x];
		inputsThread += 256 * 8;
		inputs += 256 * 8;
		inputsThread[threadIdx.x] = inputs[threadIdx.x];
		inputs += 256;
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < columnsPerThread; j++)
		{
			uint8_t offset = (halfWarpId & (columnsPerThread - 1));
			// each thread accesses different 36 elements at each iteration
			inputsThread = sharedInputs + offset * 72 + oddThread * 36 + sharedColumnIndex * 72 * columnsPerThread; 
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 36; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * columnsPerThread] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		#pragma unroll
		for (uint8_t j = 0; j < columnsPerThread; j++)
		{
			uint16_t offset = (halfWarpId & (columnsPerThread - 1)) * 64 + sharedColumnIndex * 64 * columnsPerThread;
			inputsThread = intermidiateOutputsL0 + offset + oddThread * 32;
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 32; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL1Regs[k], accumulator);
			}
			accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
			if (oddThread)
			{
				intermidiateOutputsL1[offset + rowIndex] = hrelu(accumulator, one, zero);
			}
			halfWarpId++;
		}
		__syncthreads();

		inputsThread = intermidiateOutputsL1 + L2InputsOffset;
		#pragma unroll
		for (uint8_t j = 0; j < columnsPerThread >> 2; j++)
		{
			half accumulator = 0;
			#pragma unroll
			for (uint8_t k = 0; k < 8; k++)
			{
				accumulator = hfma(inputsThread[k], weightsL2Regs[k], accumulator);
			}
			#pragma unroll
			for (uint8_t k = 4; k > 0; k >>= 1)
			{
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, k));
			}

			if ((threadIdx.x & 7) == 0)
			{
				outputs[0] = __half2float(accumulator);
			}
			inputsThread += 8 * 64;
			outputs += 8 * 4;
		}
		
		inputs += 72 * 2 * columnsPerThread * (numberOfBlocks - 1);
		outputs += 4 * 2 * columnsPerThread * (numberOfBlocks - 1);
	}
}

#if COMPILE_ALL
void printMatrix(float *d_matrix, uint32_t rows, uint32_t columns)
{
	float *matrix = new float[rows * columns];
	HANDLE_ERROR(cudaMemcpy(matrix, d_matrix, rows * columns * sizeof(float), cudaMemcpyDeviceToHost));

	cerr << fixed << setprecision(7);
	for (uint32_t i = 0; i < columns; i++)
	{
		for (uint32_t j = 0; j < rows - 1; j++)
		{
			cerr << setw(10) << matrix[i * rows + j] << ", ";
		}
		cerr << setw(10) << matrix[i * rows + rows - 1] << endl;
	}
}

void printMatrix(half *d_matrix, uint32_t rows, uint32_t columns)
{
	half *matrix = new half[rows * columns];
	HANDLE_ERROR(cudaMemcpy(matrix, d_matrix, rows * columns * sizeof(half), cudaMemcpyDeviceToHost));

	cerr << fixed << setprecision(7);
	for (uint32_t i = 0; i < columns; i++)
	{
		for (uint32_t j = 0; j < rows - 1; j++)
		{
			cerr << setw(10) << __half2float(matrix[i * rows + j]) << ", ";
		}
		cerr << setw(10) << __half2float(matrix[i * rows + rows - 1]) << endl;
	}
}
#endif

#if COMPILE_ALL
void baselineLaunch(const float* d_inputs, float* d_outputs, float* d_intermidiateOutputsL0, float* d_intermidiateOutputsL1, 
					float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, int numberOfInputs)
{
	dim3 blocks(numberOfInputs >> 4, 4, 1);
    dim3 threads(16, 16, 1);
	matMulKernel<<<blocks, threads>>>(d_inputs, d_weightsL0, d_intermidiateOutputsL0, 64, 71);
	HANDLE_ERROR(cudaDeviceSynchronize());
	reluKernel<<<numberOfInputs >> 2, 256>>>(d_intermidiateOutputsL0, d_intermidiateOutputsL0);
	HANDLE_ERROR(cudaDeviceSynchronize());
	
	matMulKernel<<<blocks, threads>>>(d_intermidiateOutputsL0, d_weightsL1, d_intermidiateOutputsL1, 64, 64);
	HANDLE_ERROR(cudaDeviceSynchronize());
	reluKernel<<<numberOfInputs >> 2, 256>>>(d_intermidiateOutputsL1, d_intermidiateOutputsL1);
	HANDLE_ERROR(cudaDeviceSynchronize());

	threads.y = 1;
	matMulKernel<<<blocks, threads>>>(d_intermidiateOutputsL1, d_weightsL2, d_outputs, 4, 64);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void fusedGlobalMemoryLaunch(const float* d_inputs, float* d_outputs, float* d_intermidiateOutputsL0, float* d_intermidiateOutputsL1, 
							 float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, int numberOfInputs)
{
	fusedGlobalMemoryKernel<<<numberOfInputs >> 2, 256>>>(d_inputs, d_outputs, d_intermidiateOutputsL0, d_intermidiateOutputsL1, 
											              d_weightsL0, d_weightsL1, d_weightsL2);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void fusedSharedMemLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
							 int numberOfInputs)
{
	fusedSharedMemKernel<4><<<numberOfInputs >> 2, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void wholeNetInSharedMemLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
							      int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		wholeNetInSharedMemKernel<4, 8, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		wholeNetInSharedMemKernel<4, 128, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void netInSharedMemAndRegsLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
							     int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		netInSharedMemAndRegsKernel<4, 8, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		netInSharedMemAndRegsKernel<4, 128, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void lastLayerCoopLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
						 int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		lastLayerCoopKernel<4, 8, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		lastLayerCoopKernel<4, 128, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void wholeNetInRegsLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
						  int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		wholeNetInRegsKernel<2, 16, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		wholeNetInRegsKernel<2, 256, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void wholeNetInRegsHalfLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						      int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		wholeNetInRegsHalfKernel<2, 16, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		wholeNetInRegsHalfKernel<2, 256, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void reducedBankConflictsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						        int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		reducedBankConflictsKernel<16, 1, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		reducedBankConflictsKernel<16, 16, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void coalescedWeightsReadsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						        int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		coalescedWeightsReadsKernel<16, 1, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		coalescedWeightsReadsKernel<16, 16, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void inputsLoadHidingLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						          int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		inputsLoadHidingKernel<1, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		inputsLoadHidingKernel<16, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void betterOccupancyLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						          int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		betterOccupancyKernel<16, 2, 400><<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	else if (numberOfInputs == 2'097'152)
	{
		betterOccupancyKernel<16, 32, 4096><<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}
#endif

void bestKernelsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2For5x5, 
					   half *d_weightsL2For100x100, int numberOfInputs)
{
	if (numberOfInputs == 12'800)
	{
		best5x5Kernel<<<numberOfInputs >> 5, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2For5x5);
	}
	else if (numberOfInputs == 2'097'152)
	{
		best100x100Kernel<<<numberOfInputs >> 9, 256>>>(d_inputs, d_outputs, d_weightsL0, d_weightsL1, d_weightsL2For100x100);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

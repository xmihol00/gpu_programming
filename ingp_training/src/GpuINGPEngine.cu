#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include "Utility.cuh"
#include <iomanip>
#include "helper_math.h"
#include "GpuINGPEngine.h"

using namespace std;

// set to true to compile all kernels, otherwise only the best are compiled
#define COMPILE_ALL false

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


void allocateMemory(void **d_data, uint32_t size)
{
	HANDLE_ERROR(cudaMalloc(d_data, size));
}

void copyMemoryToDevice(void *d_data, void *h_data, uint32_t size)
{
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
}

void copyMemoryFromDevice(void *h_data, void *d_data, uint32_t size)
{
    HANDLE_ERROR(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
}

void allocateAndSetMemory(void **d_data, void *h_data, uint32_t size)
{
	allocateMemory(d_data, size);
	copyMemoryToDevice(*d_data, h_data, size);
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

void __global__ baselineRayGenerationKernel(float *rays, Camera *camera, uint16_t size, uint32_t offset, uint16_t width, uint16_t height)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t columnIndex = (threadId + offset) / height;
	uint16_t rowIndex = (threadId + offset) % width;
	rays += threadId * 3;

	if (threadId < size)
	{
		float3 base_direction = 
		{ 
			(rowIndex + 0.5f - camera->cx) / camera->fl_x, 
			(columnIndex + 0.5f - camera->cy) / camera->fl_y, 
			1.0f 
		};
		base_direction = normalize(base_direction);

		rays[0] = dot(base_direction, camera->poseMatrix[0]);
		rays[1] = dot(base_direction, camera->poseMatrix[1]);
		rays[2] = dot(base_direction, camera->poseMatrix[2]);
	}
}

void __global__ baselineSampleGenerationKernel(float *rays, Camera *camera, float *aabb, float *samples, float *sampleDistributions)
{
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	rays += blockIdx.x * 3;
	samples += threadId * 6;

	float inversedDirections[3] = { 1.0f / rays[0], 1.0f / rays[1], 1.0f / rays[2] };

	float nearPlane = (aabb[0] - camera->origin.x) * inversedDirections[0];
	float farPlane = (aabb[3] - camera->origin.x) * inversedDirections[0];
	if (nearPlane > farPlane)
	{
		float swap = nearPlane;
		nearPlane = farPlane;
		farPlane = swap;
	}

	float nearY = (aabb[1] - camera->origin.y) * inversedDirections[1];
	float farY = (aabb[4] - camera->origin.y) * inversedDirections[1];
	if (nearY > farY)
	{
		float swap = nearY;
		nearY = farY;
		farY = swap;
	}

	if (nearPlane > farY || nearY > farPlane) 
	{
		samples[0] = 1'000;
		samples[1] = 1'000;
		samples[2] = 1'000;
		samples[3] = 1'000;
		samples[4] = 1'000;
		samples[5] = 1'000;
		return;
	}

	if (nearY > nearPlane)
	{
		nearPlane = nearY;
	}
	if (farY < farPlane)
	{
		farPlane = farY;
	}

	float nearZ = (aabb[2] - camera->origin.z) * inversedDirections[2];
	float farZ = (aabb[5] - camera->origin.z) * inversedDirections[2];
	if (nearZ > farZ)
	{
		float swap = nearZ;
		nearZ = farZ;
		farZ = swap;
	}

	if (nearPlane > farZ || nearZ > farPlane)
	{
		samples[0] = 1'000;
		samples[1] = 1'000;
		samples[2] = 1'000;
		samples[3] = 1'000;
		samples[4] = 1'000;
		samples[5] = 1'000;
		return;
	}
	
	if (nearZ > nearPlane)
	{
		nearPlane = nearZ;
	}
	if (farZ < farPlane)
	{
		farPlane = farZ;
	}

	if (nearPlane < minNear)
	{
		nearPlane = minNear;
	}

	sampleDistributions[threadId] = (farPlane - nearPlane) / numberOfSteps;

	float zValue = nearPlane + (farPlane - nearPlane) * threadIdx.x * inversedNumberOfSteps;
	float x = camera->origin.x + rays[0] * zValue;
	float y = camera->origin.y + rays[1] * zValue;
	float z = camera->origin.z + rays[2] * zValue;
	x = fminf(fmaxf(x, aabb[0]), aabb[3]);
	y = fminf(fmaxf(y, aabb[1]), aabb[4]);
	z = fminf(fmaxf(z, aabb[2]), aabb[5]);

	samples[0] = x;
	samples[1] = y;
	samples[2] = z;
	samples[3] = rays[0];
	samples[4] = rays[1];
	samples[5] = rays[2];
}

void __global__ frequencyEncodingKernel(float *samples, float *networkInputs)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	samples += threadId * 6 + 3;
	networkInputs += threadId * 71;

	networkInputs[0] = samples[0];
	networkInputs[1] = samples[1];
	networkInputs[2] = samples[2];

	constexpr uint8_t frequencies[] = { 1, 2, 4, 8, 16, 32 };
	networkInputs += 3;
	for (uint8_t i = 0; i < 6; i++)
	{
		float variation = samples[0] * frequencies[i];
		networkInputs[0] = sinf(variation);
		networkInputs[3] = cosf(variation);

		variation = samples[1] * frequencies[i];
		networkInputs[1] = sinf(variation);
		networkInputs[4] = cosf(variation);

		variation = samples[2] * frequencies[i];
		networkInputs[2] = sinf(variation);
		networkInputs[5] = cosf(variation);
		
		networkInputs += 6;
	}
}

void __global__ positionEncoderKernel(float *samples, float *embeddings, uint32_t *offsets, float *networkInputs)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	samples += threadId * 6;
	networkInputs += threadId * 71 + 39;
	
	for (uint8_t i = 0; i < 32; i++)
	{
		networkInputs[i] = 0.0f;
	}
	
	float x = (samples[0] + 1.0f) * 0.5;
	float y = (samples[1] + 1.0f) * 0.5;
	float z = (samples[2] + 1.0f) * 0.5;

	for (uint8_t level = 0; level < 16; level++)
	{
		float* grid = embeddings + (offsets[level] << 1);
		uint32_t hashmapSize = offsets[level + 1] - offsets[level];
		float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
		uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

		float positions[3];
 		positions[0] = x * scale + 0.5f;
		positions[1] = y * scale + 0.5f;
		positions[2] = z * scale + 0.5f;
				
		uint32_t gridPositions[3];
		gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
		gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
		gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
		
		float localPositions[3];
		localPositions[0] = positions[0] - gridPositions[0];
		localPositions[1] = positions[1] - gridPositions[1];
		localPositions[2] = positions[2] - gridPositions[2];
		
		for (uint8_t i = 0; i < 8; i++)
		{
			float w = 1.0f;
			uint32_t localGridPositions[3];
			
			for (uint8_t j = 0; j < 3; j++)
			{
				if ((i & (1 << j)) == 0)
				{
					w *= 1 - localPositions[j];
					localGridPositions[j] = gridPositions[j];
				}
				else
				{
					w *= localPositions[j];
					localGridPositions[j] = gridPositions[j] + 1;
				}
			}

			uint32_t stride = 1;
			uint32_t index = 0;

			for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
			{
				index += localGridPositions[j] * stride;
				stride *= resolution + 1;
			}

			if (stride > hashmapSize)
			{
				index = localGridPositions[0];
				index ^= localGridPositions[1] * 2654435761;
				index ^= localGridPositions[2] * 805459861;
			}

			index = (index % hashmapSize) * 2;
			
			networkInputs[level * 2] += w * grid[index];
			networkInputs[level * 2 + 1] += w * grid[index + 1];
		}
	}
}

template <uint8_t inputsPerBlock>
__global__ void baselineNetworkKernel(const float* inputs, float* outputs, float *weightsL0, float *weightsL1, float *weightsL2)
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

void __global__ baselineAccumulationKernel(const float *network_outputs, uint8_t *RGBA_outputs, float *sampleDistributions,
										   uint16_t size)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	RGBA_outputs += 4 * threadId;
	sampleDistributions += 512 * threadId;
	network_outputs += threadId * 4 * 512;

	float accumulators[3] = { 0.0f, 0.0f, 0.0f };
	float rem = 1.0f;

	if (threadId < size)
	{
		for (uint16_t i = 0; i < 512; i++)
		{
			float4 input = reinterpret_cast<const float4 *>(network_outputs)[i];
			float rgb[3] = { 
				1.0f / (1.0f + expf(-input.x)), 
				1.0f / (1.0f + expf(-input.y)), 
				1.0f / (1.0f + expf(-input.z)) 
			};

			float sigma = expf(input.w);
			float alpha = 1.0f - expf(-sampleDistributions[i] * sigma);

			accumulators[0] += rem * rgb[0] * alpha;
			accumulators[1] += rem * rgb[1] * alpha;
			accumulators[2] += rem * rgb[2] * alpha;

			rem *= 1.0f - alpha;
		}

		accumulators[0] += rem;
		accumulators[1] += rem;
		accumulators[2] += rem;

		RGBA_outputs[0] = static_cast<uint8_t>(min(max(0.0f, accumulators[0] * 255.0f), 255.0f));
		RGBA_outputs[1] = static_cast<uint8_t>(min(max(0.0f, accumulators[1] * 255.0f), 255.0f));
		RGBA_outputs[2] = static_cast<uint8_t>(min(max(0.0f, accumulators[2] * 255.0f), 255.0f));
		RGBA_outputs[3] = 255;
	}
}

void __global__ allInOneKernel(Camera *camera, uint16_t size, uint16_t width, uint16_t height, float *aabb, 
							   float *sampleDistributions, float *embeddings, uint32_t *offsets, float *networkInputs, 
							   float *weightsL0, float *weightsL1, float *weightsL2, uint8_t *RGBAOutputs, uint16_t iterations)
{
	RGBAOutputs += 4 * blockIdx.x;
	for (uint16_t iteration = 0; iteration < iterations; iteration++)
	{
		uint32_t offset = size * iteration;
		uint16_t columnIndex = (blockIdx.x + offset) / height;
		uint16_t rowIndex = (blockIdx.x + offset) % width;
		float rays[3];
		float zValue;
		float x;
		float y;
		float z;

		float3 baseDirection = 
		{ 
			(rowIndex + 0.5f - camera->cx) / camera->fl_x, 
			(columnIndex + 0.5f - camera->cy) / camera->fl_y, 
			1.0f 
		};
		baseDirection = normalize(baseDirection);

		rays[0] = dot(baseDirection, camera->poseMatrix[0]);
		rays[1] = dot(baseDirection, camera->poseMatrix[1]);
		rays[2] = dot(baseDirection, camera->poseMatrix[2]);

		constexpr float minNear = 0.2f;
		constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
		constexpr float numberOfSteps = 512.0f;
		uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

		float inversedDirections[3] = { 1.0f / rays[0], 1.0f / rays[1], 1.0f / rays[2] };

		float nearPlane = (aabb[0] - camera->origin.x) * inversedDirections[0];
		float farPlane = (aabb[3] - camera->origin.x) * inversedDirections[0];
		if (nearPlane > farPlane)
		{
			float swap = nearPlane;
			nearPlane = farPlane;
			farPlane = swap;
		}

		float nearY = (aabb[1] - camera->origin.y) * inversedDirections[1];
		float farY = (aabb[4] - camera->origin.y) * inversedDirections[1];
		float nearZ;
		float farZ;
		if (nearY > farY)
		{
			float swap = nearY;
			nearY = farY;
			farY = swap;
		}

		if (nearPlane > farY || nearY > farPlane) 
		{
			x = 1'000;
			y = 1'000;
			z = 1'000;
			rays[0] = 1'000;
			rays[1] = 1'000;
			rays[2] = 1'000;
			goto skip_sample_generation;
		}

		if (nearY > nearPlane)
		{
			nearPlane = nearY;
		}
		if (farY < farPlane)
		{
			farPlane = farY;
		}

		nearZ = (aabb[2] - camera->origin.z) * inversedDirections[2];
		farZ = (aabb[5] - camera->origin.z) * inversedDirections[2];
		if (nearZ > farZ)
		{
			float swap = nearZ;
			nearZ = farZ;
			farZ = swap;
		}

		if (nearPlane > farZ || nearZ > farPlane)
		{
			x = 1'000;
			y = 1'000;
			z = 1'000;
			rays[0] = 1'000;
			rays[1] = 1'000;
			rays[2] = 1'000;
			goto skip_sample_generation;
		}
		
		if (nearZ > nearPlane)
		{
			nearPlane = nearZ;
		}
		if (farZ < farPlane)
		{
			farPlane = farZ;
		}

		if (nearPlane < minNear)
		{
			nearPlane = minNear;
		}

		sampleDistributions[threadId] = (farPlane - nearPlane) / numberOfSteps;

		zValue = nearPlane + (farPlane - nearPlane) * threadIdx.x * inversedNumberOfSteps;
		x = camera->origin.x + rays[0] * zValue;
		y = camera->origin.y + rays[1] * zValue;
		z = camera->origin.z + rays[2] * zValue;
		x = fminf(fmaxf(x, aabb[0]), aabb[3]);
		y = fminf(fmaxf(y, aabb[1]), aabb[4]);
		z = fminf(fmaxf(z, aabb[2]), aabb[5]);

	skip_sample_generation:
		float *offsettedNetworkInputs = networkInputs + threadId * 71;

		offsettedNetworkInputs[0] = rays[0];
		offsettedNetworkInputs[1] = rays[1];
		offsettedNetworkInputs[2] = rays[2];

		constexpr uint8_t frequencies[] = { 1, 2, 4, 8, 16, 32 };
		offsettedNetworkInputs += 3;
		for (uint8_t i = 0; i < 6; i++)
		{
			float variation = rays[0] * frequencies[i];
			offsettedNetworkInputs[0] = sinf(variation);
			offsettedNetworkInputs[3] = cosf(variation);

			variation = rays[1] * frequencies[i];
			offsettedNetworkInputs[1] = sinf(variation);
			offsettedNetworkInputs[4] = cosf(variation);

			variation = rays[2] * frequencies[i];
			offsettedNetworkInputs[2] = sinf(variation);
			offsettedNetworkInputs[5] = cosf(variation);
			
			offsettedNetworkInputs += 6;
		}

		for (uint8_t i = 0; i < 32; i++)
		{
			offsettedNetworkInputs[i] = 0.0f;
		}
		
		x = (x + 1.0f) * 0.5;
		y = (y + 1.0f) * 0.5;
		z = (z + 1.0f) * 0.5;

		for (uint8_t level = 0; level < 16; level++)
		{
			float* grid = embeddings + (offsets[level] << 1);
			uint32_t hashmapSize = offsets[level + 1] - offsets[level];
			float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
			uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

			float positions[3];
			positions[0] = x * scale + 0.5f;
			positions[1] = y * scale + 0.5f;
			positions[2] = z * scale + 0.5f;
					
			uint32_t gridPositions[3];
			gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
			gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
			gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
			
			float localPositions[3];
			localPositions[0] = positions[0] - gridPositions[0];
			localPositions[1] = positions[1] - gridPositions[1];
			localPositions[2] = positions[2] - gridPositions[2];
			
			for (uint8_t i = 0; i < 8; i++)
			{
				float w = 1.0f;
				uint32_t localGridPositions[3];
				
				for (uint8_t j = 0; j < 3; j++)
				{
					if ((i & (1 << j)) == 0)
					{
						w *= 1 - localPositions[j];
						localGridPositions[j] = gridPositions[j];
					}
					else
					{
						w *= localPositions[j];
						localGridPositions[j] = gridPositions[j] + 1;
					}
				}

				uint32_t stride = 1;
				uint32_t index = 0;

				for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
				{
					index += localGridPositions[j] * stride;
					stride *= resolution + 1;
				}

				if (stride > hashmapSize)
				{
					index = localGridPositions[0];
					index ^= localGridPositions[1] * 2654435761;
					index ^= localGridPositions[2] * 805459861;
				}

				index = (index % hashmapSize) * 2;
				
				offsettedNetworkInputs[level * 2] += w * grid[index];
				offsettedNetworkInputs[level * 2 + 1] += w * grid[index + 1];
			}
		}
		__syncthreads();

		uint8_t warpId = threadIdx.x >> 5;
		uint8_t oddThread = threadIdx.x & 1;
		rowIndex = (threadId & 127) >> 1;
		uint8_t sharedColumnIndex = threadIdx.x >> 7;
		uint8_t L2InputsOffset = (threadIdx.x >> 7) * 64 + (threadIdx.x & 31) * 2;

		__shared__ float sharedInputs[72 * 4];
		__shared__ float intermidiateOutputs[64 * 4];
		__shared__ float sharedNetworkOutputs[4 * 512];
		float weightsL0Regs[36];
		float weightsL1Regs[32];
		float weightsL2Regs[2];

		offsettedNetworkInputs = networkInputs + 71 * 512 * blockIdx.x;
		float *networkOutputs = sharedNetworkOutputs + warpId;

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

		for (uint16_t i = 0; i < 128; i++)
		{	
			if (threadIdx.x < 71)
			{
				sharedInputs[threadIdx.x] = offsettedNetworkInputs[threadIdx.x];
			}
			else if (threadIdx.x >= 72 && threadIdx.x < 143)
			{
				sharedInputs[threadIdx.x] = offsettedNetworkInputs[threadIdx.x - 1];
			}
			else if (threadIdx.x >= 144 && threadIdx.x < 215)
			{
				sharedInputs[threadIdx.x] = offsettedNetworkInputs[threadIdx.x - 2];
			}
			else if (threadIdx.x >= 216 && threadIdx.x < 287)
			{
				sharedInputs[threadIdx.x] = offsettedNetworkInputs[threadIdx.x - 3];
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
				networkOutputs[0] = accumulator;
			}

			__syncthreads();
			offsettedNetworkInputs += 71 * 4;
			networkOutputs += 4 * 4;
		}

		if (threadIdx.x == 0)
		{
			networkOutputs = sharedNetworkOutputs;
			float *offsettedSampleDistributions = sampleDistributions + 512 * blockIdx.x;

			float accumulators[3] = { 0.0f, 0.0f, 0.0f };
			float rem = 1.0f;

			for (uint16_t i = 0; i < 512; i++)
			{
				float4 input = reinterpret_cast<const float4 *>(networkOutputs)[i];
				float rgb[3] = { 
					1.0f / (1.0f + expf(-input.x)), 
					1.0f / (1.0f + expf(-input.y)), 
					1.0f / (1.0f + expf(-input.z)) 
				};

				float sigma = expf(input.w);
				float alpha = 1.0f - expf(-offsettedSampleDistributions[i] * sigma);

				accumulators[0] += rem * rgb[0] * alpha;
				accumulators[1] += rem * rgb[1] * alpha;
				accumulators[2] += rem * rgb[2] * alpha;

				rem *= 1.0f - alpha;
			}

			accumulators[0] += rem;
			accumulators[1] += rem;
			accumulators[2] += rem;

			RGBAOutputs[0] = static_cast<uint8_t>(min(max(0.0f, accumulators[0] * 255.0f), 255.0f));
			RGBAOutputs[1] = static_cast<uint8_t>(min(max(0.0f, accumulators[1] * 255.0f), 255.0f));
			RGBAOutputs[2] = static_cast<uint8_t>(min(max(0.0f, accumulators[2] * 255.0f), 255.0f));
			RGBAOutputs[3] = 255;
			RGBAOutputs += 4 * size;
		}
		__syncthreads();
	}
}

void __global__ frequencyEncodingCoopKernel(Camera *camera, uint16_t size, uint16_t width, uint16_t height, float *aabb, 
							       float *embeddings, uint32_t *offsets, float *weightsFrequencyL0, float *weightsPositionL0, 
								   float *weightsL1, float *weightsL2, uint8_t *RGBAOutputs, uint16_t kernelIterations)
{
	constexpr uint16_t threadsPerBlock = 256;
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t L2InputsOffset = sharedColumnIndex * 64 + (threadIdx.x & 31) * 2;

	__shared__ float sharedFrequencyInputs[40];
	__shared__ float sharedBiases[64];
	__shared__ float sharedPositionInputs[256 * 32];
	__shared__ float intermidiateOutputsL0[64 * 2];
	__shared__ float intermidiateOutputsL1[64 * 2];
	__shared__ float sharedNetworkOutputs[4 * 256];
	float weightsL0Regs[16];
	float weightsL1Regs[32];
	float weightsL2Regs[2];

	RGBAOutputs += 4 * blockIdx.x;

	float *offsetedWeights = weightsPositionL0 + rowIndex * 32 + oddThread * 16;
	#pragma unroll
	for (uint8_t i = 0; i < 16; i++)
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

	for (uint16_t kernelIteration = 0; kernelIteration < kernelIterations; kernelIteration++)
	{
		uint32_t offset = size * kernelIteration;
		uint16_t imageColumnIndex = (blockIdx.x + offset) / height;
		uint16_t imageRowIndex = (blockIdx.x + offset) % width;
		bool skipSampleGeneration = false;
		float accumulators[3] = { 0.0f, 0.0f, 0.0f };
		float rem = 1.0f;
		float rays[3];
		float zValue;
		float x;
		float y;
		float z;
		float sampleDistributions = 0;
		
		float3 baseDirection = 
		{ 
			(imageRowIndex + 0.5f - camera->cx) / camera->fl_x, 
			(imageColumnIndex + 0.5f - camera->cy) / camera->fl_y, 
			1.0f 
		};
		baseDirection = normalize(baseDirection);

		rays[0] = dot(baseDirection, camera->poseMatrix[0]);
		rays[1] = dot(baseDirection, camera->poseMatrix[1]);
		rays[2] = dot(baseDirection, camera->poseMatrix[2]);

		if (threadIdx.x < 3)
		{
			sharedFrequencyInputs[threadIdx.x] = rays[threadIdx.x];
		}
		else if (threadIdx.x < 39)
		{
			uint8_t frequency = 1 << ((threadIdx.x - 3) / 6);
			float ray = rays[threadIdx.x % 3];
			float variation = ray * frequency;
			if (threadIdx.x % 6 >= 3)
			{
				sharedFrequencyInputs[threadIdx.x] = sinf(variation);
			}
			else
			{
				sharedFrequencyInputs[threadIdx.x] = cosf(variation);
			}
		}
		__syncthreads();

		if (threadIdx.x < 64)
		{
			offsetedWeights = weightsFrequencyL0 + threadIdx.x * 40;
			float accumulator = 0;
			for (uint8_t i = 0; i < 40; i++)
			{
				accumulator += offsetedWeights[i] * sharedFrequencyInputs[i];
			}
			sharedBiases[threadIdx.x] = accumulator;
		}
		__syncthreads();

		float inversedDirections[3] = { 1.0f / rays[0], 1.0f / rays[1], 1.0f / rays[2] };

		float nearPlane = (aabb[0] - camera->origin.x) * inversedDirections[0];
		float farPlane = (aabb[3] - camera->origin.x) * inversedDirections[0];
		if (nearPlane > farPlane)
		{
			float swap = nearPlane;
			nearPlane = farPlane;
			farPlane = swap;
		}

		float nearY = (aabb[1] - camera->origin.y) * inversedDirections[1];
		float farY = (aabb[4] - camera->origin.y) * inversedDirections[1];
		float nearZ;
		float farZ;
		if (nearY > farY)
		{
			float swap = nearY;
			nearY = farY;
			farY = swap;
		}

		if (nearPlane > farY || nearY > farPlane) 
		{
			x = 1'000;
			y = 1'000;
			z = 1'000;
			rays[0] = 1'000;
			rays[1] = 1'000;
			rays[2] = 1'000;
			skipSampleGeneration = true;
			goto skip_sample_generation;
		}

		if (nearY > nearPlane)
		{
			nearPlane = nearY;
		}
		if (farY < farPlane)
		{
			farPlane = farY;
		}

		nearZ = (aabb[2] - camera->origin.z) * inversedDirections[2];
		farZ = (aabb[5] - camera->origin.z) * inversedDirections[2];
		if (nearZ > farZ)
		{
			float swap = nearZ;
			nearZ = farZ;
			farZ = swap;
		}

		if (nearPlane > farZ || nearZ > farPlane)
		{
			x = 1'000;
			y = 1'000;
			z = 1'000;
			rays[0] = 1'000;
			rays[1] = 1'000;
			rays[2] = 1'000;
			goto skip_sample_generation;
		}
		
		if (nearZ > nearPlane)
		{
			nearPlane = nearZ;
		}
		if (farZ < farPlane)
		{
			farPlane = farZ;
		}

		if (nearPlane < minNear)
		{
			nearPlane = minNear;
		}

		sampleDistributions = (farPlane - nearPlane) / numberOfSteps;

		skip_sample_generation:
		for (uint16_t blockIteration = 0; blockIteration < 2; blockIteration++)
		{
			if (!skipSampleGeneration)
			{
				zValue = nearPlane + (farPlane - nearPlane) * (threadIdx.x + blockIteration * threadsPerBlock) * inversedNumberOfSteps;
				x = camera->origin.x + rays[0] * zValue;
				y = camera->origin.y + rays[1] * zValue;
				z = camera->origin.z + rays[2] * zValue;
				x = fminf(fmaxf(x, aabb[0]), aabb[3]);
				y = fminf(fmaxf(y, aabb[1]), aabb[4]);
				z = fminf(fmaxf(z, aabb[2]), aabb[5]);
			}
			float *offsettedPositionInputs = sharedPositionInputs + threadIdx.x * 32;

			for (uint8_t i = 0; i < 32; i++)
			{
				offsettedPositionInputs[i] = 0.0f;
			}
			
			x = (x + 1.0f) * 0.5;
			y = (y + 1.0f) * 0.5;
			z = (z + 1.0f) * 0.5;

			for (uint8_t level = 0; level < 16; level++)
			{
				float* grid = embeddings + (offsets[level] << 1);
				uint32_t hashmapSize = offsets[level + 1] - offsets[level];
				float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
				uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

				float positions[3];
				positions[0] = x * scale + 0.5f;
				positions[1] = y * scale + 0.5f;
				positions[2] = z * scale + 0.5f;
						
				uint32_t gridPositions[3];
				gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
				gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
				gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
				
				float localPositions[3];
				localPositions[0] = positions[0] - gridPositions[0];
				localPositions[1] = positions[1] - gridPositions[1];
				localPositions[2] = positions[2] - gridPositions[2];
				
				for (uint8_t i = 0; i < 8; i++)
				{
					float w = 1.0f;
					uint32_t localGridPositions[3];
					
					for (uint8_t j = 0; j < 3; j++)
					{
						if ((i & (1 << j)) == 0)
						{
							w *= 1 - localPositions[j];
							localGridPositions[j] = gridPositions[j];
						}
						else
						{
							w *= localPositions[j];
							localGridPositions[j] = gridPositions[j] + 1;
						}
					}

					uint32_t stride = 1;
					uint32_t index = 0;

					for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
					{
						index += localGridPositions[j] * stride;
						stride *= resolution + 1;
					}

					if (stride > hashmapSize)
					{
						index = localGridPositions[0];
						index ^= localGridPositions[1] * 2654435761;
						index ^= localGridPositions[2] * 805459861;
					}

					index = (index % hashmapSize) * 2;
					
					offsettedPositionInputs[level * 2] += w * grid[index];
					offsettedPositionInputs[level * 2 + 1] += w * grid[index + 1];
				}
			}

			offsettedPositionInputs = sharedPositionInputs + sharedColumnIndex * 32 + oddThread * 16;
			float *networkOutputs = sharedNetworkOutputs + warpId;
			for (uint16_t i = 0; i < 128; i++)
			{	
				__syncthreads();
							
				float accumulator = 0;
				#pragma unroll
				for (uint8_t j = 0; j < 16; j++)
				{
					accumulator += offsettedPositionInputs[j] * weightsL0Regs[j];
				}
				accumulator += __shfl_xor_sync(0xffffffff, accumulator, 1);
				if (oddThread)
				{
					accumulator += sharedBiases[rowIndex];
					intermidiateOutputsL0[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;	
				}
				__syncthreads();

				float *inputsThread = intermidiateOutputsL0 + sharedColumnIndex * 64 + oddThread * 32;
				accumulator = 0;
				#pragma unroll
				for (uint8_t j = 0; j < 32; j++)
				{
					accumulator += inputsThread[j] * weightsL1Regs[j];
				}
				accumulator += __shfl_xor_sync(0xffffffff, accumulator, 1);
				if (oddThread)
				{
					intermidiateOutputsL1[sharedColumnIndex * 64 + rowIndex] = accumulator > 0 ? accumulator : 0;
				}
				__syncthreads();

				inputsThread = intermidiateOutputsL1 + L2InputsOffset;
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
					networkOutputs[0] = accumulator;
				}

				offsettedPositionInputs += 32 * 2;
				networkOutputs += 4 * 2;
			}

			if (threadIdx.x == 0)
			{
				networkOutputs = sharedNetworkOutputs;

				for (uint16_t i = 0; i < 256; i++)
				{
					float4 input = reinterpret_cast<const float4 *>(networkOutputs)[i];
					float rgb[3] = { 
						1.0f / (1.0f + expf(-input.x)), 
						1.0f / (1.0f + expf(-input.y)), 
						1.0f / (1.0f + expf(-input.z)) 
					};

					float sigma = expf(input.w);
					float alpha = 1.0f - expf(-sampleDistributions * sigma);

					accumulators[0] += rem * rgb[0] * alpha;
					accumulators[1] += rem * rgb[1] * alpha;
					accumulators[2] += rem * rgb[2] * alpha;

					rem *= 1.0f - alpha;
				}
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			accumulators[0] += rem;
			accumulators[1] += rem;
			accumulators[2] += rem;

			RGBAOutputs[0] = static_cast<uint8_t>(min(max(0.0f, accumulators[0] * 255.0f), 255.0f));
			RGBAOutputs[1] = static_cast<uint8_t>(min(max(0.0f, accumulators[1] * 255.0f), 255.0f));
			RGBAOutputs[2] = static_cast<uint8_t>(min(max(0.0f, accumulators[2] * 255.0f), 255.0f));
			RGBAOutputs[3] = 255;
			RGBAOutputs += 4 * size;
		}
	}
}

void __global__ halfPrecisionKernel(Camera *camera, uint16_t size, uint16_t width, uint16_t height, float *aabb, 
							        float *embeddings, uint32_t *offsets, float *weightsFrequencyL0, half *weightsPositionL0, 
								    half *weightsL1, half *weightsL2, uint8_t *RGBAOutputs, uint16_t kernelIterations)
{
	constexpr uint16_t threadsPerBlock = 256;
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t L2InputsOffset = sharedColumnIndex * 64 + (threadIdx.x & 31) * 2;
	
	__shared__ float sharedFrequencyInputs[40];
	__shared__ half sharedBiases[64];
	__shared__ half sharedPositionInputs[32 * 256];
	__shared__ half intermidiateOutputsL0[64 * 32];
	__shared__ half intermidiateOutputsL1[64 * 32];
	__shared__ float sharedNetworkOutputs[4 * 256];
	half weightsL0Regs[16];
	half weightsL1Regs[32];
	half weightsL2Regs[2];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	RGBAOutputs += 4 * blockIdx.x;

	half *offsetedWeights = weightsPositionL0 + rowIndex * 32 + oddThread * 16;
	#pragma unroll
	for (uint8_t i = 0; i < 16; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i];
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

	for (uint16_t kernelIteration = 0; kernelIteration < kernelIterations; kernelIteration++)
	{
		uint32_t offset = size * kernelIteration;
		uint16_t imageColumnIndex = (blockIdx.x + offset) / height;
		uint16_t imageRowIndex = (blockIdx.x + offset) % width;
		bool skipSampleGeneration = false;
		float accumulators[3] = { 0.0f, 0.0f, 0.0f };
		float rem = 1.0f;
		float rays[3];
		float zValue;
		float x;
		float y;
		float z;
		float sampleDistributions = 0;
		
		float3 baseDirection = 
		{ 
			(imageRowIndex + 0.5f - camera->cx) / camera->fl_x, 
			(imageColumnIndex + 0.5f - camera->cy) / camera->fl_y, 
			1.0f 
		};
		baseDirection = normalize(baseDirection);

		rays[0] = dot(baseDirection, camera->poseMatrix[0]);
		rays[1] = dot(baseDirection, camera->poseMatrix[1]);
		rays[2] = dot(baseDirection, camera->poseMatrix[2]);

		if (threadIdx.x < 3)
		{
			sharedFrequencyInputs[threadIdx.x] = rays[threadIdx.x];
		}
		else if (threadIdx.x < 39)
		{
			uint8_t frequency = 1 << ((threadIdx.x - 3) / 6);
			float ray = rays[threadIdx.x % 3];
			float variation = ray * frequency;
			if (threadIdx.x % 6 >= 3)
			{
				sharedFrequencyInputs[threadIdx.x] = sinf(variation);
			}
			else
			{
				sharedFrequencyInputs[threadIdx.x] = cosf(variation);
			}
		}
		__syncthreads();

		if (threadIdx.x < 64)
		{
			float *offsetedWeights = weightsFrequencyL0 + threadIdx.x * 40;
			float accumulator = 0;
			for (uint8_t i = 0; i < 40; i++)
			{
				accumulator += offsetedWeights[i] * sharedFrequencyInputs[i];
			}
			sharedBiases[threadIdx.x] = __float2half(accumulator);
		}
		__syncthreads();

		float inversedDirections[3] = { 1.0f / rays[0], 1.0f / rays[1], 1.0f / rays[2] };

		float nearPlane = (aabb[0] - camera->origin.x) * inversedDirections[0];
		float farPlane = (aabb[3] - camera->origin.x) * inversedDirections[0];
		if (nearPlane > farPlane)
		{
			float swap = nearPlane;
			nearPlane = farPlane;
			farPlane = swap;
		}

		float nearY = (aabb[1] - camera->origin.y) * inversedDirections[1];
		float farY = (aabb[4] - camera->origin.y) * inversedDirections[1];
		float nearZ;
		float farZ;
		if (nearY > farY)
		{
			float swap = nearY;
			nearY = farY;
			farY = swap;
		}

		if (nearPlane > farY || nearY > farPlane) 
		{
			x = 1'000;
			y = 1'000;
			z = 1'000;
			rays[0] = 1'000;
			rays[1] = 1'000;
			rays[2] = 1'000;
			skipSampleGeneration = true;
			goto skip_sample_generation;
		}

		if (nearY > nearPlane)
		{
			nearPlane = nearY;
		}
		if (farY < farPlane)
		{
			farPlane = farY;
		}

		nearZ = (aabb[2] - camera->origin.z) * inversedDirections[2];
		farZ = (aabb[5] - camera->origin.z) * inversedDirections[2];
		if (nearZ > farZ)
		{
			float swap = nearZ;
			nearZ = farZ;
			farZ = swap;
		}

		if (nearPlane > farZ || nearZ > farPlane)
		{
			x = 1'000;
			y = 1'000;
			z = 1'000;
			rays[0] = 1'000;
			rays[1] = 1'000;
			rays[2] = 1'000;
			goto skip_sample_generation;
		}
		
		if (nearZ > nearPlane)
		{
			nearPlane = nearZ;
		}
		if (farZ < farPlane)
		{
			farPlane = farZ;
		}

		if (nearPlane < minNear)
		{
			nearPlane = minNear;
		}

		sampleDistributions = (farPlane - nearPlane) / numberOfSteps;

		skip_sample_generation:
		for (uint16_t blockIteration = 0; blockIteration < 2; blockIteration++)
		{
			if (!skipSampleGeneration)
			{
				zValue = nearPlane + (farPlane - nearPlane) * (threadIdx.x + blockIteration * threadsPerBlock) * inversedNumberOfSteps;
				x = camera->origin.x + rays[0] * zValue;
				y = camera->origin.y + rays[1] * zValue;
				z = camera->origin.z + rays[2] * zValue;
				x = fminf(fmaxf(x, aabb[0]), aabb[3]);
				y = fminf(fmaxf(y, aabb[1]), aabb[4]);
				z = fminf(fmaxf(z, aabb[2]), aabb[5]);
			}
			half *offsettedPositionInputs = sharedPositionInputs + threadIdx.x * 32;

			for (uint8_t i = 0; i < 32; i++)
			{
				offsettedPositionInputs[i] = __float2half(0.0f);
			}
			
			x = (x + 1.0f) * 0.5;
			y = (y + 1.0f) * 0.5;
			z = (z + 1.0f) * 0.5;

			for (uint8_t level = 0; level < 16; level++)
			{
				float* grid = embeddings + (offsets[level] << 1);
				uint32_t hashmapSize = offsets[level + 1] - offsets[level];
				float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
				uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

				float positions[3];
				positions[0] = x * scale + 0.5f;
				positions[1] = y * scale + 0.5f;
				positions[2] = z * scale + 0.5f;
						
				uint32_t gridPositions[3];
				gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
				gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
				gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
				
				float localPositions[3];
				localPositions[0] = positions[0] - gridPositions[0];
				localPositions[1] = positions[1] - gridPositions[1];
				localPositions[2] = positions[2] - gridPositions[2];
				
				for (uint8_t i = 0; i < 8; i++)
				{
					float w = 1.0f;
					uint32_t localGridPositions[3];
					
					for (uint8_t j = 0; j < 3; j++)
					{
						if ((i & (1 << j)) == 0)
						{
							w *= 1 - localPositions[j];
							localGridPositions[j] = gridPositions[j];
						}
						else
						{
							w *= localPositions[j];
							localGridPositions[j] = gridPositions[j] + 1;
						}
					}

					uint32_t stride = 1;
					uint32_t index = 0;

					for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
					{
						index += localGridPositions[j] * stride;
						stride *= resolution + 1;
					}

					if (stride > hashmapSize)
					{
						index = localGridPositions[0];
						index ^= localGridPositions[1] * 2654435761;
						index ^= localGridPositions[2] * 805459861;
					}

					index = (index % hashmapSize) * 2;
					
					offsettedPositionInputs[level * 2] = hadd(offsettedPositionInputs[level * 2], __float2half(w * grid[index]));
					offsettedPositionInputs[level * 2 + 1] = hadd(offsettedPositionInputs[level * 2 + 1], __float2half(w * grid[index + 1]));
				}
			}
			__syncthreads();

			offsettedPositionInputs = sharedPositionInputs;
			float *networkOutputs = sharedNetworkOutputs + warpId;
			uint8_t halfWarpId = warpId >> 1;
			half *inputsThread;
			for (uint16_t i = 0; i < 8; i++)
			{				
				#pragma unroll
				for (uint8_t j = 0; j < 16; j++)
				{
					uint8_t offset = (halfWarpId & 15);
					inputsThread = offsettedPositionInputs + offset * 32 + oddThread * 16 + sharedColumnIndex * 32 * 16; 
					half accumulator = 0;
					#pragma unroll
					for (uint8_t k = 0; k < 16; k++)
					{
						accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
					}
					accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
					if (oddThread)
					{
						intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * 16] = 
							hrelu(accumulator, one, sharedBiases[rowIndex]);
					}
					halfWarpId++;
				}
				__syncthreads();

				#pragma unroll
				for (uint8_t j = 0; j < 16; j++)
				{
					uint16_t offset = (halfWarpId & 15) * 64 + sharedColumnIndex * 64 * 16;
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
				for (uint8_t j = 0; j < 16; j++)
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
						networkOutputs[0] = __half2float(accumulator);
					}
					inputsThread += 128;
					networkOutputs += 8;
				}
				
				offsettedPositionInputs += 32 * 32;
			}
			__syncthreads();

			if (threadIdx.x == 0)
			{
				networkOutputs = sharedNetworkOutputs;

				for (uint16_t i = 0; i < 256; i++)
				{
					float4 input = reinterpret_cast<const float4 *>(networkOutputs)[i];
					float rgb[3] = { 
						1.0f / (1.0f + expf(-input.x)), 
						1.0f / (1.0f + expf(-input.y)), 
						1.0f / (1.0f + expf(-input.z)) 
					};

					float sigma = expf(input.w);
					float alpha = 1.0f - expf(-sampleDistributions * sigma);

					accumulators[0] += rem * rgb[0] * alpha;
					accumulators[1] += rem * rgb[1] * alpha;
					accumulators[2] += rem * rgb[2] * alpha;

					rem *= 1.0f - alpha;
				}
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			accumulators[0] += rem;
			accumulators[1] += rem;
			accumulators[2] += rem;

			RGBAOutputs[0] = static_cast<uint8_t>(min(max(0.0f, accumulators[0] * 255.0f), 255.0f));
			RGBAOutputs[1] = static_cast<uint8_t>(min(max(0.0f, accumulators[1] * 255.0f), 255.0f));
			RGBAOutputs[2] = static_cast<uint8_t>(min(max(0.0f, accumulators[2] * 255.0f), 255.0f));
			RGBAOutputs[3] = 255;
			RGBAOutputs += 4 * size;
		}
	}
}

void __global__ positionEncodingCoopKernel(Camera *camera, uint16_t size, uint16_t width, uint16_t height, float *aabb, 
							               float *embeddings, uint32_t *offsets, float *weightsFrequencyL0, half *weightsPositionL0, 
								           half *weightsL1, half *weightsL2, uint8_t *RGBAOutputs, uint16_t kernelIterations)
{
	constexpr uint16_t columnsPerIteration = 32;
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t L2InputsOffset = sharedColumnIndex * 64 + (threadIdx.x & 31) * 2;
	
	__shared__ float sharedFrequencyInputs[40];
	__shared__ half sharedBiases[64];
	__shared__ half sharedPositionInputs[32 * 32];
	__shared__ half intermidiateOutputsL0[64 * 32];
	__shared__ half intermidiateOutputsL1[64 * 32];
	__shared__ float sharedNetworkOutputs[4 * 32];
	half weightsL0Regs[16];
	half weightsL1Regs[32];
	half weightsL2Regs[2];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	RGBAOutputs += 4 * blockIdx.x;

	half *offsetedWeights = weightsPositionL0 + rowIndex * 32 + oddThread * 16;
	#pragma unroll
	for (uint8_t i = 0; i < 16; i++)
	{
		weightsL0Regs[i] = offsetedWeights[i];
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

	for (uint16_t kernelIteration = 0; kernelIteration < kernelIterations; kernelIteration++)
	{
		uint32_t offset = size * kernelIteration;
		uint16_t imageColumnIndex = (blockIdx.x + offset) / height;
		uint16_t imageRowIndex = (blockIdx.x + offset) % width;
		bool skipSampleGeneration = false;
		float accumulators[3] = { 0.0f, 0.0f, 0.0f };
		float rem = 1.0f;
		float rays[3];
		float zValue;
		float x;
		float y;
		float z;
		float sampleDistributions = 0;
		
		float3 baseDirection = 
		{ 
			(imageRowIndex + 0.5f - camera->cx) / camera->fl_x, 
			(imageColumnIndex + 0.5f - camera->cy) / camera->fl_y, 
			1.0f 
		};
		baseDirection = normalize(baseDirection);

		rays[0] = dot(baseDirection, camera->poseMatrix[0]);
		rays[1] = dot(baseDirection, camera->poseMatrix[1]);
		rays[2] = dot(baseDirection, camera->poseMatrix[2]);

		if (threadIdx.x < 3)
		{
			sharedFrequencyInputs[threadIdx.x] = rays[threadIdx.x];
		}
		else if (threadIdx.x < 39)
		{
			uint8_t frequency = 1 << ((threadIdx.x - 3) / 6);
			float ray = rays[threadIdx.x % 3];
			float variation = ray * frequency;
			if (threadIdx.x % 6 >= 3)
			{
				sharedFrequencyInputs[threadIdx.x] = sinf(variation);
			}
			else
			{
				sharedFrequencyInputs[threadIdx.x] = cosf(variation);
			}
		}
		__syncthreads();

		if (threadIdx.x < 64)
		{
			float *offsetedWeights = weightsFrequencyL0 + threadIdx.x * 40;
			float accumulator = 0;
			for (uint8_t i = 0; i < 40; i++)
			{
				accumulator += offsetedWeights[i] * sharedFrequencyInputs[i];
			}
			sharedBiases[threadIdx.x] = __float2half(accumulator);
		}
		__syncthreads();

		float inversedDirections[3] = { 1.0f / rays[0], 1.0f / rays[1], 1.0f / rays[2] };

		float nearPlane = (aabb[0] - camera->origin.x) * inversedDirections[0];
		float farPlane = (aabb[3] - camera->origin.x) * inversedDirections[0];
		if (nearPlane > farPlane)
		{
			float swap = nearPlane;
			nearPlane = farPlane;
			farPlane = swap;
		}

		float nearY = (aabb[1] - camera->origin.y) * inversedDirections[1];
		float farY = (aabb[4] - camera->origin.y) * inversedDirections[1];
		float nearZ;
		float farZ;
		if (nearY > farY)
		{
			float swap = nearY;
			nearY = farY;
			farY = swap;
		}

		if (nearPlane > farY || nearY > farPlane) 
		{
			if (threadIdx.x == 0)
			{
				RGBAOutputs[0] = 255;
				RGBAOutputs[1] = 255;
				RGBAOutputs[2] = 255;
				RGBAOutputs[3] = 255;
				RGBAOutputs += 4 * size;
			}
			continue;
		}

		if (nearY > nearPlane)
		{
			nearPlane = nearY;
		}
		if (farY < farPlane)
		{
			farPlane = farY;
		}

		nearZ = (aabb[2] - camera->origin.z) * inversedDirections[2];
		farZ = (aabb[5] - camera->origin.z) * inversedDirections[2];
		if (nearZ > farZ)
		{
			float swap = nearZ;
			nearZ = farZ;
			farZ = swap;
		}

		if (nearPlane > farZ || nearZ > farPlane)
		{
			if (threadIdx.x == 0)
			{
				RGBAOutputs[0] = 255;
				RGBAOutputs[1] = 255;
				RGBAOutputs[2] = 255;
				RGBAOutputs[3] = 255;
				RGBAOutputs += 4 * size;
			}
			continue;
		}
		
		if (nearZ > nearPlane)
		{
			nearPlane = nearZ;
		}
		if (farZ < farPlane)
		{
			farPlane = farZ;
		}

		if (nearPlane < minNear)
		{
			nearPlane = minNear;
		}

		sampleDistributions = (farPlane - nearPlane) / numberOfSteps;

		for (uint16_t blockIteration = 0; blockIteration < 16; blockIteration++)
		{
			uint8_t shiftedThreadId = threadIdx.x >> 3;
			uint8_t maskedThreadId = threadIdx.x & 7;
			if (!skipSampleGeneration)
			{
				zValue = nearPlane + (farPlane - nearPlane) * (shiftedThreadId + blockIteration * columnsPerIteration) * inversedNumberOfSteps;
				x = camera->origin.x + rays[0] * zValue;
				y = camera->origin.y + rays[1] * zValue;
				z = camera->origin.z + rays[2] * zValue;
				x = fminf(fmaxf(x, aabb[0]), aabb[3]);
				y = fminf(fmaxf(y, aabb[1]), aabb[4]);
				z = fminf(fmaxf(z, aabb[2]), aabb[5]);
			}
			half *offsettedPositionInputs = sharedPositionInputs + shiftedThreadId * 32 + 4 * maskedThreadId;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				offsettedPositionInputs[i] = __float2half(0.0f);
			}
			
			x = (x + 1.0f) * 0.5;
			y = (y + 1.0f) * 0.5;
			z = (z + 1.0f) * 0.5;

			for (uint8_t level = maskedThreadId << 1; level < (maskedThreadId << 1) + 2; level++)
			{
				float* grid = embeddings + (offsets[level] << 1);
				uint32_t hashmapSize = offsets[level + 1] - offsets[level];
				float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
				uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

				float positions[3];
				positions[0] = x * scale + 0.5f;
				positions[1] = y * scale + 0.5f;
				positions[2] = z * scale + 0.5f;
						
				uint32_t gridPositions[3];
				gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
				gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
				gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
				
				float localPositions[3];
				localPositions[0] = positions[0] - gridPositions[0];
				localPositions[1] = positions[1] - gridPositions[1];
				localPositions[2] = positions[2] - gridPositions[2];
				
				for (uint8_t i = 0; i < 8; i++)
				{
					float w = 1.0f;
					uint32_t localGridPositions[3];
					
					#pragma unroll
					for (uint8_t j = 0; j < 3; j++)
					{
						if ((i & (1 << j)) == 0)
						{
							w *= 1 - localPositions[j];
							localGridPositions[j] = gridPositions[j];
						}
						else
						{
							w *= localPositions[j];
							localGridPositions[j] = gridPositions[j] + 1;
						}
					}

					uint32_t stride = 1;
					uint32_t index = 0;

					for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
					{
						index += localGridPositions[j] * stride;
						stride *= resolution + 1;
					}

					if (stride > hashmapSize)
					{
						index = localGridPositions[0];
						index ^= localGridPositions[1] * 2654435761;
						index ^= localGridPositions[2] * 805459861;
					}

					index = (index % hashmapSize) * 2;
					
					offsettedPositionInputs[0] = hadd(offsettedPositionInputs[0], __float2half(w * grid[index]));
					offsettedPositionInputs[1] = hadd(offsettedPositionInputs[1], __float2half(w * grid[index + 1]));
				}
				offsettedPositionInputs += 2;
			}
			__syncthreads();
			
			float *networkOutputs = sharedNetworkOutputs + warpId;
			uint8_t halfWarpId = warpId >> 1;
			half *inputsThread;
					
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint8_t offset = (halfWarpId & 15);
				inputsThread = sharedPositionInputs + offset * 32 + oddThread * 16 + sharedColumnIndex * 32 * 16; 
				half accumulator = 0;
				#pragma unroll
				for (uint8_t k = 0; k < 16; k++)
				{
					accumulator = hfma(inputsThread[k], weightsL0Regs[k], accumulator);
				}
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
				if (oddThread)
				{
					intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * 16] = 
						hrelu(accumulator, one, sharedBiases[rowIndex]);
				}
				halfWarpId++;
			}
			__syncthreads();

			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint16_t offset = (halfWarpId & 15) * 64 + sharedColumnIndex * 64 * 16;
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
			for (uint8_t j = 0; j < 16; j++)
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
					networkOutputs[0] = __half2float(accumulator);
				}
				inputsThread += 128;
				networkOutputs += 8;
			}		
			__syncthreads();

			if (threadIdx.x == 0)
			{
				networkOutputs = sharedNetworkOutputs;

				for (uint16_t i = 0; i < 32; i++)
				{
					float4 input = reinterpret_cast<const float4 *>(networkOutputs)[i];
					float rgb[3] = { 
						1.0f / (1.0f + expf(-input.x)), 
						1.0f / (1.0f + expf(-input.y)), 
						1.0f / (1.0f + expf(-input.z)) 
					};

					float sigma = expf(input.w);
					float alpha = 1.0f - expf(-sampleDistributions * sigma);

					accumulators[0] += rem * rgb[0] * alpha;
					accumulators[1] += rem * rgb[1] * alpha;
					accumulators[2] += rem * rgb[2] * alpha;

					rem *= 1.0f - alpha;
				}

				sharedFrequencyInputs[39] = rem;
			}
			__syncthreads();
			if (sharedFrequencyInputs[39] < 0.0025f)
			{
				break;
			}
		}

		if (threadIdx.x == 0)
		{
			accumulators[0] += rem;
			accumulators[1] += rem;
			accumulators[2] += rem;

			RGBAOutputs[0] = static_cast<uint8_t>(min(max(0.0f, accumulators[0] * 255.0f), 255.0f));
			RGBAOutputs[1] = static_cast<uint8_t>(min(max(0.0f, accumulators[1] * 255.0f), 255.0f));
			RGBAOutputs[2] = static_cast<uint8_t>(min(max(0.0f, accumulators[2] * 255.0f), 255.0f));
			RGBAOutputs[3] = 255;
			RGBAOutputs += 4 * size;
		}
	}
}

void __global__ moreCoopKernel(Camera *camera, uint16_t size, uint16_t width, uint16_t height, float *aabb, 
							   float *embeddings, uint32_t *offsets, half *weightsFrequencyL0, half *weightsPositionL0, 
							   half *weightsL1, half *weightsL2, uint8_t *RGBAOutputs, uint16_t kernelIterations)
{
	constexpr uint16_t columnsPerIteration = 32;
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t sharedColumnIndex = threadIdx.x >> 7;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = (threadId & 127) >> 1;
	uint8_t L2InputsOffset = sharedColumnIndex * 64 + (threadIdx.x & 31) * 2;
	
	__shared__ half sharedFrequencyInputs[40];
	__shared__ half sharedBiases[64];
	__shared__ half sharedPositionInputs[32 * 32];
	__shared__ half intermidiateOutputsL0[64 * 32];
	__shared__ half intermidiateOutputsL1[64 * 32];
	__shared__ float sharedNetworkOutputs[4 * 32];
	__shared__ float sharedRaysAndRem[4];
	__shared__ float sharedNormalizedBaseDirections[4];
	__shared__ float sharedInversedDirections[4];
	__shared__ float sharedPlanes[8];
	half weightsFrequencyL0Regs[10];
	half weightsPositionL0Regs[16];
	half weightsL1Regs[32];
	half weightsL2Regs[2];
	half one = __float2half(1.0f);
	half zero = __float2half(0.0f);

	RGBAOutputs += 4 * blockIdx.x;

	half *offsetedWeights = weightsFrequencyL0 + threadIdx.x * 10;
	#pragma unroll
	for (uint8_t i = 0; i < 10; i++)
	{
		weightsFrequencyL0Regs[i] = offsetedWeights[i];
	}

	offsetedWeights = weightsPositionL0 + rowIndex + oddThread * 16 * 64;
	#pragma unroll
	for (uint8_t i = 0; i < 16; i++)
	{
		weightsPositionL0Regs[i] = offsetedWeights[i * 64];
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

	for (uint16_t kernelIteration = 0; kernelIteration < kernelIterations; kernelIteration++)
	{
		uint32_t offset = size * kernelIteration;
		uint16_t imageColumnIndex = (blockIdx.x + offset) / height;
		uint16_t imageRowIndex = (blockIdx.x + offset) % width;
		bool skipSampleGeneration = false;
		float outputAccumulator = 0.0f;
		float rem = 1.0f;
		float zValue;
		float x;
		float y;
		float z;
		float sampleDistributions = 0;

		if (threadIdx.x == 0)
		{
			x = (imageRowIndex + 0.5f - camera->cx) / camera->fl_x;
			y = (imageColumnIndex + 0.5f - camera->cy) / camera->fl_y;

			float reversedDotProduct = rsqrtf(x * x + y * y + 1);
			sharedNormalizedBaseDirections[0] = x * reversedDotProduct;
			sharedNormalizedBaseDirections[1] = y * reversedDotProduct;
			sharedNormalizedBaseDirections[2] = reversedDotProduct;
		}
		__syncthreads();

		if (threadIdx.x < 3)
		{	
			sharedRaysAndRem[threadIdx.x] = sharedNormalizedBaseDirections[0] * camera->poseMatrix[threadIdx.x].x + 
											sharedNormalizedBaseDirections[1] * camera->poseMatrix[threadIdx.x].y +
											sharedNormalizedBaseDirections[2] * camera->poseMatrix[threadIdx.x].z;
		}
		__syncthreads();
		
		if (threadIdx.x < 36)
		{
			uint8_t frequency = 1 << (threadIdx.x / 6);
			float ray = sharedRaysAndRem[threadIdx.x % 3];
			float variation = ray * frequency;
			if (threadIdx.x % 6 < 3)
			{
				sharedFrequencyInputs[threadIdx.x + 3] = __float2half(sinf(variation));
			}
			else
			{
				sharedFrequencyInputs[threadIdx.x + 3] = __float2half(cosf(variation));
			}
		}
		else if (threadIdx.x < 39)
		{
			sharedFrequencyInputs[threadIdx.x - 36] = __float2half(sharedRaysAndRem[threadIdx.x - 36]);
		}
		__syncthreads();

		uint8_t shiftedThreadId = threadIdx.x >> 2;
		uint8_t maskedMultipliedThreadId = (threadIdx.x & 3) * 10;
		half *offsetedFrequencyInputs = sharedFrequencyInputs + maskedMultipliedThreadId;
		half accumulator = 0;
		#pragma unroll
		for (uint8_t i = 0; i < 10; i++)
		{
			accumulator = hfma(weightsFrequencyL0Regs[i], offsetedFrequencyInputs[i], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 2));
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (maskedMultipliedThreadId == 0)
		{
			sharedBiases[shiftedThreadId] = accumulator;
		}
		__syncthreads();
		
		if (threadIdx.x < 3)
		{
			sharedInversedDirections[threadIdx.x] = 1.0f / sharedRaysAndRem[threadIdx.x];
		}
		__syncthreads();
		if (threadIdx.x < 6)
		{
			uint8_t moduloThredId = threadIdx.x % 3;
			sharedPlanes[threadIdx.x] = (aabb[threadIdx.x] - reinterpret_cast<float *>(&camera->origin)[moduloThredId]) * 
										 sharedInversedDirections[moduloThredId];
		}
		__syncthreads();
		if (threadIdx.x < 3)
		{
			if (sharedPlanes[threadIdx.x] > sharedPlanes[threadIdx.x + 3])
			{
				float swap = sharedPlanes[threadIdx.x];
				sharedPlanes[threadIdx.x] = sharedPlanes[threadIdx.x + 3];
				sharedPlanes[threadIdx.x + 3] = swap;
			}
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			if (sharedPlanes[1] > sharedPlanes[0])
			{
				sharedPlanes[0] = sharedPlanes[1];
			}
			if (sharedPlanes[4] < sharedPlanes[3])
			{
				sharedPlanes[3] = sharedPlanes[4];
			}
			if (sharedPlanes[2] > sharedPlanes[0])
			{
				sharedPlanes[0] = sharedPlanes[2];
			}
			if (sharedPlanes[5] < sharedPlanes[3])
			{
				sharedPlanes[3] = sharedPlanes[5];
			}
			if (sharedPlanes[0] < minNear)
			{
				sharedPlanes[0] = minNear;
			}

			sharedPlanes[7] = sharedPlanes[0] > sharedPlanes[4] || 
							  sharedPlanes[1] > sharedPlanes[3] || 
							  sharedPlanes[0] > sharedPlanes[5] || 
							  sharedPlanes[2] > sharedPlanes[3];
		}
		__syncthreads();
		if (sharedPlanes[7])
		{
			if (threadIdx.x < 4)
			{
				RGBAOutputs[threadIdx.x] = 255;
				RGBAOutputs += 4 * size;
			}
			continue;
		}
		if (threadIdx.x < 3)
		{
			sampleDistributions = (sharedPlanes[3] - sharedPlanes[0]) / numberOfSteps;
		}

		for (uint16_t blockIteration = 0; blockIteration < 16; blockIteration++)
		{
			uint8_t shiftedThreadId = threadIdx.x >> 3;
			uint8_t maskedThreadId = threadIdx.x & 7;
			if (!skipSampleGeneration)
			{
				zValue = sharedPlanes[0] + (sharedPlanes[3] - sharedPlanes[0]) * (shiftedThreadId + blockIteration * columnsPerIteration) * inversedNumberOfSteps;
				x = camera->origin.x + sharedRaysAndRem[0] * zValue;
				y = camera->origin.y + sharedRaysAndRem[1] * zValue;
				z = camera->origin.z + sharedRaysAndRem[2] * zValue;
				x = fminf(fmaxf(x, aabb[0]), aabb[3]);
				y = fminf(fmaxf(y, aabb[1]), aabb[4]);
				z = fminf(fmaxf(z, aabb[2]), aabb[5]);
			}
			half *offsettedPositionInputs = sharedPositionInputs + shiftedThreadId * 32 + 4 * maskedThreadId;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				offsettedPositionInputs[i] = __float2half(0.0f);
			}
			
			x = (x + 1.0f) * 0.5;
			y = (y + 1.0f) * 0.5;
			z = (z + 1.0f) * 0.5;

			for (uint8_t level = maskedThreadId << 1; level < (maskedThreadId << 1) + 2; level++)
			{
				float* grid = embeddings + (offsets[level] << 1);
				uint32_t hashmapSize = offsets[level + 1] - offsets[level];
				float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
				uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

				float positions[3];
				positions[0] = x * scale + 0.5f;
				positions[1] = y * scale + 0.5f;
				positions[2] = z * scale + 0.5f;
						
				uint32_t gridPositions[3];
				gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
				gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
				gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
				
				float localPositions[3];
				localPositions[0] = positions[0] - gridPositions[0];
				localPositions[1] = positions[1] - gridPositions[1];
				localPositions[2] = positions[2] - gridPositions[2];
				
				for (uint8_t i = 0; i < 8; i++)
				{
					float w = 1.0f;
					uint32_t localGridPositions[3];
					
					#pragma unroll
					for (uint8_t j = 0; j < 3; j++)
					{
						if ((i & (1 << j)) == 0)
						{
							w *= 1 - localPositions[j];
							localGridPositions[j] = gridPositions[j];
						}
						else
						{
							w *= localPositions[j];
							localGridPositions[j] = gridPositions[j] + 1;
						}
					}

					uint32_t stride = 1;
					uint32_t index = 0;

					for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
					{
						index += localGridPositions[j] * stride;
						stride *= resolution + 1;
					}

					if (stride > hashmapSize)
					{
						index = localGridPositions[0];
						index ^= localGridPositions[1] * 2654435761;
						index ^= localGridPositions[2] * 805459861;
					}

					index = (index % hashmapSize) * 2;
					
					offsettedPositionInputs[0] = hadd(offsettedPositionInputs[0], __float2half(w * grid[index]));
					offsettedPositionInputs[1] = hadd(offsettedPositionInputs[1], __float2half(w * grid[index + 1]));
				}
				offsettedPositionInputs += 2;
			}
			__syncthreads();
						
			float *networkOutputs = sharedNetworkOutputs + warpId;
			uint8_t halfWarpId = warpId >> 1;
			half *inputsThread;
					
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint8_t offset = (halfWarpId & 15);
				inputsThread = sharedPositionInputs + offset * 32 + oddThread * 16 + sharedColumnIndex * 32 * 16; 
				half accumulator = 0;
				#pragma unroll
				for (uint8_t k = 0; k < 16; k++)
				{
					accumulator = hfma(inputsThread[k], weightsPositionL0Regs[k], accumulator);
				}
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
				if (oddThread)
				{
					intermidiateOutputsL0[offset * 64 + rowIndex + sharedColumnIndex * 64 * 16] = 
						hrelu(accumulator, one, sharedBiases[rowIndex]);
				}
				halfWarpId++;
			}
			__syncthreads();

			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint16_t offset = (halfWarpId & 15) * 64 + sharedColumnIndex * 64 * 16;
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
			for (uint8_t j = 0; j < 16; j++)
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
					networkOutputs[0] = __half2float(accumulator);
				}
				inputsThread += 128;
				networkOutputs += 8;
			}		
			__syncthreads();
			
			if (threadIdx.x < 3)
			{
				networkOutputs = sharedNetworkOutputs;

				for (uint16_t i = 0; i < 32; i++)
				{
					float color = 1.0f / (1.0f + expf(-networkOutputs[threadIdx.x]));
					float sigma = expf(networkOutputs[3]);
					float alpha = 1.0f - expf(-sampleDistributions * sigma);
					
					outputAccumulator += rem * color * alpha;
					rem *= 1.0f - alpha;
					networkOutputs += 4;
				}

				if (threadIdx.x == 0)
				{
					sharedRaysAndRem[3] = rem;
				}
			}
			__syncthreads();
			if (sharedRaysAndRem[3] < 0.0025f)
			{
				break;
			}
		}

		if (threadIdx.x < 3)
		{
			outputAccumulator += rem;
			RGBAOutputs[threadIdx.x] = static_cast<uint8_t>(min(max(0.0f, outputAccumulator * 255.0f), 255.0f));			
		}
		else if (threadIdx.x == 3)
		{
			RGBAOutputs[3] = 255;
		}
		RGBAOutputs += 4 * size;
	}
}

void __global__ lessThreadsFixedIterationsKernel(Camera *camera, uint16_t size, uint16_t width, uint16_t height, float *aabb, 
							                     float *embeddings, uint32_t *offsets, half *weightsFrequencyL0, half *weightsPositionL0, 
							                     half *weightsL1, half *weightsL2, uint8_t *RGBAOutputs, uint16_t kernelIterations)
{
	constexpr uint16_t columnsPerIteration = 16;
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = threadIdx.x >> 1;
	uint8_t L2InputsOffset = (threadIdx.x & 31) * 2;
	
	__shared__ half sharedFrequencyInputs[40];
	__shared__ half sharedBiases[64];
	__shared__ half sharedPositionInputs[32 * columnsPerIteration];
	__shared__ half intermidiateOutputsL0[64 * columnsPerIteration];
	__shared__ half intermidiateOutputsL1[64 * columnsPerIteration];
	__shared__ float sharedNetworkOutputs[4 * columnsPerIteration];
	__shared__ float sharedRaysAndRem[4];
	__shared__ float sharedNormalizedBaseDirections[4];
	__shared__ float sharedInversedDirections[4];
	__shared__ float sharedPlanes[8];
	half weightsPositionL0Regs[16];
	half weightsL1Regs[32];
	half weightsL2Regs[2];

	RGBAOutputs += 4 * blockIdx.x;

	half *offsetedWeights = weightsPositionL0 + rowIndex + oddThread * 16 * 64;
	#pragma unroll
	for (uint8_t i = 0; i < 16; i++)
	{
		weightsPositionL0Regs[i] = offsetedWeights[i * 64];
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

	for (uint16_t kernelIteration = 0; kernelIteration < kernelIterations; kernelIteration++)
	{
		uint32_t offset = size * kernelIteration;
		uint16_t imageColumnIndex = (blockIdx.x + offset) / height;
		uint16_t imageRowIndex = (blockIdx.x + offset) % width;
		float outputAccumulator = 0.0f;
		float rem = 1.0f;
		float zValue;
		float x;
		float y;
		float z;
		float sampleDistributions = 0;

		if (threadIdx.x == 0)
		{
			x = (imageRowIndex + 0.5f - camera->cx) / camera->fl_x;
			y = (imageColumnIndex + 0.5f - camera->cy) / camera->fl_y;

			float reversedDotProduct = rsqrtf(x * x + y * y + 1);
			sharedNormalizedBaseDirections[0] = x * reversedDotProduct;
			sharedNormalizedBaseDirections[1] = y * reversedDotProduct;
			sharedNormalizedBaseDirections[2] = reversedDotProduct;
		}
		__syncthreads();

		if (threadIdx.x < 3)
		{	
			sharedRaysAndRem[threadIdx.x] = sharedNormalizedBaseDirections[0] * camera->poseMatrix[threadIdx.x].x + 
											sharedNormalizedBaseDirections[1] * camera->poseMatrix[threadIdx.x].y +
											sharedNormalizedBaseDirections[2] * camera->poseMatrix[threadIdx.x].z;
		}
		__syncthreads();
		
		if (threadIdx.x < 36)
		{
			uint8_t frequency = 1 << (threadIdx.x / 6);
			float ray = sharedRaysAndRem[threadIdx.x % 3];
			float variation = ray * frequency;
			if (threadIdx.x % 6 < 3)
			{
				sharedFrequencyInputs[threadIdx.x + 3] = __float2half(sinf(variation));
			}
			else
			{
				sharedFrequencyInputs[threadIdx.x + 3] = __float2half(cosf(variation));
			}
		}
		else if (threadIdx.x < 39)
		{
			sharedFrequencyInputs[threadIdx.x - 36] = __float2half(sharedRaysAndRem[threadIdx.x - 36]);
		}
		__syncthreads();

		uint8_t maskedMultipliedThreadId = (threadIdx.x & 1) * 20;
		half *offsetedFrequencyInputs = sharedFrequencyInputs + maskedMultipliedThreadId;
		offsetedWeights = weightsFrequencyL0 + threadIdx.x * 20;
		half accumulator = 0;
		#pragma unroll
		for (uint8_t i = 0; i < 20; i++)
		{
			accumulator = hfma(offsetedWeights[i], offsetedFrequencyInputs[i], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (maskedMultipliedThreadId == 0)
		{
			sharedBiases[rowIndex] = accumulator;
		}
				
		if (threadIdx.x < 3)
		{
			sharedInversedDirections[threadIdx.x] = 1.0f / sharedRaysAndRem[threadIdx.x];
		}
		__syncthreads();
		if (threadIdx.x < 6)
		{
			uint8_t moduloThredId = threadIdx.x % 3;
			sharedPlanes[threadIdx.x] = (aabb[threadIdx.x] - reinterpret_cast<float *>(&camera->origin)[moduloThredId]) * 
										 sharedInversedDirections[moduloThredId];
		}
		__syncthreads();
		if (threadIdx.x < 3)
		{
			if (sharedPlanes[threadIdx.x] > sharedPlanes[threadIdx.x + 3])
			{
				float swap = sharedPlanes[threadIdx.x];
				sharedPlanes[threadIdx.x] = sharedPlanes[threadIdx.x + 3];
				sharedPlanes[threadIdx.x + 3] = swap;
			}
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			if (sharedPlanes[1] > sharedPlanes[0])
			{
				sharedPlanes[0] = sharedPlanes[1];
			}
			if (sharedPlanes[4] < sharedPlanes[3])
			{
				sharedPlanes[3] = sharedPlanes[4];
			}
			if (sharedPlanes[2] > sharedPlanes[0])
			{
				sharedPlanes[0] = sharedPlanes[2];
			}
			if (sharedPlanes[5] < sharedPlanes[3])
			{
				sharedPlanes[3] = sharedPlanes[5];
			}
			if (sharedPlanes[0] < minNear)
			{
				sharedPlanes[0] = minNear;
			}

			sharedPlanes[7] = sharedPlanes[0] > sharedPlanes[4] || 
							  sharedPlanes[1] > sharedPlanes[3] || 
							  sharedPlanes[0] > sharedPlanes[5] || 
							  sharedPlanes[2] > sharedPlanes[3];
		}
		__syncthreads();
		if (sharedPlanes[7])
		{
			if (threadIdx.x < 4)
			{
				RGBAOutputs[threadIdx.x] = 255;
				RGBAOutputs += 4 * size;
			}
			continue;
		}
		if (threadIdx.x < 3)
		{
			sampleDistributions = (sharedPlanes[3] - sharedPlanes[0]) / numberOfSteps;
		}

		for (uint8_t blockIteration = 0; blockIteration < 32; blockIteration++)
		{
			uint8_t shiftedThreadId = threadIdx.x >> 3;
			uint8_t maskedThreadId = threadIdx.x & 7;
			zValue = sharedPlanes[0] + (sharedPlanes[3] - sharedPlanes[0]) * (shiftedThreadId + blockIteration * columnsPerIteration) * inversedNumberOfSteps;
			x = camera->origin.x + sharedRaysAndRem[0] * zValue;
			y = camera->origin.y + sharedRaysAndRem[1] * zValue;
			z = camera->origin.z + sharedRaysAndRem[2] * zValue;
			x = fminf(fmaxf(x, aabb[0]), aabb[3]);
			y = fminf(fmaxf(y, aabb[1]), aabb[4]);
			z = fminf(fmaxf(z, aabb[2]), aabb[5]);
			half *offsettedPositionInputs = sharedPositionInputs + shiftedThreadId * 32 + 4 * maskedThreadId;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				offsettedPositionInputs[i] = __float2half(0.0f);
			}
			
			x = (x + 1.0f) * 0.5;
			y = (y + 1.0f) * 0.5;
			z = (z + 1.0f) * 0.5;

			for (uint8_t level = maskedThreadId << 1; level < (maskedThreadId << 1) + 2; level++)
			{
				float* grid = embeddings + (offsets[level] << 1);
				uint32_t hashmapSize = offsets[level + 1] - offsets[level];
				float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
				uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

				float positions[3];
				positions[0] = x * scale + 0.5f;
				positions[1] = y * scale + 0.5f;
				positions[2] = z * scale + 0.5f;
						
				uint32_t gridPositions[3];
				gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
				gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
				gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
				
				float localPositions[3];
				localPositions[0] = positions[0] - gridPositions[0];
				localPositions[1] = positions[1] - gridPositions[1];
				localPositions[2] = positions[2] - gridPositions[2];
				
				for (uint8_t i = 0; i < 8; i++)
				{
					float w = 1.0f;
					uint32_t localGridPositions[3];
					
					#pragma unroll
					for (uint8_t j = 0; j < 3; j++)
					{
						if ((i & (1 << j)) == 0)
						{
							w *= 1 - localPositions[j];
							localGridPositions[j] = gridPositions[j];
						}
						else
						{
							w *= localPositions[j];
							localGridPositions[j] = gridPositions[j] + 1;
						}
					}

					uint32_t stride = 1;
					uint32_t index = 0;

					for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
					{
						index += localGridPositions[j] * stride;
						stride *= resolution + 1;
					}

					if (stride > hashmapSize)
					{
						index = localGridPositions[0];
						index ^= localGridPositions[1] * 2654435761;
						index ^= localGridPositions[2] * 805459861;
					}

					index = (index % hashmapSize) * 2;
					
					offsettedPositionInputs[0] = hadd(offsettedPositionInputs[0], __float2half(w * grid[index]));
					offsettedPositionInputs[1] = hadd(offsettedPositionInputs[1], __float2half(w * grid[index + 1]));
				}
				offsettedPositionInputs += 2;
			}
			__syncthreads();
			
									
			float *networkOutputs = sharedNetworkOutputs + warpId;
			uint8_t halfWarpId = warpId >> 1;
			half *inputsThread;
					
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint8_t offset = (halfWarpId & 15);
				inputsThread = sharedPositionInputs + offset * 32 + oddThread * 16; 
				half accumulator = 0;
				#pragma unroll
				for (uint8_t k = 0; k < 16; k++)
				{
					accumulator = hfma(inputsThread[k], weightsPositionL0Regs[k], accumulator);
				}
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
				if (oddThread)
				{
					intermidiateOutputsL0[offset * 64 + rowIndex] = hrelu(accumulator, __float2half(1.0f), sharedBiases[rowIndex]);
				}
				halfWarpId++;
			}
			__syncthreads();

			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint16_t offset = (halfWarpId & 15) * 64;
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
					intermidiateOutputsL1[offset + rowIndex] = hrelu(accumulator, __float2half(1.0f), __float2half(0.0f));
				}
				halfWarpId++;
			}
			__syncthreads();

			inputsThread = intermidiateOutputsL1 + L2InputsOffset;
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
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
					networkOutputs[0] = __half2float(accumulator);
				}
				inputsThread += 64;
				networkOutputs += 4;
			}		
			__syncthreads();
						
			if (threadIdx.x < 3)
			{
				networkOutputs = sharedNetworkOutputs;

				for (uint16_t i = 0; i < 16; i++)
				{
					float color = 1.0f / (1.0f + expf(-networkOutputs[threadIdx.x]));
					float sigma = expf(networkOutputs[3]);
					float alpha = 1.0f - expf(-sampleDistributions * sigma);
					
					outputAccumulator += rem * color * alpha;
					rem *= 1.0f - alpha;
					networkOutputs += 4;
				}

				if (threadIdx.x == 0)
				{
					sharedRaysAndRem[3] = rem;
				}
			}
			__syncthreads();
			if (sharedRaysAndRem[3] < 0.0025f)
			{
				break;
			}
		}

		if (threadIdx.x < 3)
		{
			outputAccumulator += rem;
			RGBAOutputs[threadIdx.x] = static_cast<uint8_t>(min(max(0.0f, outputAccumulator * 255.0f), 255.0f));			
		}
		else if (threadIdx.x == 3)
		{
			RGBAOutputs[3] = 255;
		}
		RGBAOutputs += 4 * size;
	}
}

template<uint16_t width, uint16_t height, uint32_t numberOfPixels, uint16_t numberOfBlocks>
void __global__ pixelPoolKernel(Camera *camera, uint32_t *pixelCounter, float *aabb, float *embeddings, uint32_t *offsets, 
								half *weightsFrequencyL0, half *weightsPositionL0, half *weightsL1, half *weightsL2, 
								uint8_t *RGBAOutputs, uint16_t kernelIterations)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId == 0)
	{
		*pixelCounter = numberOfBlocks;
	} 
	constexpr uint16_t columnsPerIteration = 16;
	constexpr float minNear = 0.2f;
	constexpr float inversedNumberOfSteps = 1.0f / 511.0f;
	constexpr float numberOfSteps = 512.0f;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t oddThread = threadIdx.x & 1;
	uint8_t rowIndex = threadIdx.x >> 1;
	uint8_t L2InputsOffset = (threadIdx.x & 31) * 2;
	
	__shared__ half sharedFrequencyInputs[40];
	__shared__ half sharedBiases[64];
	__shared__ half sharedPositionInputs[32 * columnsPerIteration];
	__shared__ half intermidiateOutputsL0[64 * columnsPerIteration];
	__shared__ half intermidiateOutputsL1[64 * columnsPerIteration];
	__shared__ float sharedNetworkOutputs[4 * columnsPerIteration];
	__shared__ float sharedPlanes[8];
	__shared__ float sharedRaysAndRem[4];
	__shared__ float sharedNormalizedBaseDirections[4];
	__shared__ float sharedInversedDirections[4];
	__shared__ uint32_t sharedPixelNumbers[4];
	half weightsFrequencyL0Regs[20];
	half weightsPositionL0Regs[16];
	half weightsL1Regs[32];
	half weightsL2Regs[2];

	half *offsetedWeights = weightsFrequencyL0 + threadIdx.x;
	#pragma unroll
	for (uint8_t i = 0; i < 20; i++)
	{
		weightsFrequencyL0Regs[i] = offsetedWeights[i * 128];
	}

	offsetedWeights = weightsPositionL0 + rowIndex + oddThread * 16 * 64;
	#pragma unroll
	for (uint8_t i = 0; i < 16; i++)
	{
		weightsPositionL0Regs[i] = offsetedWeights[i * 64];
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
	
	uint32_t pixelNumber = blockIdx.x;
	uint8_t *offsettedRGBAOutputs = RGBAOutputs + 4 * blockIdx.x;

	while (true)
	{
		uint16_t imageColumnIndex = pixelNumber / height;
		uint16_t imageRowIndex = pixelNumber % width;
		float outputAccumulator = 0.0f;
		float rem = 1.0f;
		float x;
		float y;
		float z;
		float sampleDistributions = 0;

		if (threadIdx.x == 0)
		{
			x = (imageRowIndex + 0.5f - camera->cx) / camera->fl_x;
			y = (imageColumnIndex + 0.5f - camera->cy) / camera->fl_y;

			float reversedDotProduct = rsqrtf(x * x + y * y + 1);
			sharedNormalizedBaseDirections[0] = x * reversedDotProduct;
			sharedNormalizedBaseDirections[1] = y * reversedDotProduct;
			sharedNormalizedBaseDirections[2] = reversedDotProduct;
		}
		__syncthreads();

		if (threadIdx.x < 3)
		{	
			sharedRaysAndRem[threadIdx.x] = sharedNormalizedBaseDirections[0] * camera->poseMatrix[threadIdx.x].x + 
											sharedNormalizedBaseDirections[1] * camera->poseMatrix[threadIdx.x].y +
											sharedNormalizedBaseDirections[2] * camera->poseMatrix[threadIdx.x].z;
		}
		__syncthreads();
		
		if (threadIdx.x < 36)
		{
			uint8_t frequency = 1 << (threadIdx.x / 6);
			float ray = sharedRaysAndRem[threadIdx.x % 3];
			float variation = ray * frequency;
			if (threadIdx.x % 6 < 3)
			{
				sharedFrequencyInputs[threadIdx.x + 3] = __float2half(sinf(variation));
			}
			else
			{
				sharedFrequencyInputs[threadIdx.x + 3] = __float2half(cosf(variation));
			}
		}
		else if (threadIdx.x < 39)
		{
			sharedFrequencyInputs[threadIdx.x - 36] = __float2half(sharedRaysAndRem[threadIdx.x - 36]);
		}
		__syncthreads();

		uint8_t maskedMultipliedThreadId = (threadIdx.x & 1) * 20;
		half *offsetedFrequencyInputs = sharedFrequencyInputs + maskedMultipliedThreadId;
		half accumulator = 0;
		#pragma unroll
		for (uint8_t i = 0; i < 20; i++)
		{
			accumulator = hfma(weightsFrequencyL0Regs[i], offsetedFrequencyInputs[i], accumulator);
		}
		accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
		if (maskedMultipliedThreadId == 0)
		{
			sharedBiases[rowIndex] = accumulator;
		}
				
		if (threadIdx.x < 3)
		{
			sharedInversedDirections[threadIdx.x] = 1.0f / sharedRaysAndRem[threadIdx.x];
		}
		__syncthreads();
		if (threadIdx.x < 6)
		{
			uint8_t moduloThredId = threadIdx.x % 3;
			sharedPlanes[threadIdx.x] = (aabb[threadIdx.x] - reinterpret_cast<float *>(&camera->origin)[moduloThredId]) * 
										 sharedInversedDirections[moduloThredId];
		}
		__syncthreads();
		if (threadIdx.x < 3)
		{
			if (sharedPlanes[threadIdx.x] > sharedPlanes[threadIdx.x + 3])
			{
				float swap = sharedPlanes[threadIdx.x];
				sharedPlanes[threadIdx.x] = sharedPlanes[threadIdx.x + 3];
				sharedPlanes[threadIdx.x + 3] = swap;
			}
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			if (sharedPlanes[1] > sharedPlanes[0])
			{
				sharedPlanes[0] = sharedPlanes[1];
			}
			if (sharedPlanes[4] < sharedPlanes[3])
			{
				sharedPlanes[3] = sharedPlanes[4];
			}
			if (sharedPlanes[2] > sharedPlanes[0])
			{
				sharedPlanes[0] = sharedPlanes[2];
			}
			if (sharedPlanes[5] < sharedPlanes[3])
			{
				sharedPlanes[3] = sharedPlanes[5];
			}
			if (sharedPlanes[0] < minNear)
			{
				sharedPlanes[0] = minNear;
			}

			sharedPlanes[7] = sharedPlanes[0] > sharedPlanes[4] || 
							  sharedPlanes[1] > sharedPlanes[3] || 
							  sharedPlanes[0] > sharedPlanes[5] || 
							  sharedPlanes[2] > sharedPlanes[3];
		}
		__syncthreads();
		if (sharedPlanes[7])
		{
			if (threadIdx.x < 4)
			{
				offsettedRGBAOutputs[threadIdx.x] = 255;
			}
			if (threadIdx.x == 0)
			{
				sharedPixelNumbers[0] = atomicAdd(pixelCounter, 1U);
			}
			__syncthreads();
			pixelNumber = sharedPixelNumbers[0];
			if (pixelNumber >= numberOfPixels)
			{
				return;
			}
			offsettedRGBAOutputs = RGBAOutputs + pixelNumber * 4;
			continue;
		}
		if (threadIdx.x < 3)
		{
			sampleDistributions = (sharedPlanes[3] - sharedPlanes[0]) / numberOfSteps;
		}

		for (uint8_t blockIteration = 0; blockIteration < 32; blockIteration++)
		{
			uint8_t shiftedThreadId = threadIdx.x >> 3;
			uint8_t maskedThreadId = threadIdx.x & 7;
			float zValue = sharedPlanes[0] + (sharedPlanes[3] - sharedPlanes[0]) * (shiftedThreadId + blockIteration * columnsPerIteration) * inversedNumberOfSteps;
			x = camera->origin.x + sharedRaysAndRem[0] * zValue;
			y = camera->origin.y + sharedRaysAndRem[1] * zValue;
			z = camera->origin.z + sharedRaysAndRem[2] * zValue;
			x = fminf(fmaxf(x, aabb[0]), aabb[3]);
			y = fminf(fmaxf(y, aabb[1]), aabb[4]);
			z = fminf(fmaxf(z, aabb[2]), aabb[5]);
			half *offsettedPositionInputs = sharedPositionInputs + shiftedThreadId * 32 + 4 * maskedThreadId;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				offsettedPositionInputs[i] = __float2half(0.0f);
			}
			
			x = (x + 1.0f) * 0.5;
			y = (y + 1.0f) * 0.5;
			z = (z + 1.0f) * 0.5;

			for (uint8_t level = maskedThreadId << 1; level < (maskedThreadId << 1) + 2; level++)
			{
				float* grid = embeddings + (offsets[level] << 1);
				uint32_t hashmapSize = offsets[level + 1] - offsets[level];
				float scale = static_cast<float>(1U << (level + 4)) - 1.0f;
				uint32_t resolution = static_cast<uint32_t>(ceilf(scale)) + 1;

				float positions[3];
				positions[0] = x * scale + 0.5f;
				positions[1] = y * scale + 0.5f;
				positions[2] = z * scale + 0.5f;
						
				uint32_t gridPositions[3];
				gridPositions[0] = static_cast<uint32_t>(floorf(positions[0]));
				gridPositions[1] = static_cast<uint32_t>(floorf(positions[1]));
				gridPositions[2] = static_cast<uint32_t>(floorf(positions[2]));
				
				float localPositions[3];
				localPositions[0] = positions[0] - gridPositions[0];
				localPositions[1] = positions[1] - gridPositions[1];
				localPositions[2] = positions[2] - gridPositions[2];
				
				#pragma unroll
				for (uint8_t i = 0; i < 8; i++)
				{
					float w = 1.0f;
					uint32_t localGridPositions[3];
					
					#pragma unroll
					for (uint8_t j = 0; j < 3; j++)
					{
						if ((i & (1 << j)) == 0)
						{
							w *= 1 - localPositions[j];
							localGridPositions[j] = gridPositions[j];
						}
						else
						{
							w *= localPositions[j];
							localGridPositions[j] = gridPositions[j] + 1;
						}
					}

					uint32_t stride = 1;
					uint32_t index = 0;

					for (uint8_t j = 0; j < 3 && stride <= hashmapSize; j++)
					{
						index += localGridPositions[j] * stride;
						stride *= resolution + 1;
					}

					if (stride > hashmapSize)
					{
						index = localGridPositions[0];
						index ^= localGridPositions[1] * 2654435761;
						index ^= localGridPositions[2] * 805459861;
					}

					index = (index % hashmapSize) * 2;
					
					offsettedPositionInputs[0] = hadd(offsettedPositionInputs[0], __float2half(w * grid[index]));
					offsettedPositionInputs[1] = hadd(offsettedPositionInputs[1], __float2half(w * grid[index + 1]));
				}
				offsettedPositionInputs += 2;
			}
			__syncthreads();
			
									
			float *networkOutputs = sharedNetworkOutputs + warpId;
			uint8_t halfWarpId = warpId >> 1;
			half *inputsThread;
					
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint8_t offset = (halfWarpId & 15);
				inputsThread = sharedPositionInputs + offset * 32 + oddThread * 16; 
				half accumulator = 0;
				#pragma unroll
				for (uint8_t k = 0; k < 16; k++)
				{
					accumulator = hfma(inputsThread[k], weightsPositionL0Regs[k], accumulator);
				}
				accumulator = hadd(accumulator, __shfl_xor_sync(0xffffffff, accumulator, 1));
				if (oddThread)
				{
					intermidiateOutputsL0[offset * 64 + rowIndex] = hrelu(accumulator, __float2half(1.0f), sharedBiases[rowIndex]);
				}
				halfWarpId++;
			}
			__syncthreads();

			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
			{
				uint16_t offset = (halfWarpId & 15) * 64;
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
					intermidiateOutputsL1[offset + rowIndex] = hrelu(accumulator, __float2half(1.0f), __float2half(0.0f));
				}
				halfWarpId++;
			}
			__syncthreads();

			inputsThread = intermidiateOutputsL1 + L2InputsOffset;
			#pragma unroll
			for (uint8_t j = 0; j < 16; j++)
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
					networkOutputs[0] = __half2float(accumulator);
				}
				inputsThread += 64;
				networkOutputs += 4;
			}		
			__syncthreads();
						
			if (threadIdx.x < 3)
			{
				networkOutputs = sharedNetworkOutputs;

				for (uint16_t i = 0; i < 16; i++)
				{
					float color = 1.0f / (1.0f + expf(-networkOutputs[threadIdx.x]));
					float sigma = expf(networkOutputs[3]);
					float alpha = 1.0f - expf(-sampleDistributions * sigma);
					
					outputAccumulator += rem * color * alpha;
					rem *= 1.0f - alpha;
					networkOutputs += 4;
				}

				if (threadIdx.x == 0)
				{
					sharedRaysAndRem[3] = rem;
				}
			}
			__syncthreads();
			if (sharedRaysAndRem[3] < 0.0025f)
			{
				break;
			}
		}

		if (threadIdx.x < 3)
		{
			outputAccumulator += rem;
			offsettedRGBAOutputs[threadIdx.x] = static_cast<uint8_t>(min(max(0.0f, outputAccumulator * 255.0f), 255.0f));			
		}
		else if (threadIdx.x == 3)
		{
			offsettedRGBAOutputs[3] = 255;
		}

		if (threadIdx.x == 0)
		{
			sharedPixelNumbers[0] = atomicAdd(pixelCounter, 1U);
		}
		__syncthreads();
		pixelNumber = sharedPixelNumbers[0];
		if (pixelNumber >= numberOfPixels)
		{
			return;
		}
		offsettedRGBAOutputs = RGBAOutputs + pixelNumber * 4;
	}
}

void baselineLaunch(float *d_rays, Camera *d_camera, float *d_aabb, float *d_samples, float *d_sampleDistributions, 
					float *d_embeddings, uint32_t *d_offsets, float *d_networkInputs, float *d_networkOutputs, float *d_weightsL0, 
					float *d_weightsL1, float *d_weightsL2, uint8_t *d_outputs, uint16_t width, uint16_t height)
{
	#if __CUDA_ARCH__ >= 530
		constexpr uint16_t samplesPerIteration = 40'000;
	#else
		constexpr uint16_t samplesPerIteration = 2'500;
	#endif
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;
	//cout << "Iterations: " << iterations << endl;
	for (uint16_t i = 0; i < iterations; i++)
	{
		baselineRayGenerationKernel<<<(size + 255) / 256, 256>>>(d_rays, d_camera, size, size * i, width, height);
		HANDLE_ERROR(cudaDeviceSynchronize());

		baselineSampleGenerationKernel<<<size, 512>>>(d_rays, d_camera, d_aabb, d_samples, d_sampleDistributions);
		HANDLE_ERROR(cudaDeviceSynchronize());

		frequencyEncodingKernel<<<size, 512>>>(d_samples, d_networkInputs);
		HANDLE_ERROR(cudaDeviceSynchronize());

		positionEncoderKernel<<<size, 512>>>(d_samples, d_embeddings, d_offsets, d_networkInputs);
		HANDLE_ERROR(cudaDeviceSynchronize());

		baselineNetworkKernel<4><<<size * 512 >> 2, 256>>>(d_networkInputs, d_networkOutputs, d_weightsL0, d_weightsL1, d_weightsL2);
		HANDLE_ERROR(cudaDeviceSynchronize());

		baselineAccumulationKernel<<<(size + 255) / 256, 512>>>(d_networkOutputs, d_outputs + 4 * samplesPerIteration * i, 
																d_sampleDistributions, size);
		HANDLE_ERROR(cudaDeviceSynchronize());
		//cout << "Iteration " << i << " done" << endl;
	}
}

void allInOneLaunch(Camera *d_camera, float *d_aabb, float *d_sampleDistributions, float *d_embeddings, uint32_t *d_offsets, 
					float *d_networkInputs, float *d_networkOutputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
					uint8_t *d_outputs, uint16_t width, uint16_t height)
{
	#if __CUDA_ARCH__ >= 530
		constexpr uint16_t samplesPerIteration = 40'000;
	#else
		constexpr uint16_t samplesPerIteration = 2'500;
	#endif
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;
	allInOneKernel<<<size, 512>>>(d_camera, size, width, height, d_aabb, d_sampleDistributions, d_embeddings, d_offsets, 
								  d_networkInputs, d_weightsL0, d_weightsL1, d_weightsL2, d_outputs, iterations);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void frequencyEncodingCoopLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, float *d_networkInputs, 
						         float *d_networkOutputs, float *d_weightsFrequencyL0, float *d_weightsPositionL0, float *d_weightsL1, 
						         float *d_weightsL2, uint8_t *d_outputs, uint16_t width, uint16_t height)
{
	uint16_t samplesPerIteration = 5'000;
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;

	frequencyEncodingCoopKernel<<<size, 256>>>(d_camera, size, width, height, d_aabb, d_embeddings, d_offsets, d_weightsFrequencyL0, 
									           d_weightsPositionL0, d_weightsL1, d_weightsL2, d_outputs, iterations);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void halfPrecisionLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, float *d_weightsFrequencyL0, 
						 half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint16_t width, 
						 uint16_t height)
{
	uint16_t samplesPerIteration = 5'000;
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;

	halfPrecisionKernel<<<size, 256>>>(d_camera, size, width, height, d_aabb, d_embeddings, d_offsets, d_weightsFrequencyL0, 
									   d_weightsPositionL0, d_weightsL1, d_weightsL2, d_outputs, iterations);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void positionEncodingCoopLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, float *d_weightsFrequencyL0, 
						        half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint16_t width, 
						        uint16_t height)
{
	uint16_t samplesPerIteration = 5'000;
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;

	positionEncodingCoopKernel<<<size, 256>>>(d_camera, size, width, height, d_aabb, d_embeddings, d_offsets, d_weightsFrequencyL0, 
									          d_weightsPositionL0, d_weightsL1, d_weightsL2, d_outputs, iterations);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void moreCoopLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, half *d_weightsFrequencyL0, 
					half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint16_t width, 
					uint16_t height)
{
	uint16_t samplesPerIteration = 5'000;
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;

	moreCoopKernel<<<size, 256>>>(d_camera, size, width, height, d_aabb, d_embeddings, d_offsets, d_weightsFrequencyL0, 
								  d_weightsPositionL0, d_weightsL1, d_weightsL2, d_outputs, iterations);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void lessThreadsFixedIterationsLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, 
									  half *d_weightsFrequencyL0, half *d_weightsPositionL0, half *d_weightsL1, 
									  half *d_weightsL2, uint8_t *d_outputs, uint16_t width, uint16_t height)
{
	uint16_t samplesPerIteration = 5'000;
	uint16_t iterations = (width * height + samplesPerIteration - 1) / samplesPerIteration;
	uint32_t size = width * height > samplesPerIteration ? samplesPerIteration : width * height;

	lessThreadsFixedIterationsKernel<<<size, 128>>>(d_camera, size, width, height, d_aabb, d_embeddings, d_offsets, 
													d_weightsFrequencyL0, d_weightsPositionL0, d_weightsL1, d_weightsL2, 
													d_outputs, iterations);
	HANDLE_ERROR(cudaDeviceSynchronize());
}


void pixelPoolLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, half *d_weightsFrequencyL0, 
					 half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint32_t *d_pixelCounter, 
					 uint16_t width, uint16_t height)
{
	constexpr uint16_t pixelsPerIteration = 4'000;
	uint16_t iterations = (width * height + pixelsPerIteration - 1) / pixelsPerIteration;
	uint32_t size = width * height > pixelsPerIteration ? pixelsPerIteration : width * height;

	if (width == 800)
	{
		pixelPoolKernel<800, 800, 640'000, pixelsPerIteration><<<size, 128>>>(d_camera, d_pixelCounter, d_aabb, d_embeddings, d_offsets, 
																              d_weightsFrequencyL0, d_weightsPositionL0, d_weightsL1, 
																              d_weightsL2, d_outputs, iterations);
	}
	if (width == 100)
	{
		pixelPoolKernel<100, 100, 10'000, pixelsPerIteration><<<size, 128>>>(d_camera, d_pixelCounter, d_aabb, d_embeddings, d_offsets, 
																             d_weightsFrequencyL0, d_weightsPositionL0, d_weightsL1, 
																             d_weightsL2, d_outputs, iterations);
	}
	else if (width == 10)
	{
		pixelPoolKernel<10, 10, 100, 100><<<size, 128>>>(d_camera, d_pixelCounter, d_aabb, d_embeddings, d_offsets, 
														 d_weightsFrequencyL0, d_weightsPositionL0, d_weightsL1, 
														 d_weightsL2, d_outputs, iterations);
	}
	else if (width == 5)
	{
		pixelPoolKernel<5, 5, 25, 25><<<size, 128>>>(d_camera, d_pixelCounter, d_aabb, d_embeddings, d_offsets, 
													 d_weightsFrequencyL0, d_weightsPositionL0, d_weightsL1, 
													 d_weightsL2, d_outputs, iterations);
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
}

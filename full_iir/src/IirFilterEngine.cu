#include "IirFilterEngine.cuh"
#include "Utility.cuh"

using namespace std;
const uint8_t THREADS = 128;

__global__ void filterGenericDevicePointersKernel(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
									              uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, 
									              uint32_t *d_filtersOffsets, uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, 
									              uint32_t numberOfSignals)
{
	uint32_t signalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (signalIndex < numberOfSignals)
	{
		// get metadata for current signal
		uint32_t signalLength = d_signalLengths[signalIndex];
		uint32_t filtersCount = d_filtersCounts[signalIndex];
		uint32_t filterSizesOffset = d_filterSizesOffsets[signalIndex];

		// move to memory locations for current signal
		const float *d_inputs_1D = d_inputs[signalIndex];
		d_inputs_buffer = &d_inputs_buffer[d_signalOffsets[signalIndex]];
		float *d_outputs_1D = d_outputs[signalIndex];
		d_filterValues = &d_filterValues[d_filtersOffsets[signalIndex]];

		for (uint32_t i = 0; i < filtersCount; i++)
		{
			uint32_t filterLenght = d_filterSizes[filterSizesOffset + i]; // filter order + 1
			for (uint32_t j = 0; j < signalLength; j++)
			{
				float output = 0;
				for (uint32_t k = 1; k < filterLenght; k++) // As first
				{
					if (j >= k)
					{
						output -= d_outputs_1D[j - k] * d_filterValues[k]; // y[j-k] * a[k]
					}
				}
				for (uint32_t k = filterLenght; k < filterLenght << 1; k++) // Bs second
				{
					if (j + filterLenght >= k)
					{
						output += d_inputs_1D[j + filterLenght - k] * d_filterValues[k]; // x[j-k] * b[k]
					}
				}
				
				d_outputs_1D[j] = output / d_filterValues[0]; // y[j] = sum(b[k] * x[j-k]) - sum(a[k] * y[j-k]) / a[0]
				if (j >= filterLenght)
				{
					d_inputs_buffer[j - filterLenght] = d_outputs_1D[j - filterLenght]; // copy the outputs to the inputs with a delay of filterSize
				}
			}

			for (uint32_t j = signalLength - filterLenght; j < signalLength; j++)
			{
				d_inputs_buffer[j] = d_outputs_1D[j]; // copy the last filterSize outputs to the inputs
			}
			// next filter will use the outputs of the previous filter as inputs and the initial inputs won't be overwritten
			d_inputs_1D = d_inputs_buffer;
			d_filterValues = &d_filterValues[filterLenght << 1]; // move to next filter
		}
	}
}

template <uint16_t signalLength, uint16_t finiteFilterLength>
__global__ void finiteFilterBlockPerSignalKernel(const float **d_inputs, float **d_outputs, float *d_filterValues)
{
	__shared__ float sharedInputs[signalLength];
	__shared__ float sharedFilterValues[finiteFilterLength << 1];
	sharedInputs[threadIdx.x] = d_inputs[blockIdx.x][threadIdx.x];
	if (signalLength > finiteFilterLength << 1 && threadIdx.x < finiteFilterLength << 1)
	{
		sharedFilterValues[threadIdx.x] = d_filterValues[threadIdx.x];
	}
	else if (threadIdx.x < finiteFilterLength)
	{
		reinterpret_cast<float2 *>(sharedFilterValues)[threadIdx.x] = reinterpret_cast<const float2 *>(d_filterValues)[threadIdx.x];
	}
	__syncthreads();
		
	uint16_t filterOffset = (finiteFilterLength - threadIdx.x - 1) * (static_cast<int>(finiteFilterLength - threadIdx.x) > 0);
	uint32_t signalOffset = (threadIdx.x - finiteFilterLength + 1) * (threadIdx.x >= finiteFilterLength);
	
	float *inputs = sharedInputs + signalOffset;
	d_filterValues = sharedFilterValues + filterOffset;
	float result = 0.0f;

	#pragma unroll
	for (uint16_t i = 0; i < finiteFilterLength; i++)
	{
		result += inputs[i] * d_filterValues[i];
	}

	d_outputs[blockIdx.x][threadIdx.x] = result;
}

template <uint16_t signalLength, uint16_t finiteFilterLength, uint8_t blockShift>
__global__ void finiteFilterMoreBlocksPerSignalKernel(const float **d_inputs, float **d_outputs, float *d_filterValues)
{
	__shared__ float sharedInputs[signalLength];
	__shared__ float sharedFilterValues[finiteFilterLength << 1];
	uint32_t blockId = blockIdx.x >> blockShift;
	uint32_t threadId = (blockIdx.x & ~(~0U << blockShift)) * blockDim.x + threadIdx.x;

	if (blockShift == 0)
	{
		sharedInputs[threadIdx.x] = d_inputs[blockIdx.x][threadIdx.x];
	}
	else if (blockShift == 1)
	{
		reinterpret_cast<float2 *>(sharedInputs)[threadIdx.x] = reinterpret_cast<const float2 *>(d_inputs[blockId])[threadIdx.x];
	}
	else if (blockShift == 2)
	{
		reinterpret_cast<float4 *>(sharedInputs)[threadIdx.x] = reinterpret_cast<const float4 *>(d_inputs[blockId])[threadIdx.x];
	}
	else
	{
		uint32_t multipliedThreadId = threadIdx.x << (blockShift - 2);
		#pragma unroll
		for (uint8_t i = 0; i <= (blockShift - 2); i++)
		{
			reinterpret_cast<float4 *>(sharedInputs)[multipliedThreadId + i] = 
				reinterpret_cast<const float4 *>(d_inputs[blockId])[multipliedThreadId + i];
		}
	}

	if ((signalLength >> blockShift) > (finiteFilterLength << 1) && threadIdx.x < (finiteFilterLength << 1))
	{
		sharedFilterValues[threadIdx.x] = d_filterValues[threadIdx.x];
	}
	else if ((signalLength >> blockShift) > finiteFilterLength && threadIdx.x < finiteFilterLength)
	{
		reinterpret_cast<float2 *>(sharedFilterValues)[threadIdx.x] = reinterpret_cast<float2 *>(d_filterValues)[threadIdx.x];
	}
	else if (threadIdx.x < (finiteFilterLength >> 1))
	{
		reinterpret_cast<float4 *>(sharedFilterValues)[threadIdx.x] = reinterpret_cast<float4 *>(d_filterValues)[threadIdx.x];
	}
	__syncthreads();
	
	uint16_t filterOffset = (finiteFilterLength - threadId - 1) * (static_cast<int>(finiteFilterLength - threadId) > 0);
	uint32_t signalOffset = (threadId - finiteFilterLength + 1) * (threadId >= finiteFilterLength);
	
	float *inputs = sharedInputs + signalOffset;
	d_filterValues = sharedFilterValues + filterOffset;
	float result = 0.0f;

	#pragma unroll
	for (uint16_t i = 0; i < finiteFilterLength; i++)
	{
		result += inputs[i] * d_filterValues[i];
	}

	d_outputs[blockId][threadId] = result;
}

template <uint16_t signalLength, uint16_t finiteFilterLength, uint8_t blockShift, uint8_t threadShift>
__global__ void finiteFilterMoreBlocksAndThredsKernel(const float **d_inputs, float **d_outputs, float *d_filterValues)
{
	const uint8_t totalShift = blockShift + threadShift;
	__shared__ float sharedInputs[signalLength];
	__shared__ float sharedFilterValues[finiteFilterLength << 1];
	uint32_t blockId = blockIdx.x >> totalShift;
	uint32_t threadId = (blockIdx.x & ~(~0U << totalShift)) * blockDim.x + threadIdx.x;

	if (blockShift == 0)
	{
		sharedInputs[threadIdx.x] = d_inputs[blockIdx.x][threadIdx.x];
	}
	else if (blockShift == 1)
	{
		reinterpret_cast<float2 *>(sharedInputs)[threadIdx.x] = reinterpret_cast<const float2 *>(d_inputs[blockId])[threadIdx.x];
	}
	else if (blockShift == 2)
	{
		reinterpret_cast<float4 *>(sharedInputs)[threadIdx.x] = reinterpret_cast<const float4 *>(d_inputs[blockId])[threadIdx.x];
	}
	else
	{
		uint32_t multipliedThreadId = threadIdx.x << (blockShift - 2);
		#pragma unroll
		for (uint8_t i = 0; i <= (blockShift - 2); i++)
		{
			reinterpret_cast<float4 *>(sharedInputs)[multipliedThreadId + i] = 
				reinterpret_cast<const float4 *>(d_inputs[blockId])[multipliedThreadId + i];
		}
	}

	if ((signalLength >> blockShift) > (finiteFilterLength << 1) && threadIdx.x < (finiteFilterLength << 1))
	{
		sharedFilterValues[threadIdx.x] = d_filterValues[threadIdx.x];
	}
	else if ((signalLength >> blockShift) > finiteFilterLength && threadIdx.x < finiteFilterLength)
	{
		reinterpret_cast<float2 *>(sharedFilterValues)[threadIdx.x] = reinterpret_cast<float2 *>(d_filterValues)[threadIdx.x];
	}
	else if (threadIdx.x < (finiteFilterLength >> 1))
	{
		reinterpret_cast<float4 *>(sharedFilterValues)[threadIdx.x] = reinterpret_cast<float4 *>(d_filterValues)[threadIdx.x];
	}
	__syncthreads();
	
	uint8_t threadModuloId = threadId & ~(~0U << threadShift);
	threadId >>= threadShift;
	uint16_t filterOffset = (finiteFilterLength - threadId - 1) * (static_cast<int>(finiteFilterLength - threadId) > 0);
	uint32_t signalOffset = (threadId - finiteFilterLength + 1) * (threadId >= finiteFilterLength);
	filterOffset += threadModuloId * (finiteFilterLength >> threadShift);
	signalOffset += threadModuloId * (finiteFilterLength >> threadShift);
	
	float *inputs = sharedInputs + signalOffset;
	d_filterValues = sharedFilterValues + filterOffset;
	float result = 0.0f;

	#pragma unroll
	for (uint16_t i = 0; i < finiteFilterLength >> threadShift; i++)
	{
		result += inputs[i] * d_filterValues[i];
	}

	#pragma unroll
	for (uint8_t i = 1 << (threadShift - 1); i > 0; i >>= 1)
	{
		result += __shfl_xor_sync(0xffffffff, result, i);
	}

	if (threadModuloId == 0)
	{
		d_outputs[blockId][threadId] = result;
	}
}

template <uint8_t finiteFilterLength, uint8_t numberOfSignals>
__global__ void finiteFilterMoreSignalsKernel(const float **d_inputs, float **d_outputs, float *d_filterValues, uint32_t signalLength)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float accumulatros[numberOfSignals] = { 0.0f, };
	const float *inputs[numberOfSignals];
	
	if (threadId < signalLength)
	{
		uint16_t filterOffset = (finiteFilterLength - threadId - 1) * (static_cast<int>(finiteFilterLength - threadId) > 0);
		uint32_t signalOffset = (threadId - finiteFilterLength + 1) * (threadId >= finiteFilterLength);

		#pragma unroll
		for (uint8_t i = 0; i < numberOfSignals; i++)
		{
			inputs[i] = d_inputs[i] + signalOffset;
		}
		d_filterValues += filterOffset;
		
		#pragma unroll
		for (uint8_t i = 0; i < finiteFilterLength; i++)
		{
			float filterValue = d_filterValues[i];
			
			#pragma unroll
			for (uint8_t j = 0; j < numberOfSignals; j++)
			{
				accumulatros[j] += inputs[j][i] * filterValue;
			}
		}

		#pragma unroll
		for (uint8_t i = 0; i < numberOfSignals; i++)
		{
			d_outputs[i][threadId] = accumulatros[i];
		}
	}
}

template <uint8_t finiteFilterLength, uint8_t outputsPerThread, uint8_t threadIdShift, uint8_t threadsPerBlockShift>
__global__ void finiteInfiniteFilterLongSignalKernel(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_infiniteFilter)
{
	uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t signalIndex = globalId >> threadIdShift;
	uint32_t threadId = globalId & ~(~0U << threadIdShift);
	uint32_t multipliedThreadId = threadId * outputsPerThread;
	const uint8_t paddedFilterLenght = finiteFilterLength + (4 - (finiteFilterLength & 3)) * ((finiteFilterLength & 3) != 0);
	__shared__ float sharedInputs[paddedFilterLenght + (outputsPerThread << threadsPerBlockShift)];
	__shared__ float sharedFilterValues[finiteFilterLength];
	float infiniteFilter[6];
	reinterpret_cast<float4 *>(infiniteFilter)[0] = reinterpret_cast<float4 *>(d_infiniteFilter)[0];
	reinterpret_cast<float2 *>(infiniteFilter)[2] = reinterpret_cast<float2 *>(d_infiniteFilter)[2];

	if (threadIdx.x < finiteFilterLength)
	{
		sharedFilterValues[threadIdx.x] = d_finiteFilter[threadIdx.x];
	}

	const float *inputs = d_inputs[signalIndex] + (threadId & (~0U << threadsPerBlockShift)) * outputsPerThread - paddedFilterLenght;
	if (threadId < paddedFilterLenght)
	{
		sharedInputs[threadId] = 0.0f;
	}
	else if (threadIdx.x < paddedFilterLenght)
	{
		sharedInputs[threadIdx.x] = inputs[threadIdx.x];
	}

	inputs = d_inputs[signalIndex] + multipliedThreadId;
	float *offesetedSharedInputs = sharedInputs + paddedFilterLenght + threadIdx.x * outputsPerThread;
	#pragma unroll
	for (uint8_t i = 0; i < outputsPerThread >> 2; i++)
	{
		reinterpret_cast<float4 *>(offesetedSharedInputs)[i] = reinterpret_cast<const float4 *>(inputs)[i];
	}
	__syncthreads();

	offesetedSharedInputs = sharedInputs + threadIdx.x * outputsPerThread + 1 + (paddedFilterLenght - finiteFilterLength);
	float *outputs = d_outputs[signalIndex] + multipliedThreadId;
	
	float secondLastOutput = 0.0f;
	float lastOutput = 0.0f;
	float secondLastInput;
	float lastInput;

	#pragma unroll
	for (uint8_t j = 0; j < finiteFilterLength; j++)
	{
		secondLastOutput += offesetedSharedInputs[j] * sharedFilterValues[j];
	}
	secondLastInput = offesetedSharedInputs[finiteFilterLength - 1];
	offesetedSharedInputs++;
	outputs[0] = secondLastOutput;
	outputs++;
	#pragma unroll
	for (uint8_t j = 0; j < finiteFilterLength; j++)
	{
		lastOutput += offesetedSharedInputs[j] * sharedFilterValues[j];
	}
	lastInput = offesetedSharedInputs[finiteFilterLength - 1];
	offesetedSharedInputs += finiteFilterLength;
	outputs[0] = lastOutput;
	outputs++;

	#pragma unroll
	for (uint8_t i = 0; i < outputsPerThread - 2; i += 2)
	{
		float2 outputBatch;
		outputBatch.x = offesetedSharedInputs[0] * infiniteFilter[3] + lastInput * infiniteFilter[4] + secondLastInput * infiniteFilter[5] - 
						lastOutput * infiniteFilter[1] - secondLastOutput * infiniteFilter[2];
		outputBatch.y = offesetedSharedInputs[1] * infiniteFilter[3] + offesetedSharedInputs[0] * infiniteFilter[4] + lastInput * infiniteFilter[5] - 
						outputBatch.x * infiniteFilter[1] - lastOutput * infiniteFilter[2];
		reinterpret_cast<float2*>(outputs + i)[0] = outputBatch;
		
		secondLastOutput = outputBatch.x;
		lastOutput = outputBatch.y;
		secondLastInput = offesetedSharedInputs[0];
		lastInput = offesetedSharedInputs[1];
		offesetedSharedInputs += 2;
	}
}

template <uint8_t finiteFilterLength, uint8_t outputsPerThread, uint8_t signalsPerBlock, uint16_t signalLength, uint8_t signalIdShift>
__global__ void finiteInfiniteFilterOrder1Kernel(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_infiniteFilter)
{
	uint32_t globalSignalIdx = (blockIdx.x * blockDim.x + threadIdx.x) >> signalIdShift;
	uint8_t blockThreadIdx = threadIdx.x & ~((~0U) << signalIdShift);
	uint16_t signalOffset = blockThreadIdx * outputsPerThread;
	int16_t finiteFilterOffset = signalOffset - finiteFilterLength;
	finiteFilterOffset = blockThreadIdx == 0 ? signalLength - finiteFilterLength : finiteFilterOffset;
	const float *inputs = &d_inputs[globalSignalIdx][finiteFilterOffset];
	float *outputs = &d_outputs[globalSignalIdx][signalOffset];
	float lastOutput = 0;
	
	if (outputsPerThread >= finiteFilterLength)
	{
		#pragma unroll
		for (uint8_t i = 0; i < finiteFilterLength >> 2; i++)
		{
			float4 filter = reinterpret_cast<float4*>(d_finiteFilter)[i];
			float4 input = reinterpret_cast<const float4*>(inputs)[i];
			lastOutput += filter.x * input.x;
			lastOutput += filter.y * input.y;
			lastOutput += filter.z * input.z;
			lastOutput += filter.w * input.w;
		}
	}
	else
	{
		#pragma unroll
		for (uint8_t i = (finiteFilterLength - outputsPerThread) >> 2; i < finiteFilterLength >> 2; i++)
		{
			float4 filter = reinterpret_cast<float4*>(d_finiteFilter)[i];
			float4 input = reinterpret_cast<const float4*>(inputs)[i];
			lastOutput += filter.x * input.x;
			lastOutput += filter.y * input.y;
			lastOutput += filter.z * input.z;
			lastOutput += filter.w * input.w;
		}

		if (finiteFilterOffset >= 0)
		{
			#pragma unroll
			for (int8_t i = 0; i < (finiteFilterLength - outputsPerThread) >> 2; i++)
			{
				float4 filter = reinterpret_cast<float4*>(d_finiteFilter)[i];
				float4 input = reinterpret_cast<const float4*>(inputs)[i];
				lastOutput += filter.x * input.x;
				lastOutput += filter.y * input.y;
				lastOutput += filter.z * input.z;
				lastOutput += filter.w * input.w;
			}
		}
	}

	inputs = &d_inputs[globalSignalIdx][signalOffset];

	float4 infiniteFilter = reinterpret_cast<float4*>(d_infiniteFilter)[0];
	float4 inputBatch = reinterpret_cast<const float4*>(inputs)[0];
	float4 outputBatch;
	
	outputBatch.x = blockThreadIdx == 0 ? inputBatch.x * infiniteFilter.y : 
									   inputBatch.x * infiniteFilter.y - lastOutput * infiniteFilter.z + (inputs - 1)[0] * infiniteFilter.w;

	outputBatch.y = inputBatch.y * infiniteFilter.y - outputBatch.x * infiniteFilter.z + inputBatch.x * infiniteFilter.w;
	outputBatch.z = inputBatch.z * infiniteFilter.y - outputBatch.y * infiniteFilter.z + inputBatch.y * infiniteFilter.w;
	outputBatch.w = inputBatch.w * infiniteFilter.y - outputBatch.z * infiniteFilter.z + inputBatch.z * infiniteFilter.w;

	reinterpret_cast<float4*>(outputs)[0] = outputBatch;

	float lastInput = inputBatch.w;
	lastOutput = outputBatch.w;

	#pragma unroll
	for (uint8_t i = 1; i < outputsPerThread >> 2; i++)
	{
		inputBatch = reinterpret_cast<const float4*>(inputs)[i];
		outputBatch.x = inputBatch.x * infiniteFilter.y - lastOutput * infiniteFilter.z + lastInput * infiniteFilter.w;
		outputBatch.y = inputBatch.y * infiniteFilter.y - outputBatch.x * infiniteFilter.z + inputBatch.x * infiniteFilter.w;
		outputBatch.z = inputBatch.z * infiniteFilter.y - outputBatch.y * infiniteFilter.z + inputBatch.y * infiniteFilter.w;
		outputBatch.w = inputBatch.w * infiniteFilter.y - outputBatch.z * infiniteFilter.z + inputBatch.z * infiniteFilter.w;
		reinterpret_cast<float4*>(outputs)[i] = outputBatch;
		lastInput = inputBatch.w;
		lastOutput = outputBatch.w;
	}
}

__global__ void filterStateSpaceParallelOrder2Kernel(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_stateSpace)
{
	const uint8_t threadsPerWarp = 32;
	uint16_t warpId = threadIdx.x >> 5;
	uint16_t warpOffset = (warpId << 5);
	uint16_t threadWarpId = threadIdx.x & (threadsPerWarp - 1);
	uint16_t stateSpaceOffset = 512 - warpId * threadsPerWarp;
	float *stateSpaceOffseted = &d_stateSpace[stateSpaceOffset];
	const float *inputs = d_inputs[blockIdx.x];
	float *outputs = d_outputs[blockIdx.x];
	float state1;
	float state2;

	if (warpId == 0)
	{
		state1 = 0;
		state2 = 0;
	}
	else
	{
		if (threadWarpId == 0)
		{
			state1 = 0;
			state2 = stateSpaceOffseted[0] * inputs[0];
		}
		else
		{
			state1 = stateSpaceOffseted[threadWarpId] * inputs[threadWarpId - 1];
			state2 = stateSpaceOffseted[threadWarpId] * inputs[threadWarpId];
		}

		for (uint16_t i = threadsPerWarp; i < warpOffset; i += threadsPerWarp)
		{
			state1 += stateSpaceOffseted[i + threadWarpId] * inputs[i + threadWarpId - 1];
			state2 += stateSpaceOffseted[i + threadWarpId] * inputs[i + threadWarpId];
		}

		for (uint8_t i = 16; i > 0; i >>= 1)
		{
			state1 += __shfl_xor_sync(0xffffffff, state1, i, 32);
			state2 += __shfl_xor_sync(0xffffffff, state2, i, 32);
		}
	}

	uint16_t filterOffset = (threadsPerWarp - threadWarpId - 1) * (static_cast<int>(threadsPerWarp - threadWarpId) > 0);
	
	d_finiteFilter = &d_finiteFilter[filterOffset];
	float result = d_finiteFilter[0] * state2;
	d_finiteFilter++;
	if (threadWarpId != 0)
	{
		result -= d_finiteFilter[0] * state1;
	}
	inputs += warpOffset;

	for (uint16_t i = 0; i < threadsPerWarp; i++)
	{
		result += d_finiteFilter[i] * inputs[i];
	}

	outputs[warpOffset + threadWarpId] = result;
}

template <uint16_t signalLength, uint8_t signalsPerBlock, uint8_t signalsPerStateSpaceMatrix, bool useLookup>
__global__ void filterStateSpaceMatrixOrder2Kernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix,
                                                   uint32_t *d_signalIndices)
{
	__shared__ float sharedInputs[signalsPerBlock][signalLength];
	const uint8_t threadsPerWarp = 32;
	uint32_t signalIndex = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t threadWarpId = threadIdx.x & 31;
	uint16_t threadMultipliedId = threadWarpId * (signalLength >> 5);

	if (useLookup)
	{
		if (signalIndex >= signalsPerStateSpaceMatrix)
		{
			d_stateSpaceMatrix += 32 * 32 * 2;
		}
		signalIndex = d_signalIndices[signalIndex];
	}

	const float *inputs = d_inputs[signalIndex]; 
	float *outputs = d_outputs[signalIndex];
	float filter[threadsPerWarp];
	float state1;
	float state2;
	float result = 0;

	d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
	#pragma unroll
	for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
	{
		reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
	}

	float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
	const float *inputsOffseted = inputs + threadMultipliedId;
	if (signalLength >= 128)
	{
		#pragma unroll
		for (int8_t i = 0; i < signalLength >> 7; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
	}
	else
	{
		#pragma unroll
		for (int8_t i = 0; i < signalLength >> 6; i++)
		{
			reinterpret_cast<float2*>(sharedInputsOffseted)[i] = reinterpret_cast<const float2*>(inputsOffseted)[i];
		}
	}
	__syncwarp();
	sharedInputsOffseted = sharedInputs[warpId];

	#pragma unroll
	for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
	{
		float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
		float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
		result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
	}
	state1 =__shfl_sync(0xffffffff, result, 30);
	state2 =__shfl_sync(0xffffffff, result, 31);
	if (threadWarpId < threadsPerWarp - 2)
	{
		outputs[threadWarpId] = result;
	}
	
	#pragma unroll
	for (uint8_t i = 0; i < (signalLength / 30) - 1; i++)
	{
		sharedInputsOffseted += threadsPerWarp - 2;
		outputs += threadsPerWarp - 2;
		result = 0;

		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		result += filter[threadsPerWarp - 2] * state1;
		result += filter[threadsPerWarp - 1] * state2;
		
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}
	}

	if (threadWarpId < signalLength - (signalLength / 30) * 30)
	{
		sharedInputsOffseted += threadsPerWarp - 2;
		outputs += threadsPerWarp - 2;
		result = 0;

		#pragma unroll
		for (uint8_t i = 0; i < signalLength - (signalLength / 30) * 30; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		result += filter[threadsPerWarp - 2] * state1;
		result += filter[threadsPerWarp - 1] * state2;
		outputs[threadWarpId] = result;
	}
}

template <uint16_t signalLength, uint8_t signalsPerBlock, uint8_t warpsPerSignal, uint8_t stateSpaceLength>
__global__ void filterStateSpaceMatrixOrder1ParallelKernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix)
{
	__shared__ float sharedInputs[signalsPerBlock][signalLength];
	const uint8_t threadsPerWarp = 32;
	const uint16_t signalLengthPerWarp = signalLength / warpsPerSignal;
	uint32_t signalIndex = (blockIdx.x * blockDim.x + threadIdx.x) / (warpsPerSignal * threadsPerWarp);
	uint8_t warpId = threadIdx.x / (warpsPerSignal * threadsPerWarp);
	uint8_t threadWarpId = threadIdx.x & (threadsPerWarp - 1);
	uint8_t threadId = threadIdx.x & (warpsPerSignal * threadsPerWarp - 1);
	uint16_t threadMultipliedId = threadId * signalLength / (warpsPerSignal * threadsPerWarp);

	const float *inputs = d_inputs[signalIndex]; 
	float *outputs = d_outputs[signalIndex] + signalLengthPerWarp * (threadId >> 5);
	float filter[stateSpaceLength];
	float state1 = 0;
	float result = 0;

	d_stateSpaceMatrix += (stateSpaceLength + ((stateSpaceLength & 3) ? 4 - (stateSpaceLength & 3) : 0)) * threadWarpId;
	#pragma unroll
	for (uint8_t i = 0; i < stateSpaceLength >> 2; i++)
	{
		reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
	}
	#pragma unroll
	for (uint8_t i = (stateSpaceLength >> 2) << 2; i < stateSpaceLength; i++)
	{
		filter[i] = d_stateSpaceMatrix[i];
	}

	float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
	const float *inputsOffseted = inputs + threadMultipliedId;
	if (signalLength >= 512)
	{
		#pragma unroll
		for (int8_t i = 0; i < signalLength / (warpsPerSignal * threadsPerWarp * 4); i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
	}
	else
	{
		return;
	}
	__syncthreads();
	sharedInputsOffseted = sharedInputs[warpId] + 
						   (threadWarpId > stateSpaceLength - 2 ? threadWarpId - stateSpaceLength + 2 - (threadWarpId == 31) : 0) +
						   signalLengthPerWarp * (threadId >> 5);

	if (threadWarpId == 31 && threadId > 31)
	{
		sharedInputsOffseted -= threadsPerWarp - 1;
		#pragma unroll
		for (uint8_t i = 0; i < stateSpaceLength - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		sharedInputsOffseted += threadsPerWarp - 1;
	}

	state1 =__shfl_sync(0xffffffff, result, 31);
	result = 0;
		
	#pragma unroll
	for (uint8_t i = 0; i < (signalLengthPerWarp / 31); i++)
	{
		#pragma unroll
		for (uint8_t i = 0; i < stateSpaceLength - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		result += filter[stateSpaceLength - 1] * state1;
		
		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}

		sharedInputsOffseted += threadsPerWarp - 1;
		outputs += threadsPerWarp - 1;
		result = 0;
	}

	if (threadWarpId < signalLengthPerWarp - (signalLengthPerWarp / 31) * 31)
	{
		#pragma unroll
		for (uint8_t i = 0; i < signalLengthPerWarp - (signalLengthPerWarp / 31) * 31 && i < stateSpaceLength - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		result += filter[stateSpaceLength - 1] * state1;
		outputs[threadWarpId] = result;
	}
}

template <uint16_t signalLengthPerWarp, uint8_t signalsPerBlock, uint8_t stateSpaceLength>
__global__ void filterStateSpaceMatrixOrder1LongSignalKernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix)
{
	const uint8_t threadsPerWarp = 32;
	__shared__ float sharedInputs[signalsPerBlock][signalLengthPerWarp + threadsPerWarp];
	uint8_t warpId = threadIdx.x / threadsPerWarp;
	uint8_t threadWarpId = threadIdx.x & (threadsPerWarp - 1);
	uint16_t threadMultipliedId = threadWarpId * signalLengthPerWarp / threadsPerWarp;
	const float *inputs = d_inputs[warpId] + signalLengthPerWarp * blockIdx.x; 
	float *outputs = d_outputs[warpId] + signalLengthPerWarp * blockIdx.x;
	float filter[stateSpaceLength];
	float state1 = 0;
	float result = 0;

	d_stateSpaceMatrix += (stateSpaceLength + ((stateSpaceLength & 3) ? 4 - (stateSpaceLength & 3) : 0)) * threadWarpId;
	#pragma unroll
	for (uint8_t i = 0; i < stateSpaceLength >> 2; i++)
	{
		reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
	}
	#pragma unroll
	for (uint8_t i = (stateSpaceLength >> 2) << 2; i < stateSpaceLength; i++)
	{
		filter[i] = d_stateSpaceMatrix[i];
	}

	float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
	const float *inputsOffseted = inputs + threadMultipliedId;
	if (signalLengthPerWarp >= 128)
	{
		#pragma unroll
		for (int8_t i = 0; i < signalLengthPerWarp >> 7; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		inputsOffseted = inputs - threadsPerWarp * (blockIdx.x > 0);
		sharedInputsOffseted = sharedInputs[warpId] + signalLengthPerWarp;
		sharedInputsOffseted[threadWarpId] = inputsOffseted[threadWarpId];
	}
	else
	{
		return;
	}
	__syncwarp();

	if (threadWarpId == 31 && blockIdx.x >= 1)
	{
		sharedInputsOffseted += threadsPerWarp - stateSpaceLength + 1;
		#pragma unroll
		for (uint8_t i = 0; i < stateSpaceLength - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
	}
	state1 =__shfl_sync(0xffffffff, result, 31);
	result = 0;

	sharedInputsOffseted = sharedInputs[warpId] + 
						   (threadWarpId > stateSpaceLength - 2 ? threadWarpId - stateSpaceLength + 2 - (threadWarpId == 31) : 0);
		
	#pragma unroll
	for (uint8_t i = 0; i < (signalLengthPerWarp / 31); i++)
	{
		#pragma unroll
		for (uint8_t i = 0; i < stateSpaceLength - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		result += filter[stateSpaceLength - 1] * state1;
		
		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}

		sharedInputsOffseted += threadsPerWarp - 1;
		outputs += threadsPerWarp - 1;
		result = 0;
	}

	if (threadWarpId < signalLengthPerWarp - (signalLengthPerWarp / 31) * 31)
	{
		#pragma unroll
		for (uint8_t i = 0; i < signalLengthPerWarp - (signalLengthPerWarp / 31) * 31 && i < stateSpaceLength - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		result += filter[stateSpaceLength - 1] * state1;
		outputs[threadWarpId] = result;
	}
}

__global__ void filterStateSpaceMatrixSignalLengthsKernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, 
												          uint32_t *d_signalIndices)
{
	__shared__ float sharedInputs[8][1024];
	const uint8_t threadsPerWarp = 32;
	uint32_t signalId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	uint32_t signalIndex = d_signalIndices[signalId];
	uint8_t warpId = threadIdx.x >> 5;
	uint8_t threadWarpId = threadIdx.x & 31;

	const float *inputs = d_inputs[signalIndex]; 
	float *outputs = d_outputs[signalIndex];
	float filter[threadsPerWarp];
	float state1;
	float state2;
	float result = 0;

	if (signalId >= 1792)
	{
		uint16_t threadMultipliedId = threadWarpId * 32;
		if (signalId >= 1896)
		{
			d_stateSpaceMatrix += 32 * 32 * 2; 
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		#pragma unroll
		for (int8_t i = 0; i < 8; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 33; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
			{
				float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
				float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
				result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			
			state1 =__shfl_sync(0xffffffff, result, 30);
			state2 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 2)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 4)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 1584)
	{
		uint16_t threadMultipliedId = threadWarpId * 32;
		if (signalId >= 1688)
		{
			d_stateSpaceMatrix += 32 * 32 * 3;
		}
		else
		{
			d_stateSpaceMatrix += 32 * 32 * 1;
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		#pragma unroll
		for (int8_t i = 0; i < 8; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}

		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 32; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			
			state1 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 1)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 1)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;
			result += filter[0] * sharedInputsOffseted[0];
			result += filter[threadsPerWarp - 1] * state1;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 1400)
	{
		uint16_t threadMultipliedId = threadWarpId * 16;
		if (signalId >= 1492)
		{
			d_stateSpaceMatrix += 32 * 32 * 2; 
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		#pragma unroll
		for (int8_t i = 0; i < 4; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 16; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
			{
				float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
				float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
				result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			
			state1 =__shfl_sync(0xffffffff, result, 30);
			state2 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 2)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 2)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 2; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 1216)
	{
		uint16_t threadMultipliedId = threadWarpId * 16;
		if (signalId >= 1308)
		{
			d_stateSpaceMatrix += 32 * 32 * 3;
		}
		else
		{
			d_stateSpaceMatrix += 32 * 32 * 1;
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		#pragma unroll
		for (int8_t i = 0; i < 4; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}

		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 15; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			
			state1 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 1)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 16)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 16; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 1016)
	{
		uint16_t threadMultipliedId = threadWarpId * 8;
		if (signalId >= 1116)
		{
			d_stateSpaceMatrix += 32 * 32 * 2; 
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		#pragma unroll
		for (int8_t i = 0; i < 2; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 7; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
			{
				float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
				float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
				result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			
			state1 =__shfl_sync(0xffffffff, result, 30);
			state2 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 2)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 16)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 16; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 816)
	{
		uint16_t threadMultipliedId = threadWarpId * 8;
		if (signalId >= 916)
		{
			d_stateSpaceMatrix += 32 * 32 * 3;
		}
		else
		{
			d_stateSpaceMatrix += 32 * 32 * 1;
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		#pragma unroll
		for (int8_t i = 0; i < 2; i++)
		{
			reinterpret_cast<float4*>(sharedInputsOffseted)[i] = reinterpret_cast<const float4*>(inputsOffseted)[i];
		}
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}

		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 7; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			
			state1 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 1)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 8)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 8; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 608)
	{
		uint16_t threadMultipliedId = threadWarpId * 4;
		if (signalId >= 712)
		{
			d_stateSpaceMatrix += 32 * 32 * 2; 
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		reinterpret_cast<float4*>(sharedInputsOffseted)[0] = reinterpret_cast<const float4*>(inputsOffseted)[0];
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 3; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
			{
				float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
				float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
				result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			
			state1 =__shfl_sync(0xffffffff, result, 30);
			state2 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 2)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 8)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 8; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 400)
	{
		uint16_t threadMultipliedId = threadWarpId * 4;
		if (signalId >= 504)
		{
			d_stateSpaceMatrix += 32 * 32 * 3;
		}
		else
		{
			d_stateSpaceMatrix += 32 * 32 * 1;
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		reinterpret_cast<float4*>(sharedInputsOffseted)[0] = reinterpret_cast<const float4*>(inputsOffseted)[0];
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}

		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}
		
		#pragma unroll
		for (uint8_t i = 0; i < 3; i++)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			
			state1 =__shfl_sync(0xffffffff, result, 31);
			if (threadWarpId < threadsPerWarp - 1)
			{
				outputs[threadWarpId] = result;
			}
		}

		if (threadWarpId < 4)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			outputs[threadWarpId] = result;
		}
	}
	else if (signalId >= 200)
	{
		uint16_t threadMultipliedId = threadWarpId * 2;
		if (signalId >= 300)
		{
			d_stateSpaceMatrix += 32 * 32 * 2; 
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		reinterpret_cast<float2*>(sharedInputsOffseted)[0] = reinterpret_cast<const float2*>(inputsOffseted)[0];
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}
		
		sharedInputsOffseted += threadsPerWarp - 2;
		outputs += threadsPerWarp - 2;
		result = 0;
		#pragma unroll
		for (uint8_t i = 0; i < (threadsPerWarp - 2) >> 1; i++)
		{
			float2 inputBatch = reinterpret_cast<float2*>(sharedInputsOffseted)[i];
			float2 filterBatch = reinterpret_cast<float2*>(filter)[i];
			result += filterBatch.x * inputBatch.x + filterBatch.y * inputBatch.y;
		}
		result += filter[threadsPerWarp - 2] * state1;
		result += filter[threadsPerWarp - 1] * state2;
		
		state1 =__shfl_sync(0xffffffff, result, 30);
		state2 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 2)
		{
			outputs[threadWarpId] = result;
		}

		if (threadWarpId < 4)
		{
			sharedInputsOffseted += threadsPerWarp - 2;
			outputs += threadsPerWarp - 2;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 4; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 2] * state1;
			result += filter[threadsPerWarp - 1] * state2;
			outputs[threadWarpId] = result;
		}
	}
	else
	{
		uint16_t threadMultipliedId = threadWarpId * 2;
		if (signalId >= 100)
		{
			d_stateSpaceMatrix += 32 * 32 * 3;
		}
		else
		{
			d_stateSpaceMatrix += 32 * 32 * 1;
		}

		d_stateSpaceMatrix += threadsPerWarp * threadWarpId;
		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp >> 2; i++)
		{
			reinterpret_cast<float4*>(filter)[i] = reinterpret_cast<float4*>(d_stateSpaceMatrix)[i];
		}

		float *sharedInputsOffseted = sharedInputs[warpId] + threadMultipliedId;
		const float *inputsOffseted = inputs + threadMultipliedId;
		reinterpret_cast<float2*>(sharedInputsOffseted)[0] = reinterpret_cast<const float2*>(inputsOffseted)[0];
		__syncwarp();
		sharedInputsOffseted = sharedInputs[warpId];

		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}

		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}
		
		sharedInputsOffseted += threadsPerWarp - 1;
		outputs += threadsPerWarp - 1;
		result = 0;

		#pragma unroll
		for (uint8_t i = 0; i < threadsPerWarp - 1; i++)
		{
			result += filter[i] * sharedInputsOffseted[i];
		}
		result += filter[threadsPerWarp - 1] * state1;
		
		state1 =__shfl_sync(0xffffffff, result, 31);
		if (threadWarpId < threadsPerWarp - 1)
		{
			outputs[threadWarpId] = result;
		}

		if (threadWarpId < 2)
		{
			sharedInputsOffseted += threadsPerWarp - 1;
			outputs += threadsPerWarp - 1;
			result = 0;

			#pragma unroll
			for (uint8_t i = 0; i < 2; i++)
			{
				result += filter[i] * sharedInputsOffseted[i];
			}
			result += filter[threadsPerWarp - 1] * state1;
			outputs[threadWarpId] = result;
		}
	}
}

void allocate2DArrayOnDevice(float ***d_signals, float **h_signalPtrs, int numSignals, int signalLength) 
{
	HANDLE_ERROR(cudaMalloc((void***)d_signals, numSignals * sizeof(float *)));

    for (int i = 0; i < numSignals; ++i)
	{
        HANDLE_ERROR(cudaMalloc((void**)&h_signalPtrs[i], signalLength * sizeof(float)));
    }

    HANDLE_ERROR(cudaMemcpy(*d_signals, h_signalPtrs, numSignals * sizeof(float*), cudaMemcpyHostToDevice));
}

void copy2DArrayToDevice(float** h_signalPtrs, const float **signals, int numSignals, int signalLength)
{
    for (int i = 0; i < numSignals; ++i)
	{
        HANDLE_ERROR(cudaMemcpy(h_signalPtrs[i], signals[i], signalLength * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void copy2DArrayToHost(float *&signals, float **h_signalPtrs, int signalIndex, int signalLength)
{
    HANDLE_ERROR(cudaMemcpy(signals, h_signalPtrs[signalIndex], signalLength * sizeof(float), cudaMemcpyDeviceToHost));
    
}

void free2DArrayOnDeviceMemory(float **d_signals, float **h_signalPtrs, int numSignals)
{
    for (int i = 0; i < numSignals; ++i)
	{
        HANDLE_ERROR(cudaFree(h_signalPtrs[i]));
    }
    HANDLE_ERROR(cudaFree(d_signals));
}

void allocateMemoryOnDevice(float *&d_memory, size_t size)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_memory, size));
}

void allocateFiniteFiltersOnDevice(vector<vector<float>> finiteFilters, float *&d_finiteFilters, bool reverseFilter)
{
	uint32_t totalSize = 0;
	for (vector<float> &filter : finiteFilters)
	{
		totalSize += filter.size() << 1;
	}
	HANDLE_ERROR(cudaMalloc((void**)&d_finiteFilters, totalSize * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_finiteFilters, 0, totalSize * sizeof(float)));

	uint32_t offset = 0;
	for (vector<float> &filter : finiteFilters)
	{	
		if (reverseFilter)
		{
			reverse(filter.begin(), filter.end());
		}
		HANDLE_ERROR(cudaMemcpy(&d_finiteFilters[offset], filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice));
		offset += filter.size() << 1;
	}
}

void allocateStateSpaceOnDevice(vector<vector<float>> stateSpace, float *&d_stateSpace)
{
	uint32_t totalSize = 0;
	for (vector<float> &state : stateSpace)
	{
		totalSize += state.size();
	}
	HANDLE_ERROR(cudaMalloc((void**)&d_stateSpace, totalSize * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_stateSpace, 0, totalSize * sizeof(float)));

	uint32_t offset = 0;
	for (vector<float> &state : stateSpace)
	{
		HANDLE_ERROR(cudaMemcpy(&d_stateSpace[offset], state.data(), state.size() * sizeof(float), cudaMemcpyHostToDevice));
		offset += state.size();
	}
}

void allocateSignalIndicesOnDevice(std::vector<std::vector<uint32_t>> signalLengthGroupsOrder1, std::vector<std::vector<uint32_t>> signalLengthGroupsOrder2, 
								   uint32_t *&d_signalIndices)
{
	uint32_t totalSize = 0;
	for (vector<uint32_t> &signalLengthGroup : signalLengthGroupsOrder1)
	{
		totalSize += signalLengthGroup.size();
	}
	for (vector<uint32_t> &signalLengthGroup : signalLengthGroupsOrder2)
	{
		totalSize += signalLengthGroup.size();
	}
	HANDLE_ERROR(cudaMalloc((void**)&d_signalIndices, totalSize * sizeof(uint32_t)));

	uint32_t offset = 0;
	for (uint8_t i = 0, j = 32; i < signalLengthGroupsOrder2.size() >> 1; i++, j++)
	{
		if (signalLengthGroupsOrder1[i].size() != 0)
		{
			HANDLE_ERROR(cudaMemcpy(&d_signalIndices[offset], signalLengthGroupsOrder1[i].data(), signalLengthGroupsOrder1[i].size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
			offset += signalLengthGroupsOrder1[i].size();
		}

		if (signalLengthGroupsOrder1[j].size() != 0)
		{
			HANDLE_ERROR(cudaMemcpy(&d_signalIndices[offset], signalLengthGroupsOrder1[j].data(), signalLengthGroupsOrder1[j].size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
			offset += signalLengthGroupsOrder1[j].size();
		}

		if (signalLengthGroupsOrder2[i].size() != 0)
		{
			HANDLE_ERROR(cudaMemcpy(&d_signalIndices[offset], signalLengthGroupsOrder2[i].data(), signalLengthGroupsOrder2[i].size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
			offset += signalLengthGroupsOrder2[i].size();
		}

		if (signalLengthGroupsOrder2[j].size() != 0)
		{
			HANDLE_ERROR(cudaMemcpy(&d_signalIndices[offset], signalLengthGroupsOrder2[j].data(), signalLengthGroupsOrder2[j].size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
			offset += signalLengthGroupsOrder2[j].size();
		}
	}
}

void allocateStateSpaceMatricesOnDevice(vector<vector<float>> stateSpaceMatrices, float *&d_stateSpaceMatrices)
{
	uint32_t totalSize = 0;
	for (vector<float> &matrix : stateSpaceMatrices)
	{
		totalSize += matrix.size();
	}
	HANDLE_ERROR(cudaMalloc((void**)&d_stateSpaceMatrices, totalSize * sizeof(float)));
	
	uint32_t offset = 0;
	for (vector<float> &matrix : stateSpaceMatrices)
	{
		HANDLE_ERROR(cudaMemcpy(&d_stateSpaceMatrices[offset], matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
		offset += matrix.size();
	}
}

void allocateInfiniteFiltersOnDevice(float *filterValues, uint32_t length, float *&d_filterValues)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_filterValues, length * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_filterValues, filterValues, length * sizeof(float), cudaMemcpyHostToDevice));
}

void allocateInfiniteFiltersOnDevice(vector<float> filterValues, float *&d_filterValues, uint32_t length)
{
	length = length == 0 ? filterValues.size() : length;
	HANDLE_ERROR(cudaMalloc((void**)&d_filterValues, length * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_filterValues, filterValues.data(), length * sizeof(float), cudaMemcpyHostToDevice));
}

void allocateMetadataOnDevice(vector<uint32_t> signalLengths, vector<uint32_t> signalOffsets, vector<uint32_t> filtersCounts, 
							  vector<uint32_t> filtersOffsets, vector<uint32_t> filterSizes, vector<uint32_t> filterSizesOffsets, 
							  uint32_t *&d_signalLengths, uint32_t *&d_signalOffsets, uint32_t *&d_filtersCounts, uint32_t *&d_filtersOffsets, 
							  uint32_t *&d_filterSizes, uint32_t *&d_filterSizesOffsets)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_signalLengths, signalLengths.size() * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_signalLengths, signalLengths.data(), signalLengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_signalOffsets, signalOffsets.size() * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_signalOffsets, signalOffsets.data(), signalOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_filtersCounts, filtersCounts.size() * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_filtersCounts, filtersCounts.data(), filtersCounts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_filtersOffsets, filtersOffsets.size() * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_filtersOffsets, filtersOffsets.data(), filtersOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_filterSizes, filterSizes.size() * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_filterSizes, filterSizes.data(), filterSizes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_filterSizesOffsets, filterSizesOffsets.size() * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_filterSizesOffsets, filterSizesOffsets.data(), filterSizesOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
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

void kernelGenericLaunchWithDevicePointers(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
										   uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, uint32_t *d_filtersOffsets, 
										   uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, uint32_t numberOfSignals)
{
	dim3 blockSize(THREADS, 1, 1);
	dim3 gridSize((numberOfSignals + THREADS - 1) / THREADS, 1, 1);
	filterGenericDevicePointersKernel<<<gridSize, blockSize>>>(d_inputs, d_inputs_buffer, d_outputs, d_filterValues, d_signalLengths, 
															   d_signalOffsets, d_filtersCounts, d_filtersOffsets, d_filterSizes, 
															   d_filterSizesOffsets, numberOfSignals);
	
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void kernelFiniteInfiniteFilterLongSignalLaunch(const float **d_inputs, float **d_outputs, float *d_finiteFilter, uint32_t filterLength, 
								        float *d_infiniteFilter)
{
	if (filterLength == 8)
	{
		finiteInfiniteFilterLongSignalKernel<8, 8, 17, 8><<<4096, 256>>>(d_inputs, d_outputs, d_finiteFilter, d_infiniteFilter);
	}
	else if (filterLength == 58)
	{
		finiteInfiniteFilterLongSignalKernel<58, 8, 17, 8><<<4096, 256>>>(d_inputs, d_outputs, d_finiteFilter, d_infiniteFilter);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void kernelStateSpaceParallelOrder2Launch(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_stateSpace, uint16_t numberOfSignals)
{
	filterStateSpaceParallelOrder2Kernel<<<numberOfSignals, 512>>>(d_inputs, d_outputs, d_finiteFilter, d_stateSpace);
}

void kernelStateSpaceMatrixOrder2Launch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, uint16_t numberOfSignals)
{
	filterStateSpaceMatrixOrder2Kernel<512, 8, 255, false><<<numberOfSignals >> 3, 256>>>(d_inputs, d_outputs, d_stateSpaceMatrix, nullptr);
}

void kernelStateSpaceMatrixOrder1ParallelLaunch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, uint16_t numberOfSignals)
{
	if (numberOfSignals == 1'000)
	{
		filterStateSpaceMatrixOrder1ParallelKernel<512, 2, 4, 9><<<numberOfSignals >> 1, 256>>>(d_inputs, d_outputs, d_stateSpaceMatrix);	
	}
	else if (numberOfSignals == 10'000)
	{
		filterStateSpaceMatrixOrder1ParallelKernel<512, 4, 2, 12><<<numberOfSignals >> 2, 256>>>(d_inputs, d_outputs, d_stateSpaceMatrix);
	}
	else
	{
		printf("Error: kernelStateSpaceMatrixOrder1ParallelLaunch: numberOfSignals = %d\n", numberOfSignals);
	}
}

void kernelStateSpaceMatrixOrder1LongSignalLaunch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix)
{
	filterStateSpaceMatrixOrder1LongSignalKernel<128, 8, 9><<<4096 * 2, 256>>>(d_inputs, d_outputs, d_stateSpaceMatrix);
}

void kernelDifferentSignalsStateSpaceMatrixLaunch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, 
												  uint32_t *d_signalIndices)
{
	filterStateSpaceMatrixSignalLengthsKernel<<<250, 256>>>(d_inputs, d_outputs, d_stateSpaceMatrix, d_signalIndices);
}

void kernelFiniteFilterBlockPerSignalLaunch(const float **d_inputs, float **d_outputs, float *d_filterValues, uint16_t numberOfSignals,
											uint16_t signalLength, uint16_t finiteFilterLength)
{
	if (signalLength == 512)
	{
		if (finiteFilterLength == 315)
		{
			finiteFilterBlockPerSignalKernel<512, 315><<<numberOfSignals, 512>>>(d_inputs, d_outputs, d_filterValues);
		}
		else if (finiteFilterLength == 61)
		{
			finiteFilterBlockPerSignalKernel<512, 61><<<numberOfSignals, 512>>>(d_inputs, d_outputs, d_filterValues);
		}
	}
	else if (signalLength == 256)
	{
		if (finiteFilterLength == 63)
		{
			finiteFilterBlockPerSignalKernel<256, 63><<<numberOfSignals, 256>>>(d_inputs, d_outputs, d_filterValues);
		}
		else if (finiteFilterLength == 58)
		{
			finiteFilterBlockPerSignalKernel<256, 58><<<numberOfSignals, 256>>>(d_inputs, d_outputs, d_filterValues);
		}
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
}

void kernelFiniteFilterMoreBlocksPerSignalLaunch(const float **d_inputs, float **d_outputs, float *d_filterValues, uint16_t numberOfSignals,
											     uint16_t signalLength, uint16_t finiteFilterLength)
{
	if (signalLength == 1024)
	{
		if (finiteFilterLength == 380)
		{
			finiteFilterMoreBlocksPerSignalKernel<1024, 380, 2><<<numberOfSignals << 2, 256>>>(d_inputs, d_outputs, d_filterValues);
			//finiteFilterMoreBlocksAndThredsKernel<1024, 380, 2, 2><<<numberOfSignals << 4, 256>>>(d_inputs, d_outputs, d_filterValues);
		}
		else if (finiteFilterLength == 60)
		{
			finiteFilterMoreBlocksPerSignalKernel<1024, 60, 2><<<numberOfSignals << 2, 256>>>(d_inputs, d_outputs, d_filterValues);
		}
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
}

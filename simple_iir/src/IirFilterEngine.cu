#include "IirFilterEngine.cuh"
#include "Utility.cuh"

using namespace std;
const uint8_t THREADS = 128;
const uint8_t PADDING = 4;

__global__ void filterGenericKernel(const float *d_inputs, float *d_inputs_buffer, float *d_outputs, float *d_filterValues, 
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
		d_inputs = &d_inputs[d_signalOffsets[signalIndex]];
		d_inputs_buffer = &d_inputs_buffer[d_signalOffsets[signalIndex]];
		d_outputs = &d_outputs[d_signalOffsets[signalIndex]];
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
						output -= d_outputs[j - k] * d_filterValues[k]; // y[j-k] * a[k]
					}
				}
				for (uint32_t k = filterLenght; k < filterLenght << 1; k++) // Bs second
				{
					if (j + filterLenght >= k)
					{
						output += d_inputs[j + filterLenght - k] * d_filterValues[k]; // x[j-k] * b[k]
					}
				}
				
				d_outputs[j] = output / d_filterValues[0]; // y[j] = sum(b[k] * x[j-k]) - sum(a[k] * y[j-k]) / a[0]
				if (j >= filterLenght)
				{
					d_inputs_buffer[j - filterLenght] = d_outputs[j - filterLenght]; // copy the outputs to the inputs with a delay of filterSize
				}
			}

			for (uint32_t j = signalLength - filterLenght; j < signalLength; j++)
			{
				d_inputs_buffer[j] = d_outputs[j]; // copy the last filterSize outputs to the inputs
			}
			// next filter will use the outputs of the previous filter as inputs and the initial inputs won't be overwritten
			d_inputs = d_inputs_buffer;
			d_filterValues = &d_filterValues[filterLenght << 1]; // move to next filter
		}
	}
}

__global__ void filterAnyOrderLength512Kernel(float *d_inputs, float *d_outputs, float *d_filterValues, uint32_t numberOfSignals, 
											  int32_t filterLength)
{
	const uint16_t signalLength = 512;
	uint32_t signalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (signalIndex < numberOfSignals)
	{
		uint64_t signalOffset = signalIndex * signalLength;
		d_inputs = &d_inputs[signalOffset];
		d_outputs = &d_outputs[signalOffset];
		d_filterValues = &d_filterValues[signalIndex * (filterLength << 1)];

		for (uint32_t i = 0; i < signalLength; i++) // the compiler may unroll, but it is not forced here
		{
			float output = 0;
			for (int32_t j = 1; j < filterLength; j++) // As first
			{
				if (i >= j)
				{
					output -= d_outputs[i - j] * d_filterValues[j]; // y[j-k] * a[k]
				}
			}

			for (int32_t j = filterLength; j < filterLength << 1; j++) // Bs second
			{
				if (i + filterLength >= j)
				{
					output += d_inputs[i + filterLength - j] * d_filterValues[j]; // x[j-k] * b[k]
				}
			}

			d_outputs[i] = output; // y[j] = sum(b[k] * x[j-k]) - sum(a[k] * y[j-k]) / a[0], where a[0] = 1
		}
	}
}

template <bool applyOffset, uint8_t padding, bool fast>
__global__ void filterOrder1VectorizedLength512Kernel(float *d_inputs, float *d_outputs, float *d_filterValues, uint32_t numberOfSignals, 
													  uint32_t offset)
{
	const uint16_t signalLength = 512;
	uint32_t signalIndex;
	if (applyOffset)
	{
		signalIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
	}
	else
	{
		signalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	}
	
	if (signalIndex < numberOfSignals)
	{
		d_inputs = &d_inputs[signalIndex * (signalLength + padding)];
		if (fast)
		{
			d_outputs = &d_outputs[signalIndex * (signalLength + padding)];
		}
		else
		{
			d_outputs = &d_outputs[signalIndex * signalLength];
		}
		d_filterValues = &d_filterValues[signalIndex << 2];
		
		// load 4 floats in one memory read
		float4 filter = reinterpret_cast<float4*>(d_filterValues)[0];
		float4 inputBatch = reinterpret_cast<float4*>(d_inputs)[0];
		float4 outputBatch;

		// the first sample must be calculated separately to avoid if-else statement in the loop
		float lastOutput = inputBatch.x * filter.y;
		float lastInput = inputBatch.x;
		outputBatch.x = lastOutput;

		lastOutput = inputBatch.y * filter.y + lastInput * filter.w - lastOutput * filter.z;
		lastInput = inputBatch.y;
		outputBatch.y = lastOutput;

		lastOutput = inputBatch.z * filter.y + lastInput * filter.w - lastOutput * filter.z;
		lastInput = inputBatch.z;
		outputBatch.z = lastOutput;

		lastOutput = inputBatch.w * filter.y + lastInput * filter.w - lastOutput * filter.z;
		lastInput = inputBatch.w;
		outputBatch.w = lastOutput;

		reinterpret_cast<float4*>(d_outputs)[0] = outputBatch; // store 4 floats in one memory write

		#pragma unroll 127 // 512 / 4 - 1 = 127 (works only for signals of length 512)
		for (uint32_t i = 4; i < signalLength; i += 4)
		{
			inputBatch = reinterpret_cast<float4*>(&d_inputs[i])[0]; // load 4 floats in one memory read

			// filtering of the 4 loaded values
			lastOutput = inputBatch.x * filter.y + lastInput * filter.w - lastOutput * filter.z;
			lastInput = inputBatch.x;
			outputBatch.x = lastOutput;

			lastOutput = inputBatch.y * filter.y + lastInput * filter.w - lastOutput * filter.z;
			lastInput = inputBatch.y;
			outputBatch.y = lastOutput;

			lastOutput = inputBatch.z * filter.y + lastInput * filter.w - lastOutput * filter.z;
			lastInput = inputBatch.z;
			outputBatch.z = lastOutput;

			lastOutput = inputBatch.w * filter.y + lastInput * filter.w - lastOutput * filter.z;
			lastInput = inputBatch.w;
			outputBatch.w = lastOutput;

			reinterpret_cast<float4*>(&d_outputs[i])[0] = outputBatch; // store 4 floats in one memory write
		}
	}
}

template <bool applyOffset, uint8_t padding, bool fast>
__global__ void filterOrder2VectorizedLength512Kernel(float *d_inputs, float *d_outputs, float *d_filterValues, uint32_t numberOfSignals, 
													  uint32_t offset)
{
	const uint16_t signalLength = 512;

	// process odd/even signals based on blockId, in order to enter the same branch for all threads in a block the number of blocks is guaranteed to be even
	uint32_t signalIndex;
	if (applyOffset) // offset is applied only when launching the kernel multiple times
	{
		signalIndex = (blockIdx.x & ~1U) * blockDim.x + (threadIdx.x << 1) + (blockIdx.x & 1U) + offset;
	}
	else
	{
		signalIndex = (blockIdx.x & ~1U) * blockDim.x + (threadIdx.x << 1) + (blockIdx.x & 1U);
	}

	if (signalIndex < numberOfSignals)
	{
		d_inputs = &d_inputs[signalIndex * (signalLength + padding)];
		if (fast) // padding is applied to the outpus only when using the "fast" memory allocation and copy
		{
			d_outputs = &d_outputs[signalIndex * (signalLength + padding)];
		}
		else
		{
			d_outputs = &d_outputs[signalIndex * signalLength];
		}
		d_filterValues = &d_filterValues[signalIndex * 6];

		float4 filterPart1;
		float2 filterPart2;
		if (blockIdx.x & 1U) // odd signalIndex, first the 2 floats are loaded, second the 4 floats are loaded
		{
			filterPart2 = reinterpret_cast<float2*>(d_filterValues)[0];     // two last filter values are loaded first in 'addSignal()'
			filterPart1 = reinterpret_cast<float4*>(&d_filterValues[2])[0]; // first four filter values are loaded last in 'addSignal()'
		}
		else // even signalIndex, first the 4 floats are loaded, second the 2 floats are loaded
		{
			// values are loaded in order in 'addSignal()'
			filterPart1 = reinterpret_cast<float4*>(d_filterValues)[0];
			filterPart2 = reinterpret_cast<float2*>(&d_filterValues[4])[0];
		}
		float4 inputBatch = reinterpret_cast<float4*>(d_inputs)[0];
		float4 outputBatch;

		// the first two samples must be calculated separately to avoid if-else statement in the loop
		float secondLastInput = inputBatch.x;
		float lastInput = inputBatch.y;
		float secondLastOutput = secondLastInput * filterPart1.y;
		float lastOutput = lastInput * filterPart1.y + secondLastInput * filterPart1.w - secondLastOutput * filterPart1.z;
		outputBatch.x = secondLastOutput;
		outputBatch.y = lastOutput;

		float currentOutput = inputBatch.z * filterPart1.y + lastInput * filterPart1.w + secondLastInput * filterPart2.y - 
							  lastOutput * filterPart1.z - secondLastOutput * filterPart2.x;
		secondLastOutput = lastOutput;
		lastOutput = currentOutput;
		secondLastInput = lastInput;
		lastInput = inputBatch.z;
		outputBatch.z = lastOutput;

		currentOutput = inputBatch.w * filterPart1.y + lastInput * filterPart1.w + secondLastInput * filterPart2.y - 
						lastOutput * filterPart1.z - secondLastOutput * filterPart2.x;
		secondLastOutput = lastOutput;
		lastOutput = currentOutput;
		secondLastInput = lastInput;
		lastInput = inputBatch.w;
		outputBatch.w = lastOutput;

		reinterpret_cast<float4*>(d_outputs)[0] = outputBatch;

		#pragma unroll 127 // 512 / 4 - 1 = 127 (works only for signals of length 512)
		for (uint32_t i = 4; i < signalLength; i += 4)
		{
			inputBatch = reinterpret_cast<float4*>(&d_inputs[i])[0]; // load 4 floats in one memory read

			// filtering of the 4 loaded values
			currentOutput = inputBatch.x * filterPart1.y + lastInput * filterPart1.w + secondLastInput * filterPart2.y - 
						 	lastOutput * filterPart1.z - secondLastOutput * filterPart2.x;
			secondLastOutput = lastOutput;
			lastOutput = currentOutput;
			secondLastInput = lastInput;
			lastInput = inputBatch.x;
			outputBatch.x = lastOutput;

			currentOutput = inputBatch.y * filterPart1.y + lastInput * filterPart1.w + secondLastInput * filterPart2.y - 
							lastOutput * filterPart1.z - secondLastOutput * filterPart2.x;
			secondLastOutput = lastOutput;
			lastOutput = currentOutput;
			secondLastInput = lastInput;
			lastInput = inputBatch.y;
			outputBatch.y = lastOutput;

			currentOutput = inputBatch.z * filterPart1.y + lastInput * filterPart1.w + secondLastInput * filterPart2.y - 
							lastOutput * filterPart1.z - secondLastOutput * filterPart2.x;
			secondLastOutput = lastOutput;
			lastOutput = currentOutput;
			secondLastInput = lastInput;
			lastInput = inputBatch.z;
			outputBatch.z = lastOutput;

			currentOutput = inputBatch.w * filterPart1.y + lastInput * filterPart1.w + secondLastInput * filterPart2.y - 
							lastOutput * filterPart1.z - secondLastOutput * filterPart2.x;
			secondLastOutput = lastOutput;
			lastOutput = currentOutput;
			secondLastInput = lastInput;
			lastInput = inputBatch.w;
			outputBatch.w = lastOutput;
			
			reinterpret_cast<float4*>(&d_outputs[i])[0] = outputBatch; // write 4 floats in one memory write
		}
	}
}

void allocateMemoryOnDevice(float *&d_memory, size_t size)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_memory, size));
}

void copySignalsToDevice(std::vector<const float *> signals, std::vector<uint32_t> signalLengths, float *&d_signals)
{
	uint32_t offset = 0;
	for (uint32_t i = 0; i < signals.size(); ++i)
	{
		HANDLE_ERROR(cudaMemcpy(&d_signals[offset], signals[i], signalLengths[i] * sizeof(float), cudaMemcpyHostToDevice));
		offset += signalLengths[i];
	}
}

void copySignalFromDevice(float *&d_signals, float *&signals, uint32_t signalLength, uint32_t signalOffset)
{
	HANDLE_ERROR(cudaMemcpy(signals, &d_signals[signalOffset], signalLength * sizeof(float), cudaMemcpyDeviceToHost));
}

void allocateFilterValuesOnDevice(std::vector<float> filterValues, float *&d_filterValues)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_filterValues, filterValues.size() * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_filterValues, filterValues.data(), filterValues.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void allocateFastInputOutputMemoryOnDevice(float *&d_inputs, float *&d_outputs, uint32_t numberOfSignals, uint32_t signalLength)
{
	// add padding for potential continuous memory transfer
	HANDLE_ERROR(cudaMalloc((void**)&d_inputs, numberOfSignals * (signalLength + PADDING) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_outputs, numberOfSignals * (signalLength + PADDING) * sizeof(float)));
}

void allocateSlowInputOutputMemoryOnDevice(float *&d_inputs, float *&d_outputs, uint64_t totalSignalLength)
{
	HANDLE_ERROR(cudaMalloc((void**)&d_inputs, totalSignalLength * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_outputs, totalSignalLength * sizeof(float)));
}

void allocateMetadataOnDevice(std::vector<uint32_t> signalLengths, std::vector<uint32_t> signalOffsets, std::vector<uint32_t> filtersCounts, 
							  std::vector<uint32_t> filtersOffsets, std::vector<uint32_t> filterSizes, std::vector<uint32_t> filterSizesOffsets, 
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

void freeMemoryOnDevice(std::initializer_list<void *> d_memoryList)
{
	for (void *d_memory : d_memoryList)
	{
		if (d_memory != nullptr)
		{
			HANDLE_ERROR(cudaFree(d_memory));
		}
	}
}

inline void copySignalsToDeviceFast(const float **inputs, float *d_inputs, uint32_t numberOfSignals, uint32_t signalLength) 
{
	const float *startOfChunk = inputs[0];
	uint32_t paddedSignalLength = signalLength + PADDING;
	for (uint32_t i = 1; i < numberOfSignals; i++)
	{
		if (inputs[i] != inputs[i - 1] + paddedSignalLength) // chunk is not continuous anymore, copy it to device
		{
			uint32_t chunkLength = inputs[i - 1] - startOfChunk + signalLength;
			HANDLE_ERROR(cudaMemcpy(d_inputs, startOfChunk, chunkLength * sizeof(float), cudaMemcpyHostToDevice));
			d_inputs = &d_inputs[chunkLength + PADDING];
			startOfChunk = inputs[i];
		}
	}

	if (startOfChunk != inputs[numberOfSignals - 1]) // last chunk was not copied to device, copy it
	{
		uint32_t chunkLength = inputs[numberOfSignals - 1] - startOfChunk + paddedSignalLength;
		HANDLE_ERROR(cudaMemcpy(d_inputs, startOfChunk, chunkLength * sizeof(float), cudaMemcpyHostToDevice));
	}
}

inline void copySignalsToDeviceSlow(const float **inputs, float *d_inputs, std::vector<uint32_t> signalLengths) 
{
	for (uint32_t i = 0; i < signalLengths.size(); i++)
	{
		HANDLE_ERROR(cudaMemcpy(d_inputs, inputs[i], signalLengths[i] * sizeof(float), cudaMemcpyHostToDevice));
		d_inputs = &d_inputs[signalLengths[i]];
	}
}

inline void copySignalsToDeviceSlow(const float **inputs, float *d_inputs, uint32_t numberOfSignals, uint32_t signalLength)
{
	for (uint32_t i = 0; i < numberOfSignals; i++)
	{
		HANDLE_ERROR(cudaMemcpy(&d_inputs[i * signalLength], inputs[i], signalLength * sizeof(float), cudaMemcpyHostToDevice));
	}
}

inline void analyzeOutputMemory(float **outputs, uint32_t numberOfSignals, uint32_t signalLength, vector<__uint128_t> &paddings, 
								vector<tuple<uint32_t, uint32_t>> &chunks)
{
	paddings.reserve(numberOfSignals - 1);
	uint32_t paddedSignalLength = signalLength + PADDING;

	uint32_t indexOfChunkStart = 0;
	for (uint32_t i = 1; i < numberOfSignals; i++)
	{
		if (outputs[i] != outputs[i - 1] + paddedSignalLength)
		{
			uint32_t chunkLength = outputs[i - 1] - outputs[indexOfChunkStart] + signalLength;
			chunks.push_back(make_tuple(indexOfChunkStart, chunkLength));
			indexOfChunkStart = i;
			paddings.push_back(0);
			continue;
		}

		__uint128_t padding = reinterpret_cast<__uint128_t *>(&outputs[i - 1][signalLength])[0];
		paddings.push_back(padding);
	}

	uint32_t chunkLength = outputs[numberOfSignals - 1] - outputs[indexOfChunkStart] + signalLength;
	chunks.push_back(make_tuple(indexOfChunkStart, chunkLength));
}

inline void copySignalsToHostFast(float **outputs, float *d_outputs, uint32_t numberOfSignals, uint32_t signalLength, 
							      vector<tuple<uint32_t, uint32_t>> &chunks)
{
	for (tuple<uint32_t, uint32_t> &chunk : chunks)
	{
		uint32_t indexOfChunkStart = get<0>(chunk);
		uint32_t chunkLength = get<1>(chunk);
		HANDLE_ERROR(cudaMemcpy(outputs[indexOfChunkStart], d_outputs, chunkLength * sizeof(float), cudaMemcpyDeviceToHost));
		d_outputs = &d_outputs[chunkLength + PADDING];
	}
}

inline void copySignalsToHostFastStreams(float **outputs, float *d_outputs, uint32_t numberOfSignals, uint32_t signalLength, 
							      vector<tuple<uint32_t, uint32_t>> &chunks)
{
	vector<cudaStream_t> streams;
	streams.reserve(64);
	for (tuple<uint32_t, uint32_t> &chunk : chunks)
	{
		uint32_t indexOfChunkStart = get<0>(chunk);
		uint32_t chunkLength = get<1>(chunk);
		cudaStream_t stream;
		HANDLE_ERROR(cudaStreamCreate(&stream));
		HANDLE_ERROR(cudaMemcpyAsync(outputs[indexOfChunkStart], d_outputs, chunkLength * sizeof(float), cudaMemcpyDeviceToHost, stream));
		d_outputs = &d_outputs[chunkLength + PADDING];
	}

	for (cudaStream_t stream : streams)
	{
		HANDLE_ERROR(cudaStreamSynchronize(stream));
		HANDLE_ERROR(cudaStreamDestroy(stream));
	}
}

inline void copySignalsToHostSlow(float **outputs, float *d_outputs, uint32_t numberOfSignals, uint32_t signalLength)
{
	for (uint32_t i = 0; i < numberOfSignals; i++)
	{
		HANDLE_ERROR(cudaMemcpy(outputs[i], &d_outputs[i * signalLength], signalLength * sizeof(float), cudaMemcpyDeviceToHost));
	}
}

inline void copySignalsToHostGeneric(float **outputs, float *d_outputs, std::vector<uint32_t> signalLengths)
{
	uint64_t offset = 0;
	for (uint32_t i = 0; i < signalLengths.size(); i++)
	{
		HANDLE_ERROR(cudaMemcpy(outputs[i], &d_outputs[offset], signalLengths[i] * sizeof(float), cudaMemcpyDeviceToHost));
		offset += signalLengths[i];
	}
}

inline void restoreOutputsPaddings(float **outputs, uint32_t signalLength, vector<__uint128_t> &paddings)
{
	for (uint32_t i = 0; i < paddings.size() - 1; i++)
	{
		if (paddings[i] != 0)
		{
			reinterpret_cast<__uint128_t *>(&outputs[i][signalLength])[0] = paddings[i];
		}
	}
}

void kernelGenericLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, float *d_filterValues, 
						 uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, uint32_t *d_filtersOffsets, 
						 uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, std::vector<uint32_t> signalLengths)
{
	copySignalsToDeviceSlow(inputs, d_inputs, signalLengths);

	dim3 blockSize(THREADS, 1, 1);
	dim3 gridSize((signalLengths.size() + THREADS - 1) / THREADS, 1, 1);
	filterGenericKernel<<<gridSize, blockSize>>>(d_inputs, d_inputs, d_outputs, d_filterValues, d_signalLengths, d_signalOffsets, 
												 d_filtersCounts, d_filtersOffsets, d_filterSizes, d_filterSizesOffsets, signalLengths.size());
	
	HANDLE_ERROR(cudaDeviceSynchronize());
	copySignalsToHostGeneric(outputs, d_outputs, signalLengths);
}

void kernelGenericLaunchWithDevicePointers(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
										   uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, uint32_t *d_filtersOffsets, 
										   uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, uint32_t numberOfSignals)
{
	dim3 blockSize(THREADS, 1, 1);
	dim3 gridSize((numberOfSignals + THREADS - 1) / THREADS, 1, 1);
	filterGenericKernel<<<gridSize, blockSize>>>(*d_inputs, d_inputs_buffer, *d_outputs, d_filterValues, d_signalLengths, d_signalOffsets, 
												 d_filtersCounts, d_filtersOffsets, d_filterSizes, d_filterSizesOffsets, numberOfSignals);
	
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void kernelAnyOrderLength512Launch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, float *d_filterValues, 
								   uint32_t numberOfSignals, int32_t filterLength)
{
	const uint16_t signalLength = 512;
	copySignalsToDeviceSlow(inputs, d_inputs, numberOfSignals, signalLength);

	dim3 blockSize(THREADS, 1, 1);
	dim3 gridSize((numberOfSignals + THREADS - 1) / THREADS, 1, 1);
	filterAnyOrderLength512Kernel<<<gridSize, blockSize>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals, filterLength);

	HANDLE_ERROR(cudaDeviceSynchronize());
	copySignalsToHostSlow(outputs, d_outputs, numberOfSignals, signalLength);
}

void kernelVectorizedLength512Launch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
									 float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength)
{
	const uint16_t signalLength = 512;
	copySignalsToDeviceSlow(inputs, d_inputs, numberOfSignals, signalLength);

	dim3 blockSize(THREADS, 1, 1);
	dim3 gridSize((numberOfSignals + THREADS - 1) / THREADS, 1, 1);
	if (filterLength == 2)
	{
		filterOrder1VectorizedLength512Kernel<false, 0, false><<<gridSize, blockSize>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals);
	}
	else
	{
		gridSize.x += gridSize.x & 1; // make sure the number of blocks is even
		filterOrder2VectorizedLength512Kernel<false, 0, false><<<gridSize, blockSize>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals);
	}
	
	HANDLE_ERROR(cudaDeviceSynchronize());
	copySignalsToHostSlow(outputs, d_outputs, numberOfSignals, signalLength);
}

void kernelVectorizedLength512AsyncStreamsLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
										   		 float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength, 
												 std::vector<cudaStream_t> streams)
{
	const uint16_t signalLength = 512;
	const uint32_t signalsPerBatch = 256;
	const uint32_t numberOfBatches = (numberOfSignals + signalsPerBatch - 1) / signalsPerBatch;

	dim3 blockSize(THREADS, 1, 1);
	for (uint32_t i = 0; i < numberOfBatches; i++)
	{
	    uint32_t startSignal = i * signalsPerBatch;
	    uint32_t endSignal = min(i * signalsPerBatch + signalsPerBatch, numberOfSignals);

	    HANDLE_ERROR(cudaStreamCreate(&streams[i]));

	    for (uint32_t j = startSignal; j < endSignal; j++)
	    {
	        HANDLE_ERROR(cudaMemcpyAsync(&d_inputs[j * signalLength], inputs[j], signalLength * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
	    }

	    dim3 gridSize(2, 1, 1);
	    if (filterLength == 2)
	    {
	        filterOrder1VectorizedLength512Kernel<true, 0, false><<<gridSize, blockSize, 0, streams[i]>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals, i * signalsPerBatch);
	    }
	    else
	    {
	        filterOrder2VectorizedLength512Kernel<true, 0, false><<<gridSize, blockSize, 0, streams[i]>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals, i * signalsPerBatch);
	    }

	    for (uint32_t j = startSignal; j < endSignal; j++)
	    {
	        HANDLE_ERROR(cudaMemcpyAsync(outputs[j], &d_outputs[j * signalLength], signalLength * sizeof(float), cudaMemcpyDeviceToHost, streams[j]));
	    }
	}

	for (uint32_t i = 0; i < numberOfBatches; i++)
	{
	    HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
	    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
	}
}

void kernelVectorizedLength512AsyncStreamsFastLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
										   		     float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength, 
												     std::vector<cudaStream_t> streams)
{
	uint32_t computedSignals = 0;
	uint32_t offset = 0;
	uint16_t numberOfBlocks = 0;
	const uint16_t minSignalsInChunk = 256;
	const uint32_t chunkMask = 255;
	const uint16_t signalLength = 512;
	const uint16_t paddedSignalLength = signalLength + PADDING;
	const float *startOfChunk = inputs[0];
	const float *endOfChunk = startOfChunk;
	float *d_inputsCopy = d_inputs;
	uint32_t streamIndex = 0;

	HANDLE_ERROR(cudaStreamCreate(&streams[streamIndex]));
	for (uint32_t i = 1; i < numberOfSignals; i++)
	{
		if (!(i & chunkMask))
		{
			endOfChunk = inputs[i - 1];
			numberOfBlocks += 2; // always force even number of blocks, so kernel for signals of order 2 can be launched properly
		}

		if (inputs[i] != inputs[i - 1] + paddedSignalLength) // chunk is not continuous anymore, copy it to device
		{
			if (startOfChunk != endOfChunk) // at least one chunk was collected, copy it and launch kernel to compute it
			{
				uint32_t chunkLength = endOfChunk - startOfChunk + signalLength;
				HANDLE_ERROR(cudaMemcpyAsync(d_inputsCopy, startOfChunk, chunkLength * sizeof(float), cudaMemcpyHostToDevice, 
										     streams[streamIndex])); // copy only memory of multiple of a chunk to device
				d_inputsCopy = &d_inputsCopy[chunkLength + PADDING];

				// launch computation
				dim3 blockSize(THREADS, 1, 1);
				dim3 gridSize(numberOfBlocks, 1, 1);
				if (filterLength == 2)
				{
					filterOrder1VectorizedLength512Kernel<true, PADDING, false><<<gridSize, blockSize, 0, streams[streamIndex]>>>
						(d_inputs, d_outputs, d_filterValues, numberOfSignals, offset);
				}
				else
				{
					gridSize.x += gridSize.x & 1; // make sure the number of blocks is even
					filterOrder2VectorizedLength512Kernel<true, PADDING, false><<<gridSize, blockSize, 0, streams[streamIndex]>>>
						(d_inputs, d_outputs, d_filterValues, min(offset + numberOfBlocks * minSignalsInChunk, numberOfSignals), offset);
				}

				for (uint32_t j = 0; j < numberOfBlocks * minSignalsInChunk; j++)
				{
					HANDLE_ERROR(cudaMemcpyAsync(outputs[computedSignals], &d_outputs[computedSignals++ * signalLength], 
												signalLength * sizeof(float), cudaMemcpyDeviceToHost, streams[streamIndex]));
				}
				
				offset += numberOfBlocks * minSignalsInChunk;
				numberOfBlocks = 0;
				
				HANDLE_ERROR(cudaStreamCreate(&streams[++streamIndex])); // create new stream for next chunk
				if (i & chunkMask) // copy the rest of the memory to device, which is not a multiple of a chunk
				{
					uint32_t chunkLength = inputs[i - 1] - endOfChunk + signalLength;
					HANDLE_ERROR(cudaMemcpyAsync(d_inputsCopy, endOfChunk, chunkLength * sizeof(float), cudaMemcpyHostToDevice, 
												 streams[streamIndex]));
					d_inputsCopy = &d_inputsCopy[chunkLength + PADDING];
				}			
				startOfChunk = inputs[i];
				endOfChunk = startOfChunk;
			}
			else // a hole chunk was not collected yet, only copy it and wait with compuation until the whole chunk is collected
			{
				uint32_t chunkLength = inputs[i - 1] - startOfChunk + signalLength;
				HANDLE_ERROR(cudaMemcpyAsync(d_inputsCopy, startOfChunk, chunkLength * sizeof(float), cudaMemcpyHostToDevice, 
											 streams[streamIndex]));
				d_inputsCopy = &d_inputsCopy[chunkLength + PADDING];
				startOfChunk = inputs[i];
			}
		}
		else if (i + 1 == numberOfSignals) // last signal, copy the rest of the memory and launch kernel to compute it
		{
			uint32_t chunkLength = inputs[i] - startOfChunk + signalLength;
			HANDLE_ERROR(cudaMemcpyAsync(d_inputsCopy, startOfChunk, chunkLength * sizeof(float), cudaMemcpyHostToDevice, streams[streamIndex]));

			// compuatation
			numberOfBlocks += 2;
			dim3 blockSize(THREADS, 1, 1);
			dim3 gridSize(numberOfBlocks, 1, 1);
			if (filterLength == 2)
			{
				filterOrder1VectorizedLength512Kernel<true, PADDING, false><<<gridSize, blockSize, 0, streams[streamIndex]>>>
					(d_inputs, d_outputs, d_filterValues, numberOfSignals, offset);
			}
			else
			{
				filterOrder2VectorizedLength512Kernel<true, PADDING, false><<<gridSize, blockSize, 0, streams[streamIndex]>>>
					(d_inputs, d_outputs, d_filterValues, numberOfSignals, offset);
			}

			// copy memory back to host using all the created streams
			for (uint32_t j = computedSignals; j < numberOfSignals; j++)
			{
				HANDLE_ERROR(cudaMemcpyAsync(outputs[j], &d_outputs[j * signalLength], 
											signalLength * sizeof(float), cudaMemcpyDeviceToHost, streams[streamIndex]));
			}
		}
	}
	
	// make sure all streams finish compuation, then destroy them
	for (uint32_t i = 0; i <= streamIndex; i++)
	{
	    HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
	    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
	}
}

void kernelVectorizedLength512FastLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
										 float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength)
{
	const uint16_t signalLength = 512;
	copySignalsToDeviceFast(inputs, d_inputs, numberOfSignals, signalLength); // copy memory as continously as possible to device

	// compute
	dim3 blockSize(THREADS, 1, 1);
	dim3 gridSize((numberOfSignals + THREADS -1) / THREADS, 1, 1);
	if (filterLength == 2)
	{
		filterOrder1VectorizedLength512Kernel<false, PADDING, true><<<gridSize, blockSize>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals);
	}
	else
	{
		gridSize.x += gridSize.x & 1; // make sure the number of blocks is even
		filterOrder2VectorizedLength512Kernel<false, PADDING, true><<<gridSize, blockSize>>>(d_inputs, d_outputs, d_filterValues, numberOfSignals);
	}
	
	// analyze output memory while the kernels are running on the device
	vector<__uint128_t> paddings;
	vector<tuple<uint32_t, uint32_t>> chunks;
	analyzeOutputMemory(outputs, numberOfSignals, signalLength, paddings, chunks); // save the memory chunks, that will be rewritten and should not be copied back to host
	
	HANDLE_ERROR(cudaDeviceSynchronize());
	copySignalsToHostFast(outputs, d_outputs, numberOfSignals, signalLength, chunks); // copy memory in large chunks
	restoreOutputsPaddings(outputs, signalLength, paddings); // restore the rewritten memory chunks
}


#pragma once

#include "filter.h"

#include <iostream>
#include <vector>
#include </usr/include/signal.h> // absolute path, because relative signal.h file exists in the project
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>

// Forward declarations of functions defined in the IirFilterEngine.cu file.

/**
 * @brief 00 Baseline: Launches a generic kernel that can handle any order and any length of the filter.
 */
void kernelGenericLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, float *d_filterValues, 
						 uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, uint32_t *d_filtersOffsets, 
						 uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, std::vector<uint32_t> signalLengths);
										
/**
 * @brief 01 Introducing Constraints: Launches a kernel that can handle only filters of order one and length 512.
 */
void kernelAnyOrderLength512Launch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, float *d_filterValues, 
								   uint32_t numberOfSignals, int32_t filterLength);

/**
 * @brief 02 Vectorizing and Unrolling: Launches a kernel that can handle only filters of order 1 and 2 and signals of length 512.
 * 	  									The memory read/writes are vectorized and loop is unrolled.
 */
void kernelVectorizedLength512Launch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
									 float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength);

/**
 * @brief 03 Streams and Asynchronicity: Launches a kernel that can handle only filters of order 1 and 2 and signals of length 512.
 * 	  									 Uses streams to partially hide memory copy latency with computation.
 */
void kernelVectorizedLength512AsyncStreamsLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
										   		 float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength, 
												 std::vector<cudaStream_t> streams);

/**
 * @brief 04 Safe Hacking: Launches a kernel that can handle only filters of order 1 and 2 and signals of length 512.
 * 						   Tries to copy memory to device in larger chunks.
 */
void kernelVectorizedLength512AsyncStreamsFastLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
										   		     float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength, 
												     std::vector<cudaStream_t> streams);

/**
 * @brief 05 Unsafe Hacking: Launches a kernel that can handle only filters of order 1 and 2 and signals of length 512.
 * 							 Tries to copy memory to device and from device in larger chunks.
 */
void kernelVectorizedLength512FastLaunch(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, 
										 float *d_filterValues, uint32_t numberOfSignals, int32_t filterLength);

/**
 * @brief Launches a generic kernel that can handle any order and any length of the filter. Expects the input data already on device.
 */					 
void kernelGenericLaunchWithDevicePointers(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
										   uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, uint32_t *d_filtersOffsets, 
										   uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, uint32_t numberOfSignals);

// other functions defined in the IirFilterEngine.cu file, which are self-explanatory by their names

void allocateMemoryOnDevice(float *&d_memory, size_t size);
void copySignalsToDevice(std::vector<const float *> signals, std::vector<uint32_t> signalLengths, float *&d_signals);
void copySignalFromDevice(float *&d_signals, float *&signals, uint32_t signalLength, uint32_t signalOffset);
void allocateFilterValuesOnDevice(std::vector<float> filterValues, float *&d_filterValues);
void allocateFastInputOutputMemoryOnDevice(float *&d_inputs, float *&d_outputs, uint32_t numberOfSignals, uint32_t signalLength);
void allocateSlowInputOutputMemoryOnDevice(float *&d_inputs, float *&d_outputs, uint64_t totalSignalLength);
void allocateMetadataOnDevice(std::vector<uint32_t> signalLengths, std::vector<uint32_t> signalOffsets, std::vector<uint32_t> filtersCounts, 
							  std::vector<uint32_t> filtersOffsets, std::vector<uint32_t> filterSizes, std::vector<uint32_t> filterSizesOffsets, 
							  uint32_t *&d_signalLengths, uint32_t *&d_signalOffsets, uint32_t *&d_filtersCounts, uint32_t *&d_filtersOffsets, 
							  uint32_t *&d_filterSizes, uint32_t *&d_filterSizesOffsets);
void freeMemoryOnDevice(std::initializer_list<void *> d_memory);

typedef void(*kernelLaunchFunction)(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, float *d_filterValues, 
									uint32_t numberOfSignals, int32_t filterLength);
typedef void(*kernelStreamedLaunchFunction)(const float **inputs, float **outputs, float *d_inputs, float *d_outputs, float *d_filterValues, 
									uint32_t numberOfSignals, int32_t filterLength, std::vector<cudaStream_t> streams);
/**
 * @brief Defines the type of the kernel that will be launched on the GPU if it is possible given the signal and filter parameters
 */
enum class KernelType
{
	BASELINE_00,
	CONSTRAINTS_01,
	VECTORIZED_UNROLLED_02,
	STREAMS_ASYNCHRONICITY_03,
	SAFE_HACKING_04,
	UNSAFE_HACKING_05,
};

/**
 * @brief The IirFilterEngine class implements a parallel CUDA-based IIR filtering engine; it supports parallel filtering of an arbitrary 
 * 		  number of signals, each signal may be of different length filters may be of order one or two
 *        [Ass 01: each signal may only be filtered by a single filter]
 *		  [Ass 02: each signal may be filtered by up to 8 filters in a cascade]
 */
class IirFilterEngine
{
private:
	// control variables
	KernelType _kernelType;
	bool _firstCall = true;
	bool _moreFilters = false;
	bool _variableFilterSize = false;
	bool _variableSignalLength = false;
	bool _a0Not1 = false;
	int32_t _filterLength = 0;
	uint32_t _numberOfSignals = 0;
	uint32_t _filtersOffset = 0;
	uint32_t _filterSizesOffset = 0;
	int32_t _signalLength = 0;
	uint32_t _totalSignalLength = 0;
	
	// host vectors
	std::vector<uint32_t> _signalLengths;
	std::vector<uint32_t> _signalOffsets;
	std::vector<uint32_t> _filtersCounts;
	std::vector<uint32_t> _filterSizes;
	std::vector<uint32_t> _filtersOffsets;
	std::vector<uint32_t> _filterSizesOffsets;
	std::vector<float> _filterFastValues;
	std::vector<float> _filterSlowValues;
	std::vector<cudaStream_t> _streams;
	
	// device pointers
	float *_d_filterValues = nullptr;
	float *_d_inputs = nullptr;
	float *_d_outputs = nullptr;
	float *_d_inputs_buffer = nullptr;
	uint32_t *_d_signalLengths = nullptr;
	uint32_t *_d_signalOffsets = nullptr;
	uint32_t *_d_filtersCounts = nullptr;
	uint32_t *_d_filterSizes = nullptr;
	uint32_t *_d_filtersOffsets = nullptr;
	uint32_t *_d_filterSizesOffsets = nullptr;

	// kernel launch functions
	kernelLaunchFunction _fastKernelLaunchFunction = nullptr;
	kernelStreamedLaunchFunction _kernelStreamedLaunchFunction = nullptr;

public:
	/**
	 * @param kernelType if possible given a signal length and filter length, the engine will choose the specified kernel type
	 */
	IirFilterEngine(KernelType kernelType = KernelType::UNSAFE_HACKING_05);
	~IirFilterEngine();
	
	/**
	 * @brief Add a signal for processing. the method is called during preparation and before finalize is being called.
	 * @param signalLength the number of samples of the signal
	 * @param filters a pointer to a cascade of filters (only 1 for ass 1)
	 * @param numFilters the number of filters in the cascade, (ass1: 1, ass2: 1-8); each filter for this signal has the same order!
	 */
	void addSignal(int signalLength, const Filter* filters, int numFilters = 1);

	/**
	 * @brief Called when all signals have been setup and guaranteed to be only called once. IirFilterEngine may carry out preparations here
	 */
	void finalize();

	/**
	 * @brief Called to kick off the processing on the GPU with data still residing on the CPU, requiring the transfer of the input data to 
	 * 		  the GPU and transferring the results back to the CPU. The buffers on the CPU are already allocated. The method may be called 
	 * 		  multiple times with different inputs. This method is called for Ass 01 and Ass 02, but performance is only measured in Ass 01.
	 * @param inputs array of pointers to the input signals in the order addSignal was called
	 * @param outputs array of pointers to where the out signals should be placed; in the order addSignal was called       
	 */
	void executeWithHostPointers(const float** inputs, float** outputs);

	/**
	 * @brief Called to kick off the processing on the GPU with pointers already present on the GPU. The method may be called multiple times
	 *        with different inputs. This method is only called for Ass 02. The performance of this method will be evaluated in Ass 02.
	 * @param d_inputs array of pointers to the input signals in the order addSignal was called
	 * @param d_outputs array of pointers to where the out signals should be placed; in the order addSignal was called
	 */
	void executeWithDevicePointers(const float** d_inputs, float** d_outputs);
};
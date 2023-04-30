#pragma once

#include "filter.h"
#include "babymath.h"
#include "cmath"

#include <iostream>
#include <vector>
#include </usr/include/signal.h> // absolute path, because relative signal.h file exists in the project
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <algorithm>
#include <bit>

// Forward declarations of functions defined in the IirFilterEngine.cu file and documented in challenge.md.			 
void kernelGenericLaunchWithDevicePointers(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
										   uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, 
										   uint32_t *d_filtersOffsets, uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, 
										   uint32_t numberOfSignals);

void kernelFiniteInfiniteFilterLongSignalLaunch(const float **d_inputs, float **d_outputs, float *d_finiteFilter, uint32_t filterLength, 
								                float *d_infiniteFilter);
void kernelStateSpaceParallelOrder2Launch(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_stateSpace, 
								          uint16_t numberOfSignals);
void kernelStateSpaceMatrixOrder2Launch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, uint16_t numberOfSignals);
void kernelStateSpaceMatrixOrder1ParallelLaunch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, uint16_t numberOfSignals);
void kernelDifferentSignalsStateSpaceMatrixLaunch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, 
												  uint32_t *d_signalIndices);
void kernelStateSpaceMatrixOrder1LongSignalLaunch(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix);
void kernelFiniteFilterBlockPerSignalLaunch(const float **d_inputs, float **d_outputs, float *d_filterValues, uint16_t numberOfSignals,
											uint16_t signalLength, uint16_t finiteFilterLength);
void kernelFiniteFilterMoreBlocksPerSignalLaunch(const float **d_inputs, float **d_outputs, float *d_filterValues, uint16_t numberOfSignals,
											     uint16_t signalLength, uint16_t finiteFilterLength);

// other functions defined in the IirFilterEngine.cu file, which are self-explanatory by their names
void allocateMemoryOnDevice(float *&d_memory, size_t size);
void allocate2DArrayOnDevice(float ***d_signals, float **h_signalPtrs, int numSignals, int signalLength);
void copy2DArrayToDevice(float** h_signalPtrs, const float **signals, int numSignals, int signalLength);
void copy2DArrayToHost(float *&signals, float **h_signalPtrs, int signalIndex, int signalLength);
void free2DArrayOnDeviceMemory(float **d_signals, float **h_signalPtrs, int numSignals);
void allocateInfiniteFiltersOnDevice(float *filterValues, uint32_t length, float *&d_filterValues);
void allocateInfiniteFiltersOnDevice(std::vector<float> filterValues, float *&d_filterValues, uint32_t length = 0);
void allocateFiniteFiltersOnDevice(std::vector<std::vector<float>> finiteFilters, float *&d_finiteFilters, bool reverseFilter = true);
void allocateStateSpaceOnDevice(std::vector<std::vector<float>> stateSpace, float *&d_stateSpace);
void allocateSignalIndicesOnDevice(std::vector<std::vector<uint32_t>> signalLengthGroupsOrder1, std::vector<std::vector<uint32_t>> signalLengthGroupsOrder2, 
								   uint32_t *&d_signalIndices);
void allocateStateSpaceMatricesOnDevice(std::vector<std::vector<float>> stateSpaceMatrices, float *&d_stateSpaceMatrices);
void allocateMetadataOnDevice(std::vector<uint32_t> signalLengths, std::vector<uint32_t> signalOffsets, std::vector<uint32_t> filtersCounts, 
							  std::vector<uint32_t> filtersOffsets, std::vector<uint32_t> filterSizes, std::vector<uint32_t> filterSizesOffsets, 
							  uint32_t *&d_signalLengths, uint32_t *&d_signalOffsets, uint32_t *&d_filtersCounts, uint32_t *&d_filtersOffsets, 
							  uint32_t *&d_filterSizes, uint32_t *&d_filterSizesOffsets);
void freeMemoryOnDevice(std::initializer_list<void *> d_memory);

typedef void(*FiniteInfiniteFilterLongSignalFunction)(const float **d_inputs, float **d_outputs, float *d_finiteFilter, uint32_t filterLength, 
								                      float *d_infiniteFilter);
typedef void(*FiniteInfiniteFilterOrder1Function)(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_infiniteFilter, 
								          uint16_t numberOfSignals);
typedef void(*DifferentSignalLengthsFunction)(const float **d_inputs, float **d_outputs, float *d_finiteFilter, uint32_t *d_signalIndices, 
										      std::vector<uint32_t> &offsets, std::vector<cudaStream_t> streams);
typedef void(*StateSpaceMatrixOrder2Function)(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, uint16_t numberOfSignals);
typedef void(*StateSpaceMatrixOrder1ParallelFunction)(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, uint16_t numberOfSignals);
typedef void(*DifferentSignalStateSpaceMatrixFunction)(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix, 
													   uint32_t *d_signalIndices);
typedef void(*StateSpaceMatrixOrder1LongSignalFunction)(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix);
typedef void(*FiniteFilterBlockPerSignalFunction)(const float **d_inputs, float **d_outputs, float *d_filterValues, uint16_t numberOfSignals,
												  uint16_t signalLength, uint16_t finiteFilterLength);
typedef void(*FiniteFilterMoreBlocksPerSignalFunction)(const float **d_inputs, float **d_outputs, float *d_filterValues, uint16_t numberOfSignals,
													   uint16_t signalLength, uint16_t finiteFilterLength);

/**
 * @brief The IirFilterEngine class implements a parallel CUDA-based IIR filtering engine; it supports parallel filtering of an arbitrary 
 * 		  number of signals, each signal may be of different length filters may be of order one or two
 *        [Ass 01: each signal may only be filtered by a single filter]
 *		  [Ass 02: each signal may be filtered by up to 8 filters in a cascade]
 */
class IirFilterEngine
{
private:
	const float EPS = 0.0000001;

	// control variables
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
	std::vector<float> _infiniteFilters;
	std::vector<cudaStream_t> _streams;
	std::vector<std::vector<float>> _uniqueFilters;
	std::vector<std::vector<float>> _finiteFilters;
	std::vector<std::vector<float>> _stateSpaces;
	std::vector<std::vector<float>> _stateSpaceMatrices;
	std::vector<float> _filterMatrix;
	std::vector<std::vector<uint32_t>> _signalLengthGroupsOrder1;
	std::vector<std::vector<uint32_t>> _signalLengthGroupsOrder2;
	std::vector<uint32_t> _groupOffsets;
	
	// device pointers
	float *_d_infiniteFilters = nullptr;
	float *_d_inputs = nullptr;
	float *_d_outputs = nullptr;
	float *_d_inputsBuffer = nullptr;
	float *_d_finiteFilter = nullptr;
	float *_d_stateSpace = nullptr;
	float *_d_stateSpaceMatrices = nullptr;
	uint32_t *_d_signalLengths = nullptr;
	uint32_t *_d_signalOffsets = nullptr;
	uint32_t *_d_filtersCounts = nullptr;
	uint32_t *_d_filterSizes = nullptr;
	uint32_t *_d_filtersOffsets = nullptr;
	uint32_t *_d_filterSizesOffsets = nullptr;
	uint32_t *_d_signalIndices = nullptr;

	// kernel launch functions
	FiniteInfiniteFilterLongSignalFunction _finiteInfiniteFilterLongSignalFunction = nullptr;
	FiniteInfiniteFilterOrder1Function _finiteInfiniteFilterOrder1Function = nullptr;
	DifferentSignalLengthsFunction _differentSignalLengthsFunction = nullptr;
	StateSpaceMatrixOrder2Function _stateSpaceMatrixOrder2Function = nullptr;
	StateSpaceMatrixOrder1ParallelFunction _stateSpaceMatrixOrder1ParallelFunction = nullptr;
	DifferentSignalStateSpaceMatrixFunction _differentSignalStateSpaceMatrixFunction = nullptr;
	StateSpaceMatrixOrder1LongSignalFunction _stateSpaceMatrixOrder1LongSignalFunction = nullptr;
	FiniteFilterBlockPerSignalFunction _finiteFilterBlockPerSignalFunction = nullptr;
	FiniteFilterMoreBlocksPerSignalFunction _finiteFilterMoreBlocksPerSignalFunction = nullptr;
    
	/**
	 * @brief Tries to match the given filters with already seen filters, or adds the filters to the list of unique filters.
	 * @param filters the filters to match
	 * @param numberOfFilters the number of filters in the cascade
	 * @return true if the filters has been matched, false otherwise
	 */
	bool matchFilter(const Filter *filters, int numberOfFilters);

	/**
	 * @brief Converts the given IIR filters to FIR filters using the state space representation.
	 * @param filters the filters to add
	 * @param numberOfFilters the number of filters in the cascade
	 * @param signalLength the length of the signal to be filtered
	 */
	void addFiniteFilter(const Filter *filters, int numberOfFilters, uint32_t signalLength);

	/**
	 * 
	 */
	void shrinkStateSpaceMatrix(uint8_t columns);

public:
	/**
	 * @param kernelType if possible given a signal length and filter length, the engine will choose the specified kernel type
	 */
	IirFilterEngine();
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
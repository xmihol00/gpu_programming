#include <cuda_runtime_api.h>
#include <cstdio>
#include <vector>
#include <tuple>

#include "filter.h"

__global__ void filterGenericDevicePointersKernel(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
									              uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, 
									              uint32_t *d_filtersOffsets, uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, 
									              uint32_t numberOfSignals);
__global__ void filterGenericDevicePointersKernel(const float **d_inputs, float *d_inputs_buffer, float **d_outputs, float *d_filterValues, 
									              uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, 
									              uint32_t *d_filtersOffsets, uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, 
									              uint32_t numberOfSignals);
__global__ void finiteFilterBlockPerSignalKernel(const float **d_inputs, float **d_outputs, float *d_filterValues);;
__global__ void finiteFilterMoreBlocksPerSignalKernel(const float **d_inputs, float **d_outputs, float *d_filterValues);
__global__ void finiteFilterMoreBlocksAndThredsKernel(const float **d_inputs, float **d_outputs, float *d_filterValues);
__global__ void finiteFilterMoreSignalsKernel(const float **d_inputs, float **d_outputs, float *d_filterValues, uint32_t signalLength);
__global__ void finiteInfiniteFilterLongSignalKernel(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_infiniteFilter);
__global__ void finiteInfiniteFilterOrder1Kernel(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_infiniteFilter);
__global__ void filterStateSpaceParallelOrder2Kernel(const float **d_inputs, float **d_outputs, float *d_finiteFilter, float *d_stateSpace);
__global__ void filterStateSpaceMatrixOrder2Kernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix,
                                                   uint32_t *d_signalIndices);
__global__ void filterStateSpaceMatrixOrder1ParallelKernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix);
__global__ void filterStateSpaceMatrixOrder1LongSignalKernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix);
__global__ void filterStateSpaceMatrixSignalLengthsKernel(const float **d_inputs, float **d_outputs, float *d_stateSpaceMatrix,
                                                          uint32_t *d_signalIndices);

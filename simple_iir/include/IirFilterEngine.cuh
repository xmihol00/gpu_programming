#include <cuda_runtime_api.h>
#include <cstdio>
#include <vector>
#include <tuple>

#include "filter.h"

/**
 * @brief A generic kernel that can filter signals of any lenght with any number of filters of any order.
 */
__global__ void filterGenericKernel(const float *d_inputs, float *d_inputs_buffer, float *d_outputs, float *d_filterValues, 
									uint32_t *d_signalLengths, uint32_t *d_signalOffsets, uint32_t *d_filtersCounts, 
									uint32_t *d_filtersOffsets, uint32_t *d_filterSizes, uint32_t *d_filterSizesOffsets, 
									uint32_t numberOfSignals);

/**
 * @brief A kernel that can filter only signals of lenght 512 with 1 filter of any order.
 */
__global__ void filterAnyOrderLength512Kernel(float *d_inputs, float *d_outputs, float *d_filterValues, uint32_t numberOfSignals, 
											  int32_t filterLength);

/**
 * @brief A kernel that can filter only signals of lenght 512 with 1 filter of order 1.
 */
template <bool aplyOffset, uint8_t padding, bool fast>
__global__ void filterOrder1VectorizedLength512Kernel(float *d_inputs, float *d_outputs, float *d_filterValues, uint32_t numberOfSignals, 
													  uint32_t offset = 0);

/**
 * @brief A kernel that can filter only signals of lenght 512 with 1 filter of order 2.
 */
template <bool aplyOffset, uint8_t padding, bool fast>
__global__ void filterOrder2VectorizedLength512Kernel(float *d_inputs, float *d_outputs, float *d_filterValues, uint32_t numberOfSignals, 
													  uint32_t offset = 0);

// other self-explanatory functions

inline void copySignalsToDeviceFast(const float **inputs, float *d_inputs, uint32_t numberOfSignals, uint32_t signalLength);

inline void copySignalsToDeviceSlow(const float **inputs, float *d_inputs, std::vector<uint32_t> signalLengths);

inline void copySignalsToDeviceSlow(const float **inputs, float *d_inputs, uint32_t numberOfSignals, uint32_t signalLength);

inline void analyzeOutputMemory(float **outputs, uint32_t numberOfSignals, uint32_t signalLength, std::vector<__uint128_t> &paddings, 
								std::vector<std::tuple<uint32_t, uint32_t>> &chunks);

inline void copySignalsToHostFast(float **outputs, float *d_outputs, uint32_t numberOfSignals, uint32_t signalLength, 
							      std::vector<std::tuple<uint32_t, uint32_t>> &chunks);

inline void copySignalsToHostSlow(float **outputs, float *d_outputs, uint32_t numberOfSignals, uint32_t signalLength);

inline void restoreOutputsPaddings(float **outputs, uint32_t signalLength, std::vector<__uint128_t> &paddings);

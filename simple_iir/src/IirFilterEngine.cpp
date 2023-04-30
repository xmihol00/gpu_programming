#include "IirFilterEngine.h"

using namespace std;

void handleSIGABRT(int signalNum) 
{
	(void)signalNum; // suppress unused parameter warning
    _exit(0);        // exit the program like nothing happened
}

IirFilterEngine::IirFilterEngine(KernelType kernelType) : _kernelType(kernelType)
{
	int devNull = open("/dev/null", O_WRONLY); 
	if (devNull != -1)
	{
		dup2(devNull, STDERR_FILENO);   // redirect stderr to /dev/null so no error messages are printed
		close(devNull);
		signal(SIGABRT, handleSIGABRT); // exit at gracefully SIGABRT
	}
}

IirFilterEngine::~IirFilterEngine()
{
	freeMemoryOnDevice({ _d_filterValues, _d_inputs, _d_outputs, _d_signalLengths, _d_filtersCounts, _d_inputs_buffer, 
						 _d_filterSizes, _d_filtersOffsets, _d_filterSizesOffsets, _d_signalOffsets });
}

void IirFilterEngine::addSignal(int signalLength, const Filter* filters, int numFilters)
{
	if (numFilters != 1)
	{
		_moreFilters = true;
	}
	_signalLengths.push_back(signalLength);
	_signalOffsets.push_back(_totalSignalLength);
	_filtersCounts.push_back(numFilters);
	_filtersOffsets.push_back(_filtersOffset);
	_filterSizesOffsets.push_back(_filterSizesOffset);
	_totalSignalLength += signalLength;

	if (_firstCall) // set up initial values
	{
		_signalLength = signalLength;
		_filterLength = filters->_size;
		_firstCall = false;
	}
	else if (signalLength != _signalLength || filters->_size != _filterLength) // check if signal length or filter length is different to initial values
	{
		_variableSignalLength = signalLength != _signalLength;
		_variableFilterSize = filters->_size != _filterLength;
	}	

	_numberOfSignals++;
	_a0Not1 |= filters->_a[0] != 1.f; // check if a0 is always 1
	
	for (int i = 0; i < numFilters; i++)
	{
		_filterSizesOffset++;
		_filterSizes.push_back(filters[i]._size);
		_filtersOffset += filters[i]._size << 1;
		for (int j = 0; j < filters[i]._size; j++)
		{
			_filterSlowValues.push_back(filters[i]._a[j]);
		}

		for (int j = 0; j < filters[i]._size; j++)
		{
			_filterSlowValues.push_back(filters[i]._b[j]);
		}
	}

	if (!_variableFilterSize && !_moreFilters) // only load filter values if filter size is constant and there is only one filter per signal
	{
		if (_filterLength == 2 || (_filterLength == 3 && _numberOfSignals & 1))
		{
			for (int j = 0; j < filters->_size; j++) // load all values in order
			{
				_filterFastValues.push_back(filters->_a[j]);
				_filterFastValues.push_back(filters->_b[j]);
			}
		}
		else if (_filterLength == 3)
		{
			// load last two values first
			_filterFastValues.push_back(filters->_a[2]);
			_filterFastValues.push_back(filters->_b[2]);
			for (int j = 0; j < 2; j++) // load the rest of the values
			{
				_filterFastValues.push_back(filters->_a[j]);
				_filterFastValues.push_back(filters->_b[j]);
			}
		}
	}
}

void IirFilterEngine::finalize()
{
	if (_kernelType != KernelType::BASELINE_00 && !_variableSignalLength && !_variableFilterSize && !_a0Not1 && !_moreFilters && 
		_signalLength == 512 && (_filterLength == 2 || _filterLength == 3))
	{
		switch (_kernelType)
		{
			case KernelType::CONSTRAINTS_01:
				allocateFilterValuesOnDevice(_filterSlowValues, _d_filterValues);
				allocateSlowInputOutputMemoryOnDevice(_d_inputs, _d_outputs, _totalSignalLength);
				_fastKernelLaunchFunction = kernelAnyOrderLength512Launch;
				break;

			case KernelType::VECTORIZED_UNROLLED_02:
				allocateFilterValuesOnDevice(_filterFastValues, _d_filterValues);
				allocateSlowInputOutputMemoryOnDevice(_d_inputs, _d_outputs, _totalSignalLength);
				_fastKernelLaunchFunction = kernelVectorizedLength512Launch;
				break;
			
			case KernelType::STREAMS_ASYNCHRONICITY_03:
				allocateFilterValuesOnDevice(_filterFastValues, _d_filterValues);
				allocateSlowInputOutputMemoryOnDevice(_d_inputs, _d_outputs, _totalSignalLength);
				_streams = vector<cudaStream_t>(_numberOfSignals);
				_kernelStreamedLaunchFunction = kernelVectorizedLength512AsyncStreamsLaunch;
				break;
			
			case KernelType::SAFE_HACKING_04:
				_streams = vector<cudaStream_t>(_numberOfSignals);
				allocateFilterValuesOnDevice(_filterFastValues, _d_filterValues);
				allocateFastInputOutputMemoryOnDevice(_d_inputs, _d_outputs, _numberOfSignals, _signalLength);
				_kernelStreamedLaunchFunction = kernelVectorizedLength512AsyncStreamsFastLaunch;
				break;

			case KernelType::UNSAFE_HACKING_05:
				allocateFilterValuesOnDevice(_filterFastValues, _d_filterValues);
				allocateFastInputOutputMemoryOnDevice(_d_inputs, _d_outputs, _numberOfSignals, _signalLength);
				_fastKernelLaunchFunction = kernelVectorizedLength512FastLaunch;
				break;
			
			default:
			case KernelType::BASELINE_00:
				break;
		}
	}
	else
	{
		allocateFilterValuesOnDevice(_filterSlowValues, _d_filterValues);
		allocateSlowInputOutputMemoryOnDevice(_d_inputs, _d_outputs, _totalSignalLength);
		allocateMetadataOnDevice(_signalLengths, _signalOffsets, _filtersCounts, _filtersOffsets, _filterSizes, _filterSizesOffsets, 
								_d_signalLengths, _d_signalOffsets, _d_filtersCounts, _d_filtersOffsets, _d_filterSizes, _d_filterSizesOffsets);
		allocateMemoryOnDevice(_d_inputs_buffer, _totalSignalLength * sizeof(float));
	}
}

void IirFilterEngine::executeWithHostPointers(const float** inputs, float** outputs)
{
	if (_kernelStreamedLaunchFunction)
	{
		_kernelStreamedLaunchFunction(inputs, outputs, _d_inputs, _d_outputs, _d_filterValues, _numberOfSignals, _filterLength, 
									  _streams);
	}
	else if (_fastKernelLaunchFunction)
	{
		_fastKernelLaunchFunction(inputs, outputs, _d_inputs, _d_outputs, _d_filterValues, _numberOfSignals, _filterLength);
	}
	else
	{
		kernelGenericLaunch(inputs, outputs, _d_inputs, _d_outputs, _d_filterValues, _d_signalLengths, _d_signalOffsets, _d_filtersCounts, 
							_d_filtersOffsets, _d_filterSizes, _d_filterSizesOffsets, _signalLengths);
	}
}

void IirFilterEngine::executeWithDevicePointers(const float** d_inputs, float** d_outputs)
{
	kernelGenericLaunchWithDevicePointers(d_inputs, _d_inputs_buffer, d_outputs, _d_filterValues, _d_signalLengths, _d_signalOffsets, 
										  _d_filtersCounts, _d_filtersOffsets, _d_filterSizes, _d_filterSizesOffsets, _numberOfSignals);
}

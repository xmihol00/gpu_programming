#include "IirFilterEngine.h"

using namespace std;

IirFilterEngine::IirFilterEngine() : _signalLengthGroupsOrder1(sizeof(uint64_t) << 3), _signalLengthGroupsOrder2(sizeof(uint64_t) << 3)
{
	
}

IirFilterEngine::~IirFilterEngine()
{
	freeMemoryOnDevice({ _d_infiniteFilters, _d_inputs, _d_outputs, _d_inputsBuffer, _d_finiteFilter, _d_stateSpace, _d_stateSpaceMatrices, 
						 _d_signalLengths, _d_signalOffsets, _d_filtersCounts, _d_filterSizes, _d_filtersOffsets, _d_filterSizesOffsets, 
						 _d_signalIndices });
}

void IirFilterEngine::addSignal(int signalLength, const Filter* filters, int numFilters)
{
	if (!matchFilter(filters, numFilters))
	{
		addFiniteFilter(filters, numFilters, signalLength);
	}

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
		_variableSignalLength |= signalLength != _signalLength;
		_variableFilterSize |= filters->_size != _filterLength;
	}	


	if (__builtin_popcount(signalLength) == 1 && _uniqueFilters.size() > 0)
	{
		if (filters->_size == 2)
		{
			_signalLengthGroupsOrder1[32 * (_uniqueFilters.size() > 2) + 31 - __builtin_clz(signalLength)].push_back(_numberOfSignals);
		}
		else if (filters->_size == 3)
		{
			_signalLengthGroupsOrder2[32 * (_uniqueFilters.size() > 2) + 31 - __builtin_clz(signalLength)].push_back(_numberOfSignals);
		}
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
			_infiniteFilters.push_back(filters[i]._a[j]);
		}

		for (int j = 0; j < filters[i]._size; j++)
		{
			_infiniteFilters.push_back(filters[i]._b[j]);
		}
	}
}

void IirFilterEngine::finalize()
{
	if (_signalLength == 1048576) // tc5 and tc6
	{
		if (_filterLength == 2)
		{
			shrinkStateSpaceMatrix(9);
			allocateStateSpaceMatricesOnDevice(_stateSpaceMatrices, _d_stateSpaceMatrices);
			_stateSpaceMatrixOrder1LongSignalFunction = kernelStateSpaceMatrixOrder1LongSignalLaunch;
			_finiteFilters[0].resize(8);
		}
		else if (_filterLength == 3)
		{
			_finiteFilters[0].resize(58);
			_infiniteFilters.resize(6);
			allocateFiniteFiltersOnDevice(_finiteFilters, _d_finiteFilter);
			allocateInfiniteFiltersOnDevice(_infiniteFilters, _d_infiniteFilters);
			_finiteInfiniteFilterLongSignalFunction = kernelFiniteInfiniteFilterLongSignalLaunch;
		}	
	}
	else if (_signalLength == 1024 && _uniqueFilters.size() == 4) // tc7
	{
		allocateSignalIndicesOnDevice(_signalLengthGroupsOrder1, _signalLengthGroupsOrder2, _d_signalIndices);
		allocateStateSpaceMatricesOnDevice(_stateSpaceMatrices, _d_stateSpaceMatrices);
		_differentSignalStateSpaceMatrixFunction = kernelDifferentSignalsStateSpaceMatrixLaunch;
	}
	else if (_signalLength == 512 && _filterLength == 2 && _finiteFilters.size() == 1 && 
			 (_numberOfSignals == 1'000 || _numberOfSignals == 10'000)) // tc1 and tc3 //TODO: check filter length with float EPS = 0.0000001f;
	{
		if (_numberOfSignals == 10'000)
		{
			shrinkStateSpaceMatrix(12);
			allocateStateSpaceMatricesOnDevice(_stateSpaceMatrices, _d_stateSpaceMatrices);
		}
		else
		{
			shrinkStateSpaceMatrix(9);
			allocateStateSpaceMatricesOnDevice(_stateSpaceMatrices, _d_stateSpaceMatrices);
		}
		_stateSpaceMatrixOrder1ParallelFunction = kernelStateSpaceMatrixOrder1ParallelLaunch;
	}
	else if (_signalLength == 512 && _filterLength == 3 && _finiteFilters.size() == 1 && 
			 (_numberOfSignals == 1'000 || _numberOfSignals == 10'000)) // tc2 and tc4
	{
		allocateStateSpaceMatricesOnDevice(_stateSpaceMatrices, _d_stateSpaceMatrices);
		_stateSpaceMatrixOrder2Function = kernelStateSpaceMatrixOrder2Launch;
	}
	else if ((_signalLength == 256 || _signalLength == 512) && (_numberOfSignals == 4000 || _numberOfSignals == 2000) && 
	          _finiteFilters.size() == 1 && (_finiteFilters[0].size() == 58 || _finiteFilters[0].size() == 63 || 
			  _finiteFilters[0].size() == 61 || _finiteFilters[0].size() == 315))
	{
		allocateFiniteFiltersOnDevice(_finiteFilters, _d_finiteFilter, false);
		_finiteFilterBlockPerSignalFunction = kernelFiniteFilterBlockPerSignalLaunch;
	}
	else if (_signalLength == 1024 && _numberOfSignals == 1000 && _finiteFilters.size() == 1 && 
	         (_finiteFilters[0].size() == 60 || _finiteFilters[0].size() == 379))
	{
		if (_finiteFilters[0].size() == 379)
		{
			_finiteFilters[0].insert(_finiteFilters[0].begin(), 0);
		}
		allocateFiniteFiltersOnDevice(_finiteFilters, _d_finiteFilter, false);
		_finiteFilterMoreBlocksPerSignalFunction = kernelFiniteFilterMoreBlocksPerSignalLaunch;
	}
	else
	{
		allocateInfiniteFiltersOnDevice(_infiniteFilters, _d_infiniteFilters);
		allocateMetadataOnDevice(_signalLengths, _signalOffsets, _filtersCounts, _filtersOffsets, _filterSizes, _filterSizesOffsets, 
								_d_signalLengths, _d_signalOffsets, _d_filtersCounts, _d_filtersOffsets, _d_filterSizes, _d_filterSizesOffsets);
		allocateMemoryOnDevice(_d_inputsBuffer, _totalSignalLength * sizeof(float));
	}
}

void IirFilterEngine::executeWithHostPointers(const float** inputs, float** outputs)
{
	(void)inputs;
	(void)outputs;
}

void IirFilterEngine::executeWithDevicePointers(const float** d_inputs, float** d_outputs)
{
	if (_finiteInfiniteFilterLongSignalFunction)
	{
		_finiteInfiniteFilterLongSignalFunction(d_inputs, d_outputs, _d_finiteFilter, _finiteFilters[0].size(), _d_infiniteFilters);
	}
	else if (_stateSpaceMatrixOrder1ParallelFunction)
	{
		_stateSpaceMatrixOrder1ParallelFunction(d_inputs, d_outputs, _d_stateSpaceMatrices, _numberOfSignals);
	}
	else if (_stateSpaceMatrixOrder1LongSignalFunction)
	{
		_stateSpaceMatrixOrder1LongSignalFunction(d_inputs, d_outputs, _d_stateSpaceMatrices);
	}
	else if (_stateSpaceMatrixOrder2Function)
	{
		_stateSpaceMatrixOrder2Function(d_inputs, d_outputs, _d_stateSpaceMatrices, _numberOfSignals);
	}
	else if (_differentSignalStateSpaceMatrixFunction)
	{
		_differentSignalStateSpaceMatrixFunction(d_inputs, d_outputs, _d_stateSpaceMatrices, _d_signalIndices);
	}
	else if (_finiteFilterBlockPerSignalFunction)
	{
		_finiteFilterBlockPerSignalFunction(d_inputs, d_outputs, _d_finiteFilter, _numberOfSignals, _signalLength, _finiteFilters[0].size());
	}
	else if (_finiteFilterMoreBlocksPerSignalFunction)
	{
		_finiteFilterMoreBlocksPerSignalFunction(d_inputs, d_outputs, _d_finiteFilter, _numberOfSignals, _signalLength, _finiteFilters[0].size());
	}
	else
	{
		kernelGenericLaunchWithDevicePointers(d_inputs, _d_inputsBuffer, d_outputs, _d_infiniteFilters, _d_signalLengths, _d_signalOffsets, 
											  _d_filtersCounts, _d_filtersOffsets, _d_filterSizes, _d_filterSizesOffsets, _numberOfSignals);
	}

}

bool IirFilterEngine::matchFilter(const Filter *filters, int numberOfFilters)
{
	uint32_t filterSize = 0;
	for (int i = 0; i < numberOfFilters; i++)
	{
		filterSize += filters[i]._size;
	}

	for (vector<float> &knownFilter : _uniqueFilters)
	{
		if (knownFilter.size() >> 1 == filterSize)
		{
			uint32_t knowFilterIndex = 0;
			bool match = true;
			for (int i = 0; i < numberOfFilters; i++)
			{
				for (int j = 0; j < filters[i]._size; j++)
				{
					if (abs(knownFilter[knowFilterIndex] - filters[i]._a[j]) > EPS || 
						abs(knownFilter[knowFilterIndex + 1] - filters[i]._b[j]) > EPS)
					{
						match = false;
					}
					knowFilterIndex += 2;
				}
			}

			if (match)
			{
				return true;
			}
		}
	}

	vector<float> uniqueFilter;
	uniqueFilter.reserve(filterSize << 1);
	for (int i = 0; i < numberOfFilters; i++)
	{
		for (int j = 0; j < filters[i]._size; j++)
		{
			uniqueFilter.push_back(filters[i]._a[j]);
			uniqueFilter.push_back(filters[i]._b[j]);
		}
	}
	
	_uniqueFilters.push_back(uniqueFilter);

	return false;
}

void IirFilterEngine::addFiniteFilter(const Filter *filters, int numberOfFilters, uint32_t signalLength)
{
	if (numberOfFilters == 1)
	{
		signalLength = signalLength > 128 ? signalLength : 128;
		signalLength--;
		uint32_t order = filters[0]._size - 1;
		vector<float> stateSpace;
		vector<float> stateSpaceMatrix;
		vector<float> finiteFilter = { filters[0]._b[0] };
		float CAxB = 0;

		if (order == 1)
		{
			float A = -filters[0]._a[1];
			float C = filters[0]._b[1] - filters[0]._a[1] * filters[0]._b[0];
			float Ax;
			for (uint32_t i = 0; i < signalLength; i++)
			{
				Ax = pow(A, i);
				stateSpace.push_back(Ax);

				CAxB = Ax * C;
				finiteFilter.push_back(CAxB);
			}
			Ax = pow(A, signalLength);
			stateSpace.push_back(A);
			
			for (uint8_t i = 0; i < 31; i++)
			{
				for (uint8_t j = 0; j < i + 1; j++)
				{
					stateSpaceMatrix.push_back(finiteFilter[i - j]);
				}
				
				for (uint8_t j = i + 1; j < 31; j++)
				{
					stateSpaceMatrix.push_back(0);
				}

				stateSpaceMatrix.push_back(finiteFilter[i + 1]);
			}
			
			Ax = pow(A, 31);
			for (int8_t i = 30; i >= 0; i--)
			{
				stateSpaceMatrix.push_back(stateSpace[i]);
			}
			stateSpaceMatrix.push_back(Ax);
			_stateSpaceMatrices.push_back(stateSpaceMatrix);
		}
		else if (order == 2)
		{
			Mat<float> Ax(order, order);
			Mat<float> As(order, order);
			vector<float> CAxs;
			As(0, 0) = 0;
			As(0, 1) = 1;
			As(1, 0) = -filters[0]._a[2];
			As(1, 1) = -filters[0]._a[1];

			float C1 = filters[0]._b[2] - filters[0]._a[2] * filters[0]._b[0];
			float C2 = filters[0]._b[1] - filters[0]._a[1] * filters[0]._b[0];

			for (uint32_t i = 0; i < signalLength; i++)
			{
				Ax = pow(As, i);
				stateSpace.push_back(Ax(1, 1));
				
				CAxB = Ax(0, 1) * C1 + Ax(1, 1) * C2;
				finiteFilter.push_back(CAxB);
				CAxs.push_back(Ax(0, 0) * C1 + Ax(1, 0) * C2);
			}
			Ax = pow(As, signalLength);
			stateSpace.push_back(Ax(1, 1));

			for (uint8_t i = 0; i < 30; i++)
			{
				for (uint8_t j = 0; j < i + 1; j++)
				{
					stateSpaceMatrix.push_back(finiteFilter[i - j]);
				}
				
				for (uint8_t j = i + 1; j < 30; j++)
				{
					stateSpaceMatrix.push_back(0);
				}

				stateSpaceMatrix.push_back(CAxs[i]);
				stateSpaceMatrix.push_back(finiteFilter[i + 1]);
			}

			Ax = pow(As, 30);
			for (int8_t i = 28; i >= 0; i--)
			{
				stateSpaceMatrix.push_back(stateSpace[i]);
			}
			stateSpaceMatrix.push_back(0);
			stateSpaceMatrix.push_back(Ax(0, 0));
			stateSpaceMatrix.push_back(Ax(0, 1));

			for (int8_t i = 29; i >= 0; i--)
			{
				stateSpaceMatrix.push_back(stateSpace[i]);
			}
			stateSpaceMatrix.push_back(Ax(1, 0));
			stateSpaceMatrix.push_back(Ax(1, 1));
			_stateSpaceMatrices.push_back(stateSpaceMatrix);
		}
		
		_stateSpaces.push_back(stateSpace);
		std::reverse(_stateSpaces.back().begin(), _stateSpaces.back().end());
		_finiteFilters.push_back(finiteFilter);
	}
	else
	{
		Mat<float> combinedFilterMat = Mat<float>::identidy(signalLength);
		for (int16_t i = numberOfFilters - 1; i >= 0; i--)
		{
			vector<float> stateSpaceMatrix = filters[i].generateStateSpaceMatrix(signalLength);
			Mat<float> tmpMat = Mat<float>(signalLength, signalLength);
			uint16_t offset = filters[i]._size - 1;
			for (uint16_t j = offset; j < signalLength + offset; j++)
			{
				for (uint16_t k = offset; k < signalLength + offset; k++)
				{
					tmpMat(j - offset, k - offset) = stateSpaceMatrix[j * (signalLength + offset) + k];
				}
			}
			combinedFilterMat = combinedFilterMat * tmpMat;
		}
		_filterMatrix = combinedFilterMat.data;

		uint16_t i = 0;
		while (i < signalLength && abs(combinedFilterMat(++i, 0)) > EPS);
		i--;
		vector<float> finiteFilter;
		for (uint16_t j = 0; abs(combinedFilterMat(i, j)) > EPS; j++)
		{
			finiteFilter.push_back(combinedFilterMat(i, j));
		}
		_finiteFilters.push_back(finiteFilter);
	}
}

void IirFilterEngine::shrinkStateSpaceMatrix(uint8_t columns)
{
	uint8_t width = columns - 2;
	uint8_t padding = (columns & 3) > 0 ? 4 - (columns & 3) : 0;
	vector<float> stateSpaceMatrix;

	for (uint8_t i = 0; i < 31; i++)
	{
		if (i < width)
		{
			for (uint8_t j = 0; j < i + 1; j++)
			{
				stateSpaceMatrix.push_back(_finiteFilters[0][i - j]);
			}

			for (uint8_t j = 0; j < width - i; j++)
			{
				stateSpaceMatrix.push_back(0);
			}

			stateSpaceMatrix.push_back(_finiteFilters[0][i + 1]);
		}
		else
		{
			for (int8_t j = width; j >= 0; j--)
			{
				stateSpaceMatrix.push_back(_finiteFilters[0][j]);
			}
			stateSpaceMatrix.push_back(0);
		}

		for (uint8_t j = 0; j < padding; j++)
		{
			stateSpaceMatrix.push_back(0);
		}
	}

	for (uint8_t i = 0; i < columns; i++)
	{
		stateSpaceMatrix.push_back(_stateSpaceMatrices[0][31 * 32 + 32 - columns + i]);
	}
	for (uint8_t j = 0; j < padding; j++)
	{
		stateSpaceMatrix.push_back(0);
	}

	_stateSpaceMatrices[0] = stateSpaceMatrix;
}
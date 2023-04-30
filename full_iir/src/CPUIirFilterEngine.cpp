#include "CPUIirFilterEngine.h"

#include <type_traits>

namespace {

	template<typename T1, typename T2>
	std::common_type_t<T1, T2> dot(const std::vector<T1>& a, const std::vector<T2>& b)
	{
		std::common_type_t<T1, T2> res = 0;
		auto b_it = begin(b);
		for (auto a_val : a)
			res += a_val * *b_it++;
		return res;
	}

	template<typename T1, typename T2, typename T3>
	void matMul(std::vector<T3>& res, size_t M, const std::vector<T1>& A, const std::vector<T2>& b)
	{
		for (size_t r = 0; r < M; ++r)
		{
			std::common_type_t<T1, T2> acc = 0;
			for (size_t c = 0; c < M; ++c)
				acc += A[r * M + c] * b[c];
			res[r] = acc;
		}
	}

	template<typename T1, typename T2, typename T3>
	void mad(std::vector<T1>& a, const std::vector<T2>& b, T3 c)
	{
		auto b_it = begin(b);
		for (auto& a_ref : a)
			a_ref += *b_it++ * c;
	}
}



template<typename T>
template<typename T2>
void CpuIirFilterEngine<T>::addSignal(int signalLength, const GenericFilter<T2>* filters, int numFilters)
{
	// store the setup
	_filterSetup.emplace_back(SignalFilterSetup{ signalLength, std::vector<FilterImplementation>{} });
	auto& newSignalFilterSetup = _filterSetup.back()._filterImplementation;
	for (int i = 0; i < numFilters; ++i)
	{
		newSignalFilterSetup.emplace_back(FilterImplementation{ *filters++, {}, {}, {}, 0, {} });
		auto& filterImpl = newSignalFilterSetup.back();
		if (_mode == 0)
			filterImpl.prepareStateSpace();
		else if(_mode > 0)
			filterImpl.prepareParallelMatrix(_mode);
	}
	_maxSignalLength = std::max(_maxSignalLength, signalLength);
	
}

template<typename T>
CpuIirFilterEngine<T>::CpuIirFilterEngine(Mode mode, int parallelOutputs) : 
	_mode{ mode == Mode::Traditional ? -1 : mode == Mode::StateSpace ? 0 : parallelOutputs }
{
}

template<typename T>
void CpuIirFilterEngine<T>::finalize()
{
	_tempBuffer[0].resize(_maxSignalLength);
	_tempBuffer[1].resize(_maxSignalLength);
}

template<typename T>
template<typename T2, typename T3>
void CpuIirFilterEngine<T>::executeWithHostPointers(const T2** inputs, T3** outputs)
{

	for (int signal = 0; signal < static_cast<int>(_filterSetup.size()); ++signal)
	{
		const auto& signalFilterSetup = _filterSetup[signal];
		const T2* input = inputs[signal];
		T3* output = outputs[signal];
		int length = signalFilterSetup._signalLength;

		if (signalFilterSetup._filterImplementation.size() == 1)
			runFilterImplementation(signalFilterSetup._filterImplementation[0], length, input, output);
		else
		{
			runFilterImplementation(signalFilterSetup._filterImplementation.front(), length, input, _tempBuffer[0].data());
			for (int cascade = 1; cascade < static_cast<int>(signalFilterSetup._filterImplementation.size()-1); ++cascade)
			{
				runFilterImplementation(signalFilterSetup._filterImplementation[cascade], length, _tempBuffer[0].data(), _tempBuffer[1].data());
				std::swap(_tempBuffer[0], _tempBuffer[1]);
			}
			runFilterImplementation(signalFilterSetup._filterImplementation.back(), length, _tempBuffer[0].data(), output);
		}
	}
}

template<typename T>
template<typename T2, typename T3>
void CpuIirFilterEngine<T>::runFilterImplementation(const FilterImplementation& filter, int length, const T2* input, T3* output) const
{
	if (_mode == -1)
		filter.runTraditional(length, input, output);
	else if (_mode == 0)
		filter.runStateSpace(length, input, output);
	else
		filter.runParallelMatrix(length, input, output, _mode);
}


template<typename T>
template<typename T2, typename T3>
void CpuIirFilterEngine<T>::FilterImplementation::runTraditional(int length, const T2* input, T3* output) const
{
	for (int n = 0; n < length; ++n)
	{
		T y = 0;
		for (int i = 0; i < _filter._size; ++i)
		{
			if (n - i >= 0)
				y += _filter._b[i] * input[n - i];
		}
		for (int i = 1; i < _filter._size; ++i)
		{
			if (n - i >= 0)
				y += -_filter._a[i] * output[n - i];
		}
		if (_filter._a[0] != static_cast<T>(1))
			y /= _filter._a[0];
		output[n] = static_cast<T3>(y);
	}
}

template<typename T>
void CpuIirFilterEngine<T>::FilterImplementation::prepareStateSpace()
{
	_filter.generateStateSpace(_A, _B, _C, _D);
}

template<typename T>
template<typename T2, typename T3>
void CpuIirFilterEngine<T>::FilterImplementation::runStateSpace(int length, const T2* input, T3* output) const
{
	// initialize state
	size_t M = static_cast<size_t>(_filter._size) - 1;
	std::vector<T> x(M, 0), x_temp(M, 0);

	for (int n = 0; n < length; ++n)
	{
		T2 u = input[n];

		// y = C * x + D * u
		T y = dot(_C, x) + _D * u;

		// x_{i+1} = A * x_i + B * u 
		matMul(x_temp, M, _A, x);
		mad(x_temp, _B, u);

		std::swap(x, x_temp);
		output[n] = static_cast<T3>(y);
	}
}

template<typename T>
void CpuIirFilterEngine<T>::FilterImplementation::prepareParallelMatrix(int outputs)
{
	_parallelMatrix = _filter.generateStateSpaceMatrix(outputs);
}

template<typename T>
template<typename T2, typename T3>
void CpuIirFilterEngine<T>::FilterImplementation::runParallelMatrix(int length, const T2* input, T3* output, int outputs) const
{
	size_t M = static_cast<size_t>(_filter._size) - 1;
	size_t MM = M + outputs;
	std::vector<T> x(MM, 0), x_temp(MM, 0);
	for (int n = 0; n < length; n+=outputs)
	{
		for (int i = 0; i < outputs; ++i)
		{
			if (n + i < length)
				x[M + i] = input[n + i];
			else
				x[M + i] = 0;
		}

		matMul(x_temp, MM, _parallelMatrix, x);

		for (int i = 0; i < outputs; ++i)
		{
			if (n + i < length)
				output[n + i] = static_cast<T3>(x_temp[M + i]);
		}
		std::swap(x, x_temp);
	}

}




template void CpuIirFilterEngine<float>::executeWithHostPointers(const float** inputs, float** outputs);
template void CpuIirFilterEngine<float>::addSignal(int signalLength, const GenericFilter<float>* filters, int numFilters);

template void CpuIirFilterEngine<double>::executeWithHostPointers(const float** inputs, float** outputs);
template void CpuIirFilterEngine<double>::executeWithHostPointers(const double** inputs, float** outputs);
template void CpuIirFilterEngine<double>::executeWithHostPointers(const float** inputs, double** outputs);
template void CpuIirFilterEngine<double>::executeWithHostPointers(const double** inputs, double** outputs);
template void CpuIirFilterEngine<double>::addSignal(int signalLength, const GenericFilter<float>* filters, int numFilters);
template void CpuIirFilterEngine<double>::addSignal(int signalLength, const GenericFilter<double>* filters, int numFilters);

template class CpuIirFilterEngine<float>;
template class CpuIirFilterEngine<double>;
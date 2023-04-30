#pragma once

#include "filter.h"
#include <string>

/// <summary>
/// CpuIirFIlterEngine implements a smple CPU IIR filter engine
/// it supports filtering of an arbitrary number of signals,
/// each signal may be of different length
/// filters may be of order one or two
/// each signal may be filtered by an arbitrary cascade of filters
/// </summary>

template<typename T>
class CpuIirFilterEngine
{
	
	struct FilterImplementation
	{
		GenericFilter<T> _filter;
		std::vector<T> _A, _B, _C;
		T _D;
		std::vector<T> _parallelMatrix;
		
		template<typename T2, typename T3>
		void runTraditional(int length, const T2* input, T3* output) const;
		void prepareStateSpace();
		template<typename T2, typename T3>
		void runStateSpace(int length, const T2* input, T3* output) const;
		void prepareParallelMatrix(int outputs);
		template<typename T2, typename T3>
		void runParallelMatrix(int length, const T2* input, T3* output, int outputs) const;
	};
	struct SignalFilterSetup
	{
		int _signalLength;
		std::vector<FilterImplementation> _filterImplementation;
	};

	std::vector<SignalFilterSetup> _filterSetup;

	const int _mode{ -1 };
	int _maxSignalLength{ 0 };
	std::vector<T> _tempBuffer[2];

	template<typename T2, typename T3>
	void runFilterImplementation(const FilterImplementation& filter, int length, const T2* input, T3* output) const;
public:

	/// <summary>
	/// Enum to select among the supported modes
	/// </summary>
	enum class Mode : int
	{
		Traditional = 1,
		StateSpace = 2,
		ParallelMatrix = 3
	};

	/// <summary>
	/// Constructor that accepts a mode descrition
	/// </summary>
	/// <param name="mode">either of the supported modes</param>
	/// <param name="parallelOutputs">parallel outputs, only for ParallelMatrix mode</param>
	CpuIirFilterEngine(Mode mode = Mode::Traditional, int parallelOutputs = 1);


	/// <summary>
	/// add a signal for processing. the method is called during preparation 
	/// and before finalize it being called.
	/// </summary>
	/// <param name="signalLength">the number of samples of the signal</param>
	/// <param name="filters">a pointer to a cascade of filters</param>
	/// <param name="numFilters">the number of filters in the cascade</param>
	template<typename T2>
	void addSignal(int signalLength, const GenericFilter<T2>* filters, int numFilters = 1);

	/// <summary>
	/// called when all signals have been setup and guaranteed to be only called once
	/// </summary>
	void finalize();

	/// <summary>
	/// called to kick off the processing on the CPU 
	/// The buffers on the CPU are already allocated.
	/// The method may be called multiple times with different inputs.
	/// </summary>
	/// <param name="inputs">array of pointers to the input signals in the order addSignal was called</param>
	/// <param name="outputs">array of pointers to where the out signals should be placed;
	///  in the order addSignal was called</param>
	template<typename T2, typename T3>
	void executeWithHostPointers(const T2** inputs, T3** outputs);


	/// <summary>
	/// name encoding the method
	/// </summary>
	/// <returns>the name</returns>
	std::string name() const
	{
		if (_mode < 0)
			return "CPU";
		if (_mode == 0)
			return "CPUStateSpace";
		else
			return std::string("CPUParallelMatrix") + std::to_string(_mode);
	}

};



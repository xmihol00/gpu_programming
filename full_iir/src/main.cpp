#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <filesystem>

#include "CPUTimer.h"
#include "GPUTimer.cuh"

#include "ConfigLoader.h"
#include "IirFilterEngine.h"
#include "CPUIirFilterEngine.h"

struct HostSignals
{
	std::vector<const float*> _inputs;
	std::vector<float*> _outputs;
	std::vector<Signal> _outputSignals;
	HostSignals() = default;
	HostSignals(ConfigLoader& loader)
	{
		int num = loader.signals();
		_inputs.reserve(num);
		_outputs.reserve(num);
		_outputSignals.reserve(num);
		for (int i = 0; i < num; ++i)
		{
			auto& s = loader.getSignal(i);
			_inputs.push_back(s.get());
			_outputSignals.emplace_back(Signal(s.length(), 0.0f));
			_outputs.push_back(_outputSignals.back().get());
		}
	}

	void transferFromCPU(){ }
	void transferToCPU(){ }

	const float** inputPointers()
	{
		return _inputs.data();
	}
	float** outputPointers()
	{
		return _outputs.data();
	}
};



struct DeviceSignals 
{
	std::vector<const float*> _inputs;
	std::vector<Signal> _outputSignals;
	std::vector<uint32_t> _signalLengths;

	float **_d_inputs = nullptr;
	float **_d_outputs = nullptr;
	float** h_inputPtrs;
	float** h_outputPtrs;

	DeviceSignals() = default;
	DeviceSignals(ConfigLoader & loader)
	{
		int num = loader.signals();
		_inputs.reserve(num);
		_outputSignals.reserve(num);
		for (int i = 0; i < num; ++i)
		{
			auto& s = loader.getSignal(i);
			_inputs.push_back(s.get());
			_outputSignals.emplace_back(Signal(s.length(), 0.0f));
			_signalLengths.push_back(s.length());
		}
		
		h_inputPtrs = new float*[_inputs.size()];
		h_outputPtrs = new float*[_inputs.size()];
		allocate2DArrayOnDevice(&_d_inputs, h_inputPtrs, _inputs.size(), _signalLengths[0]);
		allocate2DArrayOnDevice(&_d_outputs, h_outputPtrs, _inputs.size(), _signalLengths[0]);
	}

	~DeviceSignals()
	{
		free2DArrayOnDeviceMemory(_d_inputs, h_inputPtrs, _inputs.size());
		free2DArrayOnDeviceMemory(_d_outputs, h_outputPtrs, _inputs.size());
		delete[] h_inputPtrs;
		delete[] h_outputPtrs;
	}

	void transferFromCPU() 
	{
		copy2DArrayToDevice(h_inputPtrs, _inputs.data(), _inputs.size(), _signalLengths[0]);
	}

	void transferToCPU() 
	{
		for (size_t i = 0; i < _outputSignals.size(); i++)
		{
			float* signal = _outputSignals[i].get();
			copy2DArrayToHost(signal, h_outputPtrs, i, _signalLengths[i]);
		}
	}

	const float** inputPointers()
	{
		uint64_t address = reinterpret_cast<uint64_t>(_d_inputs);
		return reinterpret_cast<const float**>(address);
	}

	float** outputPointers()
	{
		return _d_outputs;
	}
};

template<bool RunFromHost>
struct SignalSelector;

template<>
struct SignalSelector<true>
{
	using Signals = HostSignals;
};
template<>
struct SignalSelector<false>
{
	using Signals = DeviceSignals;
};


template<bool CPUTiming, bool GPUTiming, bool RunFromHost, typename Engine>
void run(ConfigLoader& loader, Engine& testEngine, const std::string& name, const std::string& tc_name)
{
	typename SignalSelector<RunFromHost>::Signals signals(loader);

	CpuIirFilterEngine<double> referenceEngine;
	HostSignals referenceSignals;
	std::vector<const Signal*> comparisonSignals;
	if (!loader.hasCompareSignals() && loader.compare())
	{
		referenceSignals = HostSignals{ loader };
		loader.setupFilters(referenceEngine);
		referenceEngine.finalize();
		for (int i = 0; i < static_cast<int>(referenceSignals._inputs.size()); ++i)
			comparisonSignals.push_back(&referenceSignals._outputSignals[i]);
	}
	else
	{
		comparisonSignals.reserve(signals._inputs.size());
		for (int i = 0; i < static_cast<int>(signals._inputs.size()); ++i)
			comparisonSignals.push_back(loader.getOutputSignal(i));
	}

	loader.setupFilters(testEngine);
	testEngine.finalize();

	CPUTimer cputimer(loader.runs());
	GPUTimer gputimer(loader.runs());

	int failedRuns = 0;
	float maxMeanError = 0, maxMaxError = 0;
	float maxMean = loader.compareThreshold();
	float maxErr = loader.compareMaxThreshold();

	for (int run = -1; run < loader.runs(); ++run)
	{
		if (loader.prepareRun(run) || run == -1)
		{
			if constexpr (!RunFromHost)
				signals.transferFromCPU();
		}

		if(run >= 0)
		{ 
			if constexpr (CPUTiming)
				cputimer.start();
			if constexpr (GPUTiming)
				gputimer.start();
		}
		if constexpr (RunFromHost)
			testEngine.executeWithHostPointers(signals.inputPointers(), signals.outputPointers());
		else
			testEngine.executeWithDevicePointers(signals.inputPointers(), signals.outputPointers());
		if (run >= 0)
		{
			if constexpr (GPUTiming)
				gputimer.end();
			if constexpr (CPUTiming)
				cputimer.end();
		}

		if constexpr (!RunFromHost)
			signals.transferToCPU();

		if (!loader.hasCompareSignals() && loader.compare())
		{
			referenceEngine.executeWithHostPointers(referenceSignals._inputs.data(), referenceSignals._outputs.data());
		}

		if (!loader.output().empty())
		{
			auto f = loader.output() + "/" + name + "/";
			if (loader.changingSignals() && loader.runs() > 1)
				f = f + "run_" + std::to_string(run) + "/";
			if (!std::filesystem::exists(f))
				std::filesystem::create_directories(f);
			auto f_in = f + "input/";
			if (loader.changingSignals())
				if (!std::filesystem::exists(f_in))
					std::filesystem::create_directories(f_in);
			for (size_t s = 0; s < signals._outputSignals.size(); ++s)
			{
				signals._outputSignals[s].store(f + std::to_string(s) + ".signal");
				if (loader.changingSignals())
					loader.getSignal(s).store(f_in + std::to_string(s) + ".signal");
			}
		}

		if (loader.compare())
		{
			bool success = true;
			for (int s = 0; s < static_cast<int>(signals._outputSignals.size()); ++s)
			{
				auto [mean_error, mse, max_error] = signals._outputSignals[s].compare(comparisonSignals[s]->get());

				maxMeanError = std::max(mean_error, maxMeanError);
				maxMaxError = std::max(max_error, maxMaxError);
				if (!(mean_error <= maxMean && max_error <= maxErr))
				{
					success = false;
				}
			}
			if (!success && run >= 0)
				++failedRuns;
		}
	}

	auto cpures = cputimer.generateResult();
	auto gpures = gputimer.generateResult();
	if constexpr (CPUTiming && GPUTiming)
		std::cout << name << " required " << gpures.mean_ << "ms on average with std " << gpures.std_dev_ << "ms on the GPU (" << cpures.mean_ << " +/-" << cpures.std_dev_ << "ms on the CPU)" << std::endl;
	else if constexpr (CPUTiming)
		std::cout << name << " required " << cpures.mean_ << "ms on average with std " << cpures.std_dev_ << "ms on the CPU" << std::endl;
	else if constexpr (GPUTiming)
		std::cout << name << " required " << gpures.mean_ << "ms on average with std " << gpures.std_dev_ << "ms on the GPU" << std::endl;

	if (loader.compare())
	{
		if (failedRuns == 0)
			std::cout << "SUCCESS all runs achieved within the threshold";
		else
			std::cout << "FAILED " << failedRuns << "/" << loader.runs();

		std::cout << " with max mean error " << maxMeanError << " (<" << maxMean << ") and max max error " << maxMaxError << " (<" << maxErr << ")" << std::endl;
	}

	if constexpr (GPUTiming)
	{
		// Write to output file
		std::ofstream results_csv;
		results_csv.open("results.csv", std::ios_base::app);
		results_csv << tc_name << "," << gpures.mean_ << "," << (failedRuns==0 ? "1" : "0") << std::endl;
		results_csv.close();
	}


}

int main(int argc, char* argv[])
{
	std::cout << "Assignment 01 - IIR Filtering" << std::endl;

	if(argc != 2)
	{
		std::cout << "Usage: ./iir <config_file>" << std::endl;
		return -1;
	}

	try {

		ConfigLoader loader{ argv[1] };
		
		std::cout << "test case " << loader.name() << std::endl;

		if (loader.createReference())
		{
			HostSignals signals(loader);
			CpuIirFilterEngine<double> engine;
			loader.setupFilters(engine);
			engine.finalize();

			engine.executeWithHostPointers(signals._inputs.data(), signals._outputs.data());

			auto f = loader.output() + "/reference/";
			if (!std::filesystem::exists(f))
				std::filesystem::create_directories(f);
			auto f_in = f + "input/";
			if (loader.changingSignals())
				if (!std::filesystem::exists(f_in))
					std::filesystem::create_directories(f_in);
			for (size_t i = 0; i < signals._outputSignals.size(); ++i)
			{
				signals._outputSignals[i].store(f + std::to_string(i) + ".signal");
				if (loader.changingSignals())
					loader.getSignal(i).store(f_in + std::to_string(i) + ".signal");
			}
		}

		if (loader.runCPU())
		{
			std::cout << "\n--- CPU runs ---" << std::endl;
			auto [mode, outputs] = loader.getCPUMode();
			CpuIirFilterEngine<float> cpuEngine(static_cast<CpuIirFilterEngine<float>::Mode>(mode), outputs);
			run<true, false, true>(loader, cpuEngine, cpuEngine.name(), loader.name());
		}
		if (loader.runGPU())
		{
			std::cout << "\n*** GPU runs ***" << std::endl;
			IirFilterEngine gpuEngine;
			if(loader.runWithDevicePointers())
				run<true, true, false>(loader, gpuEngine, "GPU", loader.name());
			else
				run<true, true, true>(loader, gpuEngine, "GPU", loader.name());
		}
	}
	catch (std::exception& ex)
	{
		std::cout << "Error: " << ex.what();
		return -1;
	}

	return 0;
}
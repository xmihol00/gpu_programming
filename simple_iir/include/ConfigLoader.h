#pragma once

#include "signal.h"
#include "filter.h"

#include <string>

class ConfigLoader
{
	using FilterSetup = std::tuple<int, std::vector<Filter>>;
	std::vector<FilterSetup> _filterSetup;

	struct SignalGenerator
	{
		bool generate{ false };
		int length{ 1024 };
		int min_pow_length = -1, max_pow_length = -1;
		float min_sin_offset{ 0 }, min_sin_freq{ 0.3141459f }, min_sin_ampl{ 0.5f }, min_noise_ampl{ 0 };
		float max_sin_offset{ 0 }, max_sin_freq{ 0.3141459f }, max_sin_ampl{ 0.5f }, max_noise_ampl{ 0 };
		float getSinOffset(float v) { return min_sin_offset + v * (max_sin_offset - min_sin_offset); }
		float getSinFreq(float v) { return min_sin_freq + v * (max_sin_freq - min_sin_freq); }
		float getSinAmpl(float v) { return min_sin_ampl + v * (max_sin_ampl - min_sin_ampl); }
		float getNoiseAmpl(float v) { return min_noise_ampl + v * (max_noise_ampl - min_noise_ampl); }
	};

	SignalGenerator _signalGenerator;
	std::vector<int> _genSignals;
	std::vector<Signal> _inputSignals;
	std::vector<Signal> _outputSignals;

	bool _runCPU{ false },
		_runGPU{ true },
		_runWithDevicePointers{ false },
		_compare{ false },
		_changingSignals{ false },
		_hasCompareSignals{ false },
		_createReference{ false },
		_regenerate{ false };
	int _signals, _runs{ 10 };

	float _compareThreshold{ 0.001f }, _compareMaxThreshold{ 0.01f };

	std::string _name, _folder, _output, _cpuMode;


	int loadSignalsFromFolder(const std::string& folder, std::vector<Signal>& signals);
	void prepareAll();
public:

	enum class CPUMode : int
	{
		None,
		Traditional = 1,
		StateSpace = 2,
		ParallelMatrix = 3
	};

	ConfigLoader(const std::string& config_file);
	bool runGPU() const { return _runGPU; }
	bool runCPU() const { return _runCPU; }
	bool runWithDevicePointers() const { return _runWithDevicePointers; }
	bool createReference() const { return _createReference; }

	std::tuple<CPUMode, int> getCPUMode() const;

	bool compare() const { return _compare; }

	bool hasCompareSignals() const { return _hasCompareSignals; }
	float compareThreshold() const { return _compareThreshold; }
	float compareMaxThreshold() const { return _compareMaxThreshold; }

	bool changingSignals()  const { return _changingSignals; }

	template<typename Engine>
	void setupFilters(Engine& engine);

	int runs() const { return _runs; }
	int signals() const { return _signals; }


	bool prepareRun(int run);
	const Signal& getSignal(size_t signal) const;
	const Signal* getOutputSignal(size_t signal) const;

	const std::string& name() { return _name; }
	const std::string& output() { return _output; }
	const std::string& folder() { return _folder; }
};



template<typename Engine>
inline void ConfigLoader::setupFilters(Engine& engine)
{
	for (const auto& f : _filterSetup)
	{
		engine.addSignal(std::get<0>(f), std::get<1>(f).data(), static_cast<int>(std::get<1>(f).size()));
	}
}

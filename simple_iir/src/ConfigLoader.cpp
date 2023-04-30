#include "ConfigLoader.h"

#include "json.hpp"
#include <random>
#include <iostream>
#include <filesystem>


namespace {
	Filter extractFilterCoefficients(const nlohmann::json& a, const nlohmann::json& b)
	{
		if (a.size() != b.size())
			throw std::runtime_error("Cannont convert filter description as 'a' and 'b' need to be of same size");
		return Filter([&](int i) {return a[i]; }, [&](int i) {return b[i]; }, static_cast<int>(a.size()));
	}
}

ConfigLoader::ConfigLoader(const std::string& config_file)
{
	// here goes the heavy lifting
	using json = nlohmann::json;

	if (!std::filesystem::exists(config_file.c_str()))
		throw std::runtime_error("Input config file does not exist (" + config_file + ").");

	std::ifstream configf;
	configf.exceptions(std::ifstream::badbit);
	configf.open(config_file.c_str());
	json j;
	configf >> j;

	if (j.contains("name"))
		_name = j["name"].get<std::string>();
	else
		_name = "unnamed";

	if (j.contains("folder"))
		_folder = j["folder"].get<std::string>();
	else
		_folder = "unnamed_test";

	if (j.contains("output") && !j["output"].is_null() && !j["output"].is_boolean())
		_output = _folder + "/" + j["output"].get<std::string>();

	if (j.contains("CPU") && !j["CPU"].is_null())
	{
		_runCPU = true;
		_cpuMode = j["CPU"].get<std::string>();
	}
	
	if (j.contains("createReference"))
		_createReference = j["createReference"].get<bool>();

	if (j.contains("GPU"))
		_runGPU = j["GPU"].get<bool>();

	if (j.contains("runWithDevicePointers"))
		_runWithDevicePointers = j["runWithDevicePointers"].get<bool>();

	if (j.contains("compare"))
		_compare = j["compare"].get<bool>();

	if (j.contains("compareThreshold"))
		_compareThreshold = j["compareThreshold"].get<float>();

	if (j.contains("compareMaxThreshold"))
		_compareMaxThreshold = j["compareMaxThreshold"].get<float>();

	if (j.contains("runs"))
		_runs = j["runs"].get<int>();


	bool generate = false;
	std::string signal_loader;
	if (j.contains("signals"))
	{
		auto jsig = j["signals"];
		
		if (jsig.contains("generate"))
			generate = jsig["generate"].get<bool>();

		if (jsig.contains("regenerate"))
			_regenerate = jsig["regenerate"].get<bool>();
		else
			_regenerate = generate;

		if(jsig.contains("folder"))
		{
			signal_loader = _folder + "/" + jsig["folder"].get<std::string>();
		}
		if (generate)
		{
			if (jsig.contains("length"))
			{
				if(jsig["length"].is_number_integer())
					_signalGenerator.length = jsig["length"].get<int>();
				else
				{
					std::string genString = jsig["length"].get<std::string>();
					if (genString.find_first_of("pow") == 0)
					{
						size_t to = genString.find_first_of("-");
						if (to != std::string::npos)
						{
							auto n = to - 3;
							_signalGenerator.min_pow_length = std::atoi(genString.substr(3, n).c_str());
							_signalGenerator.max_pow_length = std::atoi(genString.substr(to + 1).c_str());
							_signalGenerator.length = -1;
						}
						else
						{
							int pow = std::atoi(genString.substr(3).c_str());
							_signalGenerator.length = 1u << pow;
						}

					}
				}
			}
				
			if (jsig.contains("min_sin_freq"))
				_signalGenerator.min_sin_freq = jsig["min_sin_freq"].get<float>();
			if (jsig.contains("max_sin_freq"))
				_signalGenerator.max_sin_freq = jsig["max_sin_freq"].get<float>();
			if (jsig.contains("min_sin_offset"))
				_signalGenerator.min_sin_offset = jsig["min_sin_offset"].get<float>();
			if (jsig.contains("max_sin_offset"))
				_signalGenerator.max_sin_offset = jsig["max_sin_offset"].get<float>();
			if (jsig.contains("min_sin_ampl"))
				_signalGenerator.min_sin_ampl = jsig["min_sin_ampl"].get<float>();
			if (jsig.contains("max_sin_ampl"))
				_signalGenerator.max_sin_ampl = jsig["max_sin_ampl"].get<float>();
			if (jsig.contains("min_noise_ampl"))
				_signalGenerator.min_noise_ampl = jsig["min_noise_ampl"].get<float>();
			if (jsig.contains("max_noise_ampl"))
				_signalGenerator.max_noise_ampl = jsig["max_noise_ampl"].get<float>();
		}
	}
	
	if(!signal_loader.empty() || !generate)
	{
		if(signal_loader.empty())
			signal_loader = _folder + "signals/";

		std::cout << "trying to load signals from " << signal_loader << std::endl;

		int loaded = loadSignalsFromFolder(signal_loader, _inputSignals);

		std::cout << "loaded " << loaded << " signals with max id " << _inputSignals.size() - 1 << std::endl;
	}


	std::string reference_folder = _folder + "/reference/";
	if (j.contains("reference"))
		reference_folder = _folder + j["reference"].get<std::string>();

	int refs = loadSignalsFromFolder(signal_loader, _outputSignals);
	_hasCompareSignals = refs != 0;
	if (_hasCompareSignals)
	{
		std::cout << "loaded " << refs << " reference signals with max id " << _outputSignals.size() - 1 << std::endl;
	}


	// load the filters and ensure we have signals for the filters
	if (!j.contains("filter_cascades"))
		throw std::runtime_error("filter_cascades is required in config");

	for (auto& filter_signal : j["filter_cascades"])
	{
		if (!filter_signal.contains("filters"))
			throw std::runtime_error("Config Error: every filter_cascade needs a 'filters' entry");

		int duplicate = 1;
		int cas_duplicate = 1;

		if (filter_signal.contains("duplicate"))
			duplicate = filter_signal["duplicate"].get<int>();
		if (filter_signal.contains("cascade_duplicate"))
			cas_duplicate = filter_signal["cascade_duplicate"].get<int>();

		FilterSetup thisSetup;
		auto& filters = std::get<1>(thisSetup);


		for (auto& j_filter: filter_signal["filters"])
		{

			if (j_filter.is_array())
			{
				if(j_filter.size() != 2)
					throw std::runtime_error("Config Error: a 'filter' declared as an array requires 2 entries");
				filters.emplace_back(extractFilterCoefficients(j_filter[0], j_filter[1]));
			}
			else
			{
				filters.emplace_back(extractFilterCoefficients(j_filter["a"], j_filter["b"]));
			}
		}

		if (cas_duplicate > 1)
		{
			int toCopy = static_cast<int>(filters.size());
			filters.reserve(static_cast<size_t>(toCopy) * cas_duplicate);
			for (int d = 1; d < cas_duplicate; ++d)
			{
				for (int i = 0; i < toCopy; ++i)
					filters.push_back(filters[i]);
			}
		}

		//std::random_device randomDevice;
		std::mt19937 r(5489U); // run with default seed
		std::uniform_int_distribution<int> ld(_signalGenerator.min_pow_length, _signalGenerator.max_pow_length);

		for (int i = 0; i < duplicate; ++i)
		{
			_filterSetup.push_back(thisSetup);
			if (_inputSignals.size() >= _filterSetup.size() && !_inputSignals[_filterSetup.size() - 1].empty())
			{
				std::get<0>(_filterSetup.back()) = _inputSignals[_filterSetup.size() - 1].length();
			}
			else if (generate)
			{
				int length = _signalGenerator.length;
				if (length == -1)
					length = 1u << ld(r);
				std::get<0>(_filterSetup.back()) = length;
			}
			else
				throw std::runtime_error(std::string("do not have signal for filter ") + std::to_string(_filterSetup.size() - 1));
		}
	}
	if (_inputSignals.size() > _filterSetup.size())
		std::cout << "Warning more input signals loaded (" << _inputSignals.size() << ") than filters specified (" << _filterSetup.size() << "), ignoring excess signals!" << std::endl;
	
	_inputSignals.resize(_filterSetup.size());

	if(generate)
		for (int i = 0; i < static_cast<int>(_inputSignals.size()); ++i)
		{
			if (_inputSignals[i].empty())
				_genSignals.push_back(i);
		}


	_signals = static_cast<int>(_inputSignals.size());
	_changingSignals = !_genSignals.empty();
	if (_changingSignals && !generate)
		throw std::runtime_error("Need to generate signals but no generator specified");

	prepareAll();
}

std::tuple<ConfigLoader::CPUMode, int> ConfigLoader::getCPUMode() const
{
	if (!_runCPU)
		return { CPUMode::None, 0 };

	if(_cpuMode == "Traditional")
		return { CPUMode::Traditional, 0 };

	if (_cpuMode == "StateSpace")
		return { CPUMode::StateSpace, 0 };

	if (_cpuMode == "ParallelMatrix")
		return { CPUMode::ParallelMatrix, 1 };

	if (_cpuMode.rfind("ParallelMatrix", 0) == 0)
		return { CPUMode::ParallelMatrix, std::atoi(_cpuMode.substr(sizeof("ParallelMatrix")-1).c_str()) };
		
	
	return { CPUMode::None, 0 }; 
}

int ConfigLoader::loadSignalsFromFolder(const std::string& folder, std::vector<Signal>& signals)
{
	int loaded = 0;
	if (std::filesystem::exists(folder))
	{
		for (auto& f : std::filesystem::directory_iterator(folder))
		{
			if (f.path().extension() == ".signal")
			{
				auto stem = f.path().stem().generic_string();
				if (std::all_of(stem.begin(), stem.end(), ::isdigit))
				{
					auto s = std::atoll(stem.c_str());
					if ((decltype(s)) signals.size() <= s)
						signals.resize(s + 1);
					signals[s].load(f.path());
					//std::cout << "loaded " << f.path() << std::endl;
					++loaded;
				}
			}
		}
	}
	return loaded;
}

void ConfigLoader::prepareAll()
{
	std::random_device randomDevice;
	std::mt19937 r(randomDevice());
	std::uniform_real_distribution<float> d(0.0f, 1.0f); 
	auto& g = _signalGenerator;
	for (const auto& s : _genSignals)
	{
		int length = std::get<0>(_filterSetup[s]);
		_inputSignals[s].fill(length, g.getSinOffset(d(r)), g.getSinFreq(d(r)), g.getSinAmpl(d(r)), g.getNoiseAmpl(d(r)));
	}
		
}


bool ConfigLoader::prepareRun(int run)
{
	if (run > 0 && _changingSignals && _regenerate)
	{
		prepareAll();
		return true;
	}
	return false;
}

const Signal& ConfigLoader::getSignal(size_t signal) const
{
	return _inputSignals[signal];
}

const Signal* ConfigLoader::getOutputSignal(size_t signal) const
{
	if (_outputSignals.size() > signal)
		return &_outputSignals[signal];
	return nullptr;
}

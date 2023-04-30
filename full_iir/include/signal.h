#pragma once

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

class Signal
{
	std::vector<float> _data;

public:
	Signal() = default;

	Signal(const std::string& filename)
	{
		load(filename);
	}
	Signal(const std::filesystem::path& filename)
	{
		load(filename);
	}
	Signal(int length, float val)
	{
		_data.clear();
		_data.resize(length, val);
	}
	Signal(int length, float sin_offset, float sin_freq, float sin_ampl, float noise_ampl)
	{
		fill(length, sin_offset, sin_freq, sin_ampl, noise_ampl);
	}

	void fill(int length, float sin_offset, float sin_freq, float sin_ampl, float noise_ampl)
	{
		std::random_device randomDevice;
		std::mt19937 generator(randomDevice());
		std::normal_distribution<float> distribution(0.0, noise_ampl);

		_data.resize(length);
		for (int i = 0; i < length; ++i)
			_data[i] = sin_ampl * sinf(sin_offset + sin_freq * static_cast<float>(i)) + distribution(generator);
	}

	float* get() { return _data.data(); }
	const float* get() const { return _data.data(); }
	int length() const { return static_cast<int>(_data.size()); }
	bool empty() const { return _data.empty(); }

	void load(const std::filesystem::path& filename)
	{
		auto size = std::filesystem::file_size(filename);
		std::ifstream infile(filename.c_str(), std::ios_base::binary);
		_data.resize(size / sizeof(float));
		infile.read(reinterpret_cast<char*>(_data.data()), _data.size() * sizeof(float));
	}
	void store(const std::string& filename) const
	{
		std::ofstream outfile(filename.c_str(), std::ios_base::binary);
		outfile.write(reinterpret_cast<const char*>(_data.data()), _data.size() * sizeof(float));
	}
	std::tuple<float, float, float> compare(const float* other) const
	{
		float mse = 0;
		float mean_error = 0;
		float max_error = 0;
		for (auto f : _data)
		{
			float d = std::abs(f - *other++);
			mean_error += d;
			mse += d * d;
			max_error = std::max(d, max_error);
		}
		mean_error /= _data.size();
		mse /= _data.size();
		return { mean_error, mse, max_error };

	}
};
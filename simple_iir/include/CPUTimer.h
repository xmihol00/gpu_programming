#pragma once

#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Utility.cuh"

#include<vector>

struct CPUTimer
{
	decltype(std::chrono::steady_clock::now()) t0, t1;
	std::vector<float> measurements_;

	CPUTimer(int runs = 100) 
	{
		measurements_.reserve(runs);
	}
	void inline start()
	{
		t0 = std::chrono::steady_clock::now();
	}

	float inline end(bool add = true)
	{
		t1 = std::chrono::steady_clock::now();
		float t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000000.0f;
		if (add)
			measurements_.push_back(t);
		return t;
	}

	void addMeasure(CPUTimer& measure)
	{
		measurements_.insert(measurements_.end(), measure.measurements_.begin(), measure.measurements_.end());
	}

	float mean()
	{
		return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / static_cast<float>(measurements_.size());
	}

	float median()
	{
		std::vector<float> sorted_measurements(measurements_);
		std::sort(sorted_measurements.begin(), sorted_measurements.end());
		return sorted_measurements[sorted_measurements.size() / 2];
	}

	float std_dev(float mean)
	{
		std::vector<float> stdmean_measurements(measurements_);
		for (auto& elem : stdmean_measurements)
			elem = (elem - mean) * (elem - mean);
		return std::sqrt(std::accumulate(stdmean_measurements.begin(), stdmean_measurements.end(), 0.0f) / static_cast<float>(stdmean_measurements.size()));
	}

	void reset(int runs = 100)
	{
		measurements_.clear();
		measurements_.reserve(runs);
	}

	Result generateResult()
	{
		if (measurements_.size() == 0)
			return Result{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0 };
		float mean_val = mean();
		return Result{
			mean_val,
			std_dev(mean_val),
			median(),
			*std::min_element(measurements_.begin(), measurements_.end()),
			*std::max_element(measurements_.begin(), measurements_.end()),
			static_cast<int>(measurements_.size())
		};
	}
};

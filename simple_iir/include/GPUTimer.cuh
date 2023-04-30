#pragma once

#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Utility.cuh"


struct GPUTimer
{
	cudaEvent_t ce_start, ce_stop;
	std::vector<float> measurements_;

	void start() 
	{ 
		start_clock(ce_start);
	}

	float end(bool add = true) 
	{ 
		auto timing = end_clock(ce_start, ce_stop);
		if(add)
			measurements_.push_back(timing);
		return timing;
	}

	GPUTimer(int runs)
	{
		measurements_.reserve(runs);
		HANDLE_ERROR(cudaEventCreate(&ce_start));
		HANDLE_ERROR(cudaEventCreate(&ce_stop));

	}
	~GPUTimer()
	{
		HANDLE_ERROR(cudaEventDestroy(ce_start));
		HANDLE_ERROR(cudaEventDestroy(ce_stop));
	}
	void addMeasure(GPUTimer& measure)
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
		return sqrt(std::accumulate(stdmean_measurements.begin(), stdmean_measurements.end(), 0.0f) / static_cast<float>(stdmean_measurements.size()));
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
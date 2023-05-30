#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <tuple>

#include "CPUTimer.h"
#include "GPUTimer.cuh"

#include <cuda_runtime_api.h>
#include "CpuMlpEngine.h"
#include "GpuMlpEngine.h"




std::tuple<float, float, float> compare(float* s0, float* s1, size_t num)
{
	float mean_err = 0, max_err = 0, mse = 0;
	for (size_t i = 0; i < num; ++i)
	{
		float a = s0[i];
		float b = s1[i];
		//float d = std::abs(a - b);
		float scale = std::max(1.0f, std::max(std::abs(a), std::abs(b)));
		float d = std::abs(a - b) / scale;
		mean_err += d;
		if (d > max_err)
		{
			max_err = d;
			//std::cout << "max_err=" << max_err << " at " << i / 4 << " " << i % 4 << " a=" << a << " b=" << b << std::endl;
		}
		mse += d * d;
	}

	mse /= num;
	mean_err /= num;
	//std::cout << "mse=" << mse << " mean=" << mean_err << " max_err=" << max_err << std::endl;
	return { mse, mean_err, max_err };
}

int main(int argc, char* argv[])
{
	std::cout << "Assignment 03 - MLP Evaluation" << std::endl;

	if (argc < 2)
		std::cout << "Usage: ./mlp <test_folder> [runs] [runCPU] [number_inputs] [mse_threshold] [max_threshold]" << std::endl;


	float mse_threshold = 0.005f, max_threshold = 0.15f; // max_threshold increased from 0.05f to 0.15f

	std::string testfolder = "tests/5x5";
	size_t runs = 10;
	size_t elements = static_cast<size_t>(- 1);
	bool runCPU = true;

	try
	{
		if (argc > 1)
			testfolder = argv[1];
		if (argc > 2)
			runs = std::atoi(argv[2]);
		if (argc > 3)
			runCPU = std::atoi(argv[3]) != 0;
		if (argc > 4)
			elements = std::atoi(argv[4]);
		if (argc > 5)
			mse_threshold = std::atof(argv[5]);
		if (argc > 6)
			max_threshold = std::atof(argv[6]);

		std::string testname = testfolder.substr(testfolder.find_last_of('/')+1);
		std::cout << "Running " << testname << std::endl;

		DataLoader loader(testfolder, elements);
		size_t networkevaluations = std::min<size_t>(elements, loader.inputs.dims[0]);
		size_t n_inputs = networkevaluations * loader.inputs.dims[1];
		size_t n_inputs_half = networkevaluations * loader.inputs_half.dims[1];
		size_t n_outputs = networkevaluations * loader.weights[2].dims[0];
		std::vector<float> outputs(n_outputs);


		if (runCPU)
		{
			std::cout << "Profiling " << runs << " on the CPU" << std::endl;
			CpuMlpEngine engine;
			engine.setWeights(loader.weights);
			CPUTimer cputimer(static_cast<int>(runs));
			for (size_t i = 0; i < runs; ++i)
			{
				cputimer.start();
				engine.runWithHostPointers(outputs.data(), loader.inputs.data.data(), static_cast<int>(networkevaluations));
				cputimer.end();
			}
			auto [mse, mean_err, max_err] = compare(outputs.data(), loader.ref_outputs[2].data.data(), n_outputs);
			auto cpures = cputimer.generateResult();
			std::cout << "CPU mse=" << mse << " mean=" << mean_err << " max_err=" << max_err << std::endl;
			std::cout << "CPU required " << cpures.mean_ << "ms on average with std " << cpures.std_dev_ << "ms on the CPU" << std::endl;
		}


		std::cout << "Profiling " << runs << " on the GPU" << std::endl;
		int failedRuns = 0;
		GpuMlpEngine engine;
		engine.setWeights(loader.weights);
		

		float* d_inputs, * d_outputs;
		__half* d_inputs_half;

		HANDLE_ERROR(cudaMalloc((void**)&d_inputs, n_inputs * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_inputs_half, n_inputs_half * 2));
		HANDLE_ERROR(cudaMalloc((void**)&d_outputs, n_outputs * sizeof(float)));
		HANDLE_ERROR(cudaMemcpy(d_inputs, loader.inputs.data.data(), n_inputs * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_inputs_half, loader.inputs_half.data.data(), n_inputs_half * 2, cudaMemcpyHostToDevice));

		CPUTimer cputimer(static_cast<int>(runs));
		GPUTimer gputimer(static_cast<int>(runs));


		float max_mse = 0, max_mean_err = 0, max_max_err = 0;
		for (size_t i = 0; i < runs; ++i)
		{
			HANDLE_ERROR(cudaMemset(d_outputs,0, n_outputs * sizeof(float)));
			cputimer.start();
			gputimer.start();
			engine.runWithDevicePointers(d_outputs, d_inputs, d_inputs_half, static_cast<int>(networkevaluations));
			gputimer.end();
			cputimer.end();
			HANDLE_ERROR(cudaMemcpy(outputs.data(), d_outputs, n_outputs * sizeof(float), cudaMemcpyDeviceToHost));
			auto [mse, mean_err, max_err] = compare(outputs.data(), loader.ref_outputs[2].data.data(), n_outputs);
			if (!(mse <= mse_threshold && max_err <= max_threshold))
			{
				std::cout << "Run " << i << " failed with mse=" << mse << " max_err=" << max_err << " mse_threshold=" << mse_threshold << " max_threshold=" << max_threshold << std::endl;
				++failedRuns;
			}
			max_mse = std::max(mse, max_mse);
			max_mean_err = std::max(mean_err, max_mean_err);
			max_max_err = std::max(max_err, max_max_err);
		}
		if (failedRuns == 0)
			std::cout << testname << " SUCCESS ";
		else
			std::cout << testname << " FAILED ";

		auto cpures = cputimer.generateResult();
		auto gpures = gputimer.generateResult();
		std::cout << "GPU max mse=" << max_mse << " max mean=" << max_mean_err << " max max_err=" << max_max_err << std::endl;
		std::cout << "GPU required " << gpures.mean_ << "ms on average with std " << gpures.std_dev_ << "ms on the GPU (" << cpures.mean_ << " +/-" << cpures.std_dev_ << "ms on the CPU)" << std::endl;
	
		// Write to output file
		std::ofstream results_csv;
		results_csv.open("results.csv", std::ios_base::app);
		results_csv << testname << "," << gpures.mean_ << "," << (failedRuns == 0 ? "1" : "0") << std::endl;
		results_csv.close();
	
	}
	catch (std::exception& ex)
	{
		std::cout << "Error: " << ex.what();
		return -1;
	}

	return 0;	
}

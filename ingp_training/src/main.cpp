#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <tuple>

#include "CPUTimer.h"
#include "GPUTimer.cuh"
#include "fpng.h"

#include <cuda_runtime_api.h>
#include "CpuINGPEngine.h"
#include "GpuINGPEngine.h"


float3 abs(const float3& v)
{
	return { std::abs(v.x), std::abs(v.y), std::abs(v.z) };
}
float max(const float3& v)
{
	return std::max(std::max(v.x, v.z), v.z);
}


std::tuple<float, float, float> compare(const uchar4* s0, const uchar4* s1, size_t num)
{
	float mean_err = 0, max_err = 0, mse = 0;
	for (size_t i = 0; i < num; ++i)
	{
		float3 a = make_float3(s0[i].x, s0[i].y, s0[i].z);
		float3 b = make_float3(s1[i].x, s1[i].y, s1[i].z);
		//float d = std::abs(a - b);
		float3 d = abs(a - b);
		mean_err += d.x + d.y + d.z;
		float m = max(d);
		if (m > max_err)
			max_err = m;
		mse += dot(d, d);
	}

	mse /= num * 3;
	mean_err /= num * 3;
	//std::cout << "mse=" << mse << " mean=" << mean_err << " max_err=" << max_err << std::endl;
	return { mse, mean_err, max_err };
}

int main(int argc, char* argv[])
{
	std::cout << "Assignment 04 - Instant NGP" << std::endl;

	if (argc < 2)
	{
		std::cout << "Usage: ./ingp <test_folder> [runs] [runCPU] [camera_idx_min] [camera_idx_max] [mse_threshold] [max_threshold] " << std::endl;
	}

	//writeout("tests/100x100/debug_data.json", "tests/100x100");

	float mse_threshold = 5.0f, max_threshold = 5.0f;

	std::string testfolder = "tests/5x5";
	size_t runs = 1;
	bool runCPU = true;
	int camera_idx0 = 0;
	int camera_idxn = -1;

	try
	{
		fpng::fpng_init();

		if (argc > 1)
			testfolder = argv[1];
		if (argc > 2)
			runs = std::atoi(argv[2]);
		if (argc > 3)
			runCPU = std::atoi(argv[3]) != 0;
		if (argc > 4)
			camera_idx0 = std::atoi(argv[4]);
		if (argc > 5)
			camera_idxn = std::atoi(argv[5]);
		if (argc > 6)
			mse_threshold = std::atof(argv[6]);
		if (argc > 7)
			max_threshold = std::atof(argv[7]);


		std::string testname = testfolder.substr(testfolder.find_last_of('/')+1);
		std::cout << "Running " << testname << std::endl;

		std::ofstream results_csv;
		results_csv.open("results.csv", std::ios_base::app);

		DataLoader loader(testfolder);

		camera_idxn = camera_idxn < 0 ? int(loader.cameras.size()) : camera_idxn;
		for (int camera_idx = camera_idx0; camera_idx < camera_idxn; ++camera_idx) {
			auto& camera = loader.cameras[camera_idx];
			size_t n_pixel = camera.w * camera.h;
			std::vector<uchar4> output_image(n_pixel);


			if (runCPU)
			{
				std::cout << "Profiling pose " << camera_idx << " with " << runs << " runs on the CPU" << std::endl;
				CpuINGPEngine engine(loader.weights, loader.embedding, loader.offsets, loader.aabb);
				CPUTimer cputimer(static_cast<int>(runs));
				for (size_t i = 0; i < runs; ++i)
				{
					cputimer.start();
					engine.generateImage(camera, output_image.data());
					cputimer.end();
				}
				if (loader.ref_images.size() > camera_idx && !loader.ref_images[camera_idx].empty())
				{
					auto [mse, mean_err, max_err] = compare(output_image.data(), loader.ref_images[camera_idx].data(), n_pixel);
					std::cout << "CPU mse=" << mse << " mean=" << mean_err << " max_err=" << max_err << std::endl;
				}
				auto cpures = cputimer.generateResult();
				std::cout << "CPU required " << cpures.mean_ << "ms on average with std " << cpures.std_dev_ << "ms on the CPU" << std::endl;
				std::string res_img = "result_" + std::to_string(camera_idx) + "_" + testname + "_cpu.png";
				
				fpng::fpng_encode_image_to_file(res_img.c_str(), output_image.data(), camera.w, camera.h, 4);
			}


			std::cout << "Profiling pose " << camera_idx << " with " << runs << " runs on the GPU" << std::endl;
			int failedRuns = 0;
			GpuINGPEngine engine(loader.weights, loader.embedding, loader.offsets, loader.aabb);
			

			uchar4* d_image;

			HANDLE_ERROR(cudaMalloc((void**)&d_image, n_pixel * sizeof(uchar4)));

			CPUTimer cputimer(static_cast<int>(runs));
			GPUTimer gputimer(static_cast<int>(runs));


			float max_mse = 0, max_mean_err = 0, max_max_err = 0;
			bool didcompare = true;
			for (size_t i = 0; i < runs; ++i)
			{
				HANDLE_ERROR(cudaMemset(d_image, 0, n_pixel * sizeof(uchar4)));
				cputimer.start();
				gputimer.start();
				engine.generateImage(camera, d_image);
				gputimer.end();
				cputimer.end();
				HANDLE_ERROR(cudaMemcpy(output_image.data(), d_image, n_pixel * sizeof(uchar4), cudaMemcpyDeviceToHost));
				if (loader.ref_images.size() > camera_idx && !loader.ref_images[camera_idx].empty())
				{

					auto [mse, mean_err, max_err] = compare(output_image.data(), loader.ref_images[camera_idx].data(), n_pixel);
					if (!(mse <= mse_threshold && max_err <= max_threshold))
						++failedRuns;
					max_mse = std::max(mse, max_mse);
					max_mean_err = std::max(mean_err, max_mean_err);
					max_max_err = std::max(max_err, max_max_err);
				}
				else
					didcompare = false;
			}
			if (!didcompare)
			{
				std::cout << testname << " NOCOMPARISON ";
				failedRuns = 1;
			}
			else if (failedRuns == 0)
				std::cout << testname << " SUCCESS ";
			else
				std::cout << testname << " FAILED ";


			std::string res_img = "result_" + std::to_string(camera_idx) + "_" + testname + ".png";
			fpng::fpng_encode_image_to_file(res_img.c_str(), output_image.data(), camera.w, camera.h, 4);


			auto cpures = cputimer.generateResult();
			auto gpures = gputimer.generateResult();
			if (!loader.ref_images.empty())
				std::cout << "GPU max mse=" << max_mse << " max mean=" << max_mean_err << " max max_err=" << max_max_err << std::endl;
			std::cout << "GPU required " << gpures.mean_ << "ms on average with std " << gpures.std_dev_ << "ms on the GPU (" << cpures.mean_ << " +/-" << cpures.std_dev_ << "ms on the CPU)" << std::endl;
		
			// Write to output file
			results_csv << testname << "_" << camera_idx << "," << gpures.mean_ << "," << (failedRuns == 0 ? "1" : "0") << std::endl;
				
		}
		results_csv.close();
	
	}
	catch (std::exception& ex)
	{
		std::cout << "Error: " << ex.what();
		return -1;
	}

	return 0;	
}

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ##############################################################################################################################################
//
static inline void HandleError(cudaError_t err,
	const char* string,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		//printf("%s\n", string);
		//printf("%s in \n\n%s at line %d\n", cudaGetErrorString(err), file, line);
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err) + " " + string + " in " + file + " at line " + std::to_string(line));
	}
}

// ##############################################################################################################################################
//
static inline void HandleError(const char *file,
	int line) {
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		//printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err) + " in " + file + " at line " + std::to_string(line));
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, "", __FILE__, __LINE__ ))
#define HANDLE_ERROR_S( err , string) (HandleError( err, string, __FILE__, __LINE__ ))

// ##############################################################################################################################################
//
void inline start_clock(cudaEvent_t &start)
{
	HANDLE_ERROR(cudaEventRecord(start, 0));
}

// ##############################################################################################################################################
//
float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	HANDLE_ERROR(cudaEventRecord(end, 0));
	HANDLE_ERROR(cudaEventSynchronize(end));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));

	// Returns ms
	return time;
}

struct Result
{
	float mean_{ 0.0f };
	float std_dev_{ 0.0f };
	float median_{ 0.0f };
	float min_{ 0.0f };
	float max_{ 0.0f };
	int num_{ 0 };
};
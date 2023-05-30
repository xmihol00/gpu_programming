#pragma once

#include <string>
#include <array>
#include <vector>
#include <cuda_fp16.h>

template<typename T = float>
struct Matrix
{
	std::array<int, 2> dims;
	std::vector<T> data;
};


class DataLoader
{
	Matrix<float> loadMatrix(const std::string& name);
public:
	DataLoader(const std::string& director, size_t num);

	Matrix<float> weights[3];
	Matrix<float> inputs;
	Matrix<__half> inputs_half;
	Matrix<float> ref_outputs[3];
};
#pragma once

#include <string>
#include <array>
#include <vector>

#include "Camera.h"

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
	DataLoader(const std::string& director);

	std::vector<Camera> cameras;

	std::array<Matrix<float>, 3> weights;
	std::vector<float> embedding;
	std::vector<unsigned> offsets;
	std::array<float, 6> aabb;

	std::vector<std::vector<uchar4>> ref_images;
};
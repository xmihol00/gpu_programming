#pragma once

#include "DataLoader.h"

class CpuMlpEngine
{
	const Matrix<float>* _weights;
public:
	void setWeights(const Matrix<float> weights[3]);
	void runWithHostPointers(float* outputs, const float* input_data, int num_inputs);
};
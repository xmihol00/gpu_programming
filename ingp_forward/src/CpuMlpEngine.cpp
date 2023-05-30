#include "CpuMlpEngine.h"
#include <iostream>

void CpuMlpEngine::setWeights(const Matrix<float> weights[3])
{
	_weights = weights;
}

void CpuMlpEngine::runWithHostPointers(float* outputs, const float* input_data, int num_inputs)
{

	int n_evaluations = num_inputs;

	std::vector<float> tmp_outputs[2];
	const float* input = input_data;
	for (int layer = 0; layer < 3; ++layer)
	{
		auto& weight_matrix = _weights[layer];
		int n_outputs = weight_matrix.dims[0];
		int n_weights = weight_matrix.dims[1];

		float* output = nullptr;
		if (layer != 2)
		{
			tmp_outputs[layer].resize(n_evaluations * n_outputs);
			output = tmp_outputs[layer].data();
		}
		else
			output = outputs;
			
		for (int i = 0; i < n_evaluations; ++i)
		{
			for (int o = 0; o < n_outputs; ++o)
			{
				float val = 0.0f;
				for (int w = 0; w < n_weights; ++w)
				{
					val += input[i * n_weights + w] * weight_matrix.data[o * n_weights + w];
					//std::cout << "val  += " << input[i * n_weights + w] << " * " << weight_matrix.data[o * n_weights + w] << " = " << val << std::endl;
				}
				if (layer != 2)
					val = std::max(0.0f, val);
				output[i * n_outputs + o] = val;
				//std::cout << o << ": " << val << std::endl;
			}
			//std::cout << std::endl;
		}
		input = output;

	}
}

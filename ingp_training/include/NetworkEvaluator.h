#pragma once

#include <vector_types.h>
#include "helper_math.h"

#include "DataLoader.h"
#include <array>

struct NetworkEvaluator
{
	std::array<Matrix<float>, 3> weights;

	float4 evaluate(const float input[71])
	{
		float fo[4];

		std::vector<float> tmp_outputs[2];
		for (int layer = 0; layer < 3; ++layer)
		{
			auto& weight_matrix = weights[layer];
			int n_outputs = weight_matrix.dims[0];
			int n_weights = weight_matrix.dims[1];

			float* output = nullptr;
			if (layer != 2)
			{
				tmp_outputs[layer].resize(n_outputs);
				output = tmp_outputs[layer].data();
			}
			else
				output = fo;


			for (int o = 0; o < n_outputs; ++o)
			{
				float val = 0.0f;
				for (int w = 0; w < n_weights; ++w)
				{
					val += input[w] * weight_matrix.data[o * n_weights + w];
				}
				if (layer != 2)
					val = std::max(0.0f, val);
				output[o] = val;
			}
	
			input = output;
		}
		return float4{ fo[0], fo[1], fo[2], fo[3] };
	}
};
#pragma once

#include <vector_types.h>
#include "helper_math.h"

template<int MaxFreqLog2 = 5>
struct FrequencyEncoder
{
	static constexpr int max_freq_log2 = MaxFreqLog2;
	static constexpr int Outputs = 3 + 3 * 2 * (max_freq_log2 + 1);
	void encode(const float3& val, float* outputs)
	{
		outputs[0] = val.x;
		outputs[1] = val.y;
		outputs[2] = val.z;

		int j = 3;
		for (int i = 0; i <= max_freq_log2; ++i)
		{
			double freq = pow(2.0, static_cast<double>(i));
			for (int k = 0; k < 3; ++k)
			{
				double v = freq * outputs[k];
				outputs[j + k] = static_cast<float>(sin(v));
				outputs[j + 3 + k] = static_cast<float>(cos(v));
			}
			j += 6;
		}
	}
};
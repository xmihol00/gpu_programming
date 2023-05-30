#pragma once

#include <vector_types.h>
#include "helper_math.h"


int w, h;
uchar4* output_image;

int lastx{ -1 }, lasty{ -1 };


struct OutputAccumulator
{
	float rem = 1.0f;
	float3 accumulation{ 0.f, 0.f, 0.f };

	static unsigned char f_to_uchar(float f)
	{
		return static_cast<unsigned char>(std::min(std::max(0.0f, roundf(f * 255.f)), 255.0f));
	}

	void reset()
	{
		rem = 1.0f;
		accumulation = { 0.f, 0.f, 0.f };
	}
	float sigmoid(float x)
	{
		return 1.0f / (1.0f + exp(-x));
	}

	void accumulate(float step, const float4& networkoutput)
	{
		float3 rgb{ sigmoid(networkoutput.x), sigmoid(networkoutput.y), sigmoid(networkoutput.z) };
		float sigma = exp(networkoutput.w);

		float alpha = 1.0f - exp(-step * sigma);
		accumulation += rem * rgb * alpha;
		rem *= 1.f - alpha;
	}

	float3 get(float3 bg = { 1.f, 1.f, 1.f })
	{
		return accumulation + bg * rem;
	}
	uchar4 getRGBA(float3 bg = { 1.f, 1.f, 1.f })
	{
		float3 f = get(bg);
		return { f_to_uchar(f.x), f_to_uchar(f.y), f_to_uchar(f.z), 255 };
	}
};
#pragma once

#include <vector_types.h>
#include "helper_math.h"


template<class DirEncoder, class PosEncoder>
struct SampleEncoder
{
	DirEncoder& dirEncoder;
	PosEncoder& posEncoder;
	SampleEncoder(DirEncoder& dirEncoder, PosEncoder& posEncoder) :
		dirEncoder(dirEncoder),
		posEncoder(posEncoder)
	{
	}
	template<typename F>
	auto encode(const float3& pos, const float3& dir, F&& inputReceiver)
	{
		std::array<float, DirEncoder::Outputs + PosEncoder::Outputs> enc;
		dirEncoder.encode(dir, enc.data());
		posEncoder.encode(pos, enc.data() + DirEncoder::Outputs);
		return inputReceiver(enc);
	}
};
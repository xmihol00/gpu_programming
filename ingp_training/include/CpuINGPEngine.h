#pragma once

#include "Camera.h"
#include "DataLoader.h"


#include "HashEncoder.h"
#include "NetworkEvaluator.h"
#include "SampleGenerator.h"

class CpuINGPEngine
{
	HashEncoder<> posEncoder;
	NetworkEvaluator network;
	SampleGenerator sampleGenerator;
public:
	CpuINGPEngine(const std::array<Matrix<float>, 3>& weights, const std::vector<float>& embedding, const std::vector<unsigned>& offsets, const std::array<float, 6>& aabb);
	void generateImage(const Camera& camera, uchar4* image);
};
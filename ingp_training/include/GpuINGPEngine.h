#pragma once

#include <vector_types.h>

#include "Camera.h"
#include "DataLoader.h"

#include "HashEncoder.h"
#include "NetworkEvaluator.h"
#include "SampleGenerator.h"

#include <iostream>
#include <iomanip>
#include <cuda_fp16.h>

class GpuINGPEngine
{
private:
	Camera *_d_camera = nullptr;
	float *_d_rays = nullptr;
	float *_d_aabb = nullptr;
	float *_d_samples = nullptr;
	float *_d_sampleDistributions = nullptr;
	float *_d_networkInputs = nullptr;
	float *_d_networkOutputs = nullptr;
	float *_d_embeddings = nullptr;
	uint32_t *_d_offsets = nullptr;
	float *_d_weightsL0 = nullptr;
	float *_d_weightsL1 = nullptr;
	float *_d_weightsL2 = nullptr;
	float *_d_weightsPaddedL0 = nullptr;
	float *_d_weightsFrequencyL0 = nullptr;
	float *_d_weightsPositionL0 = nullptr;
	half *_d_halfWeightsFrequencyL0 = nullptr;
	half *_d_halfWeightsFrequencyTransposedL0 = nullptr;
	half *_d_halfWeightsPositionL0 = nullptr;
	half *_d_halfWeightsPositionTransposedL0 = nullptr;
	half *_d_halfWeightsTransposedL1 = nullptr;
	half *_d_halfWeights2WayAddressingL2 = nullptr;
	uint32_t *_d_pixelCounter = nullptr;

public:
	/// <summary>
	/// Constructor getting the network weights (compare assignment 3), the hash embedding, the offsets for the multi-level hash values, and the AABB for ray generation
	/// Look at CpuINGPEngine for how the different objects are being used
	/// </summary>
	/// <param name="weights">network weights</param>
	/// <param name="embedding">all levels of hash data</param>
	/// <param name="offsets">offsets to each level of hash data</param>
	/// <param name="aabb">axis aligned bounding box in which samples shouild be taken</param>
	GpuINGPEngine(const std::array<Matrix<float>, 3>& weights, const std::vector<float>& embedding, const std::vector<unsigned>& offsets, const std::array<float, 6>& aabb);

	~GpuINGPEngine();

	/// <summary>
	/// main method called for evaluation
	/// </summary>
	/// <param name="camera">camera parameters, including ray origin, transformation matrix, and camera parameters. data resides on the CPU</param>
	/// <param name="d_image">memory of w*h expecting RGBA data in row major ordering. data resides on the GPU</param>
	void generateImage(const Camera& camera, uchar4* d_image);
};
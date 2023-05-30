#include "CpuINGPEngine.h"

#include "FrequencyEncoder.h"
#include "RayGeneration.h"
#include "OutputAccumulator.h"
#include "SampleEncoder.h"

CpuINGPEngine::CpuINGPEngine(const std::array<Matrix<float>, 3>& weights, const std::vector<float>& embedding, const std::vector<unsigned>& offsets, const std::array<float, 6>& aabb) :
	posEncoder{ embedding, offsets }, network{ weights }, sampleGenerator{aabb}
{
}

void CpuINGPEngine::generateImage(const Camera& cam, uchar4* image)
{
	RayGeneration raygen(cam);

	FrequencyEncoder<> dirEnc;
	SampleEncoder sampleEncoder{ dirEnc, posEncoder };


	raygen.generateRays([&](int x, int y, const float3& origin, const float3& dir)
		{
			OutputAccumulator acc;
			sampleGenerator.generateSamples(x, y, origin, dir, 
				[&](const float3& pos, const float3& dir) {
				return sampleEncoder.encode(pos, dir, [&](const std::array<float, 71> enc_inputs) {
					return network.evaluate(enc_inputs.data());
					});
				}, 
				[&](int x, int y, float step, const float4& output) {
					acc.accumulate(step, output);
				});
			image[y * cam.w + x] = acc.getRGBA();
		});
}

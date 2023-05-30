#pragma once

#include <vector_types.h>
#include "helper_math.h"
#include <array>


struct SampleGenerator
{
	std::array<float, 6> aabb;
	static constexpr float min_near = 0.2f;
	static constexpr int num_steps = 512;

	SampleGenerator() = default;
	SampleGenerator(const std::array<float, 6>& aabb) : aabb{ aabb } {}
	SampleGenerator(float a0, float a1, float a2, float a3, float a4, float a5) : aabb{ a0, a1, a2, a3, a4, a5 } {}

	template<typename F, typename O>
	void generateSamples(int x, int y, const float3& origin, const float3& dir, F&& sampleReceiver, O&& accumulator)
	{
		float3 r_dir{ 1 / dir.x , 1 / dir.y, 1 / dir.z };

		// get near far (assume cube scene)
		float near = (aabb[0] - origin.x) * r_dir.x;
		float far = (aabb[3] - origin.x) * r_dir.x;
		if (near > far) std::swap(near, far);

		float near_y = (aabb[1] - origin.y) * r_dir.y;
		float far_y = (aabb[4] - origin.y) * r_dir.y;
		if (near_y > far_y) std::swap(near_y, far_y);

		if (near > far_y || near_y > far) {
			//std::cout << "all misses" << std::endl;
			return;
		}

		if (near_y > near)
			near = near_y;
		if (far_y < far)
			far = far_y;

		float near_z = (aabb[2] - origin.z) * r_dir.z;
		float far_z = (aabb[5] - origin.z) * r_dir.z;
		if (near_z > far_z) std::swap(near_z, far_z);

		if (near > far_z || near_z > far) {
			//std::cout << "all misses" << std::endl;
			return;
		}

		if (near_z > near)
			near = near_z;
		if (far_z < far)
			far = far_z;

		if (near < min_near)
			near = min_near;

		// stepping
		float sample_dist = (far - near) / num_steps;
		float inv_s = 1.0f / (num_steps - 1);
		for (int i = 0; i < num_steps; ++i)
		{

			float z_val = near + (far - near) * i * inv_s;
			float3 xyz = origin + dir * z_val;
			xyz = fminf(fmaxf(xyz, float3{ aabb[0], aabb[1], aabb[2] }), float3{ aabb[3], aabb[4], aabb[5] });

			float4 res = sampleReceiver(xyz, dir);
			accumulator(x, y, sample_dist, res);
		}
	}
};
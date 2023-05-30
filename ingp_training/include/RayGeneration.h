#pragma once

#include <vector_types.h>
#include "helper_math.h"

#include "Camera.h"

struct RayGeneration
{
	Camera cam;

	RayGeneration() = default;
	RayGeneration(const Camera& cam) : cam(cam) { }

	template<typename F>
	void generateRays(F&& sampleGenerator)
	{
		for (int y = 0; y < cam.h; ++y)
			for (int x = 0; x < cam.w; ++x)
			{
				float3 dir_base = { (x + 0.5f - cam.cx) / cam.fl_x, (y + 0.5f - cam.cy) / cam.fl_y, 1.0f };
				dir_base = normalize(dir_base);

				float3 dir;
				dir.x = dot(dir_base, cam.poseMatrix[0]);
				dir.y = dot(dir_base, cam.poseMatrix[1]);
				dir.z = dot(dir_base, cam.poseMatrix[2]);

				sampleGenerator(x, y, cam.origin, dir);
			}
	}
};
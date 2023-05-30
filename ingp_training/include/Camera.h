#pragma once

#include <vector_types.h>
#include "helper_math.h"

struct Camera
{
	float3 origin;
	float3 poseMatrix[3];
	int w, h;
	float fl_x, fl_y;
	float cx, cy;

	Camera() = default;
	Camera(const float3& o, float3 pM[3], int w, int h, float fl_x, float fl_y, float cx, float cy) :
		origin{ o }, poseMatrix{ pM[0], pM[1], pM[2] }, w{ w }, h{ h }, fl_x{ fl_x }, fl_y{ fl_y }, cx{ cx }, cy{ cy }
	{}

};
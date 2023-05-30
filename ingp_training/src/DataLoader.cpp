#include "DataLoader.h"

#include <npy.hpp>
#include <json.hpp>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "helper_math.h"
#include <random>
#include <filesystem>
#include "fpng.h"


Matrix<float> DataLoader::loadMatrix(const std::string& name)
{
	std::vector<unsigned long> shape;
	bool fortran_order;
	Matrix m;
	npy::LoadArrayFromNumpy(name, shape, fortran_order, m.data);
	if (shape.size() != 2)
		throw std::runtime_error("array is not two dimensional");
	m.dims[0] = shape[0];
	m.dims[1] = shape[1];
	return m;
}


DataLoader::DataLoader(const std::string& director)
{
	// here goes the heavy lifting
	using json = nlohmann::json;

	std::string config_file = director + "/config.json";

	if (!std::filesystem::exists(config_file))
		throw std::runtime_error("Input config file.");


	std::vector<unsigned long> shape;
	std::vector<int> ioffsets;
	bool fortran_order;

	npy::LoadArrayFromNumpy(director + "/../encoder.offsets.npy", shape, fortran_order, ioffsets);
	npy::LoadArrayFromNumpy(director + "/../encoder.embeddings.npy", shape, fortran_order, embedding);

	offsets = std::vector<unsigned>(begin(ioffsets), end(ioffsets));

	std::ifstream configf;
	configf.exceptions(std::ifstream::badbit);
	configf.open(config_file.c_str());
	json j;
	configf >> j;

	//json cpy;
	
	bool hasImages = true;

	for (int pose_idx = 0; pose_idx < j["poses"].size(); ++pose_idx) 
	{
		auto pose = j["poses"][pose_idx];
		//cpy["poses"][0] = pose;
		float3 poseMatrix[3];
		float origin_xyz[3];
		int i = 0;
		for (auto& r : pose)
		{
			poseMatrix[i].x = r[0];
			poseMatrix[i].y = r[1];
			poseMatrix[i].z = r[2];
			origin_xyz[i] = r[3];
			++i;
			if (i == 3)
				break;
		}
		float3 origin{ origin_xyz[0], origin_xyz[1], origin_xyz[2] };

		int w = j["transform"]["w"];
		int h = j["transform"]["h"];
		//cpy["transform"] = j["transform"];

		auto intrinsics = j["intrinsics"];
		float fl_x = intrinsics["fl_x"];
		float fl_y = intrinsics["fl_y"];
		float cx = intrinsics["cx"];
		float cy = intrinsics["cy"];

		//cpy["intrinsics"] = j["intrinsics"];
		
		auto aabb_field = j["aabb"];
		//cpy["aabb"] = j["aabb"];
		
		auto camera = Camera{ origin, poseMatrix, w, h, fl_x, fl_y, cx, cy };
		cameras.push_back(camera);
		aabb = { aabb_field[0], aabb_field[1], aabb_field[2], aabb_field[3], aabb_field[4], aabb_field[5] };


		weights[0] = loadMatrix(director + "/../net.0.weight.npy");
		weights[1] = loadMatrix(director + "/../net.1.weight.npy");
		weights[2] = loadMatrix(director + "/../net.2.weight.npy");

		//std::ofstream config_cpy("cpy.json");
		//config_cpy << cpy;
		//config_cpy.close();

		std::string gt = director + "/ref_image_" + std::to_string(pose_idx) + ".png";
		if (std::filesystem::exists(gt))
		{
			uint32_t w, h, c;
			std::vector<uint8_t> data;
			fpng::fpng_decode_file(gt.c_str(), data, w, h, c, 4);
			if (w != camera.w || h != camera.h || data.empty())
			{
				hasImages = false;
				ref_images.clear();
			}
			ref_images.push_back({});
			ref_images.back().resize(w * h);
			for (size_t i = 0; i < w * h; ++i)
			{
				size_t p = 4 * i;
				ref_images.back()[i] = uchar4{ data[p], data[p + 1], data[p + 2], 255 };
			}
		}
		else
			ref_images.push_back({});
	}
}

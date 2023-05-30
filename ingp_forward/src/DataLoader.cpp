#include "DataLoader.h"

#include <npy.hpp>
#include <json.hpp>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <random>

Matrix<__half> toHalf(const Matrix<float>& mat)
{
	Matrix<__half> out;
	out.dims[0] = mat.dims[0];
	out.dims[1] = mat.dims[1] + (mat.dims[1] & 0x1);
	auto it = mat.data.begin();
	for (int i = 0; i < mat.dims[0]; ++i)
	{
		int j = 0;
		for (; j < mat.dims[1]; ++j)
			out.data.push_back(__float2half(*it++));
		for (; j < out.dims[1]; ++j)
			out.data.push_back(__float2half(0.0f));
	}
	return out;
}


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

DataLoader::DataLoader(const std::string& director, size_t num)
{
	weights[0] = loadMatrix(director + "/../net.0.weight.npy");
	weights[1] = loadMatrix(director + "/../net.1.weight.npy");
	weights[2] = loadMatrix(director + "/../net.2.weight.npy");

	inputs = loadMatrix(director + "/enc_inputs.npy");

	ref_outputs[0] = loadMatrix(director + "/layeroutputs.0.npy");
	ref_outputs[1] = loadMatrix(director + "/layeroutputs.1.npy");
	ref_outputs[2] = loadMatrix(director + "/layeroutputs.2.npy");


	if (num != static_cast<size_t>(-1) && num > static_cast<size_t>(inputs.dims[0]))
	{
		Matrix<float> temp_in, temp_refs[3];
		temp_in.dims[0] = num;
		temp_in.dims[1] = inputs.dims[1];
		temp_in.data.reserve(num * inputs.dims[1]);
		for (auto v : inputs.data)
			temp_in.data.emplace_back(v);

		for (int i = 0; i < 3; ++i)
		{
			temp_refs[i].dims[0] = num;
			temp_refs[i].dims[1] = ref_outputs[i].dims[1];
			temp_refs[i].data.reserve(num * ref_outputs[i].dims[1]);
			for (auto v : ref_outputs[i].data)
				temp_refs[i].data.emplace_back(v);
		}
		
		std::random_device rd; 
		std::mt19937 gen(rd());
		std::uniform_int_distribution<size_t> distr(0, inputs.dims[0]-1);



		while (temp_in.data.size() < num * inputs.dims[1])
		{
			size_t sel = distr(gen);
			for(auto it = inputs.data.begin() + sel * inputs.dims[1]; it != inputs.data.begin() + (sel + 1)* inputs.dims[1]; ++it)
				temp_in.data.emplace_back(*it);

			for (int i = 0; i < 3; ++i)
				for (auto it = ref_outputs[i].data.begin() + sel * ref_outputs[i].dims[1]; it != ref_outputs[i].data.begin() + (sel + 1) * ref_outputs[i].dims[1]; ++it)
					temp_refs[i].data.emplace_back(*it);
		}

		std::swap(inputs, temp_in);
		for (int i = 0; i < 3; ++i)
			std::swap(ref_outputs[i], temp_refs[i]);

	}

	inputs_half = toHalf(inputs);


}

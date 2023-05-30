#include <npy.hpp>
#include <json.hpp>
#include <string>
#include <vector>

void writeout(const std::string& jsondata, const std::string& folderout, const std::string& name = "")
{

	using json = nlohmann::json;
	std::ifstream configf;
	configf.exceptions(std::ifstream::badbit);
	configf.open(jsondata.c_str());
	json j;
	configf >> j;

	//auto a = j["input"]["pos"][0];
	//auto b = j["input"]["dir"][0];

	auto enc_pos = j["encoded_input"]["pos"][0];
	size_t pos_samples_cnt = enc_pos.size();
	size_t pos_data_cnt = enc_pos[0].size();

	auto enc_dir = j["encoded_input"]["dir"][0];
	size_t dir_samples_cnt = enc_dir.size();
	size_t dir_data_cnt = enc_dir[0].size();

	assert(pos_samples_cnt == dir_samples_cnt);

	std::vector<float> enc_inputs;
	enc_inputs.reserve(pos_samples_cnt * (pos_data_cnt + dir_data_cnt));

	for (size_t i = 0; i < pos_samples_cnt; ++i)
	{
		for (auto d : enc_dir[i])
			enc_inputs.push_back(d);
		for (auto p : enc_pos[i])
			enc_inputs.push_back(p);
	}

	std::array<unsigned long, 2> outshape{ {pos_samples_cnt,pos_data_cnt + dir_data_cnt} };
	npy::SaveArrayAsNumpy(folderout + "/enc_inputs" + name + ".npy", false, outshape.size(), outshape.data(), enc_inputs);



	auto layer_output = j["layer_output"];
	for (auto l : { "0", "1", "2" })
	{
		size_t a = layer_output[l][0].size();
		size_t b = layer_output[l][0][0].size();
		std::vector<float> outputs;
		outputs.reserve(a * b);
		for (auto& n : layer_output[l][0])
			for (auto v : n)
				outputs.push_back(v);

		std::array<unsigned long, 2> outshape{ {a, b} };
		npy::SaveArrayAsNumpy(std::string(folderout + "/layeroutputs" + name + ".") + l + ".npy", false, outshape.size(), outshape.data(), outputs);
	}

}
#pragma once

#include <vector_types.h>
#include "helper_math.h"

#include <stdint.h>
#include <vector>
#include <cassert>


template<typename Vec3>
auto toArray(const Vec3& v)
{
	return std::array<decltype(v.x), 3>{v.x, v.y, v.z};
}


template<int Levels = 16, int EmbeddingDimensions = 2, int ResolutionMultiplier = 1, int BaseResolution = 16>
struct HashEncoder
{
	// S = Resolution Multiplier
	// C = Embedding Dimension
	// L = Levels
	// D = Coord Dimensions


	static constexpr int Dimensions = 3;
	static constexpr int Outputs = Levels * EmbeddingDimensions;
	std::vector<float> embedding;
	std::vector<unsigned> offsets;

	HashEncoder(std::vector<float>&& embedding, std::vector<unsigned>&& offsets) :
		embedding{ std::move(embedding) }, offsets{ std::move(offsets) }
	{
		assert(this->offsets.size() == Levels + 1);
	}

	HashEncoder(const std::vector<float>& embedding, const std::vector<unsigned>& offsets) :
		embedding{ embedding }, offsets{ offsets }
	{
		assert(this->offsets.size() == Levels + 1);
	}


	uint32_t fast_hash(const uint32_t pos_grid[Dimensions])
	{
		constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

		uint32_t result = 0;
		for (uint32_t i = 0; i < Dimensions; ++i) {
			result ^= pos_grid[i] * primes[i];
		}

		return result;
	}



	uint32_t get_grid_index(const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[Dimensions]) {
		uint32_t stride = 1;
		uint32_t index = 0;

		for (uint32_t d = 0; d < Dimensions && stride <= hashmap_size; d++) {
			index += pos_grid[d] * stride;
			stride *= (resolution + 1);
		}

		if (stride > hashmap_size) {
			index = fast_hash(pos_grid);
		}

		return (index % hashmap_size) * EmbeddingDimensions;
	}

	void encode(const float3& val, float* outputs)
	{
		// zero output
		for (int i = 0; i < Levels * EmbeddingDimensions; ++i)
			outputs[i] = 0.0f;

		float3 input = (val + 1.0f) / 2.0f;
		for (int level = 0; level < Levels; ++level)
		{
			float* grid = embedding.data() + offsets[level] * EmbeddingDimensions;
			unsigned hashmap_size = offsets[level + 1] - offsets[level];
			float scale = exp2f(level * ResolutionMultiplier) * BaseResolution - 1.0f;
			unsigned resolution = static_cast<unsigned>(ceil(scale)) + 1;

			// pos in the virtual grid of the level
			float3 pos = input * scale + 0.5f;
			uint3 pos_grid3 = make_uint3(floorf(pos.x), floorf(pos.y), floorf(pos.z));
			auto pos_grid = toArray(pos_grid3);
			auto local_pos = toArray(pos - make_float3(pos_grid3));

			// 8 interpolation corners
			for (int idx = 0; idx < (1 << Dimensions); idx++)
			{
				float w = 1.0f;
				uint32_t pos_grid_local[Dimensions];

				// contribution of each corner
				for (uint32_t d = 0; d < Dimensions; d++)
				{
					if ((idx & (1 << d)) == 0)
					{
						w *= 1 - local_pos[d];
						pos_grid_local[d] = pos_grid[d];
					}
					else
					{
						w *= local_pos[d];
						pos_grid_local[d] = pos_grid[d] + 1;
					}
				}

				uint32_t index = get_grid_index(hashmap_size, resolution, pos_grid_local);
				for (uint32_t ch = 0; ch < EmbeddingDimensions; ch++)
					outputs[level * EmbeddingDimensions + ch] += w * grid[index + ch];
			}

		}
	}
};
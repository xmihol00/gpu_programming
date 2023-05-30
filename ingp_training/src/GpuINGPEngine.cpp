#include "GpuINGPEngine.h"

using namespace std;

void freeMemoryOnDevice(initializer_list<void *> d_memoryList);
void allocateMemory(void **d_data, uint32_t size);
void copyMemoryToDevice(void *d_data, void *h_data, uint32_t size);
void copyMemoryFromDevice(void *h_data, void *d_data, uint32_t size);
void allocateAndSetMemory(void **d_data, void *h_data, uint32_t size);
void allocateAndSetWeights(vector<float> weights, float *&d_weights);
void allocateAndSetHalfWeights(vector<half> weights, half *&d_weights);

void baselineLaunch(float *d_rays, Camera *d_camera, float *d_aabb, float *d_samples, float *d_sampleDistributions, 
					float *d_embeddings, uint32_t *d_offsets, float *d_networkInputs, float *d_networkOutputs, float *d_weightsL0, 
					float *d_weightsL1, float *d_weightsL2, uint8_t *d_outputs, uint16_t width, uint16_t height);
void allInOneLaunch(Camera *d_camera, float *d_aabb, float *d_sampleDistributions, float *d_embeddings, uint32_t *d_offsets, 
					float *d_networkInputs, float *d_networkOutputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
					uint8_t *d_outputs, uint16_t width, uint16_t height);
void frequencyEncodingCoopLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, float *d_networkInputs, 
						         float *d_networkOutputs, float *d_weightsFrequencyL0, float *d_weightsPositionL0, float *d_weightsL1, 
						         float *d_weightsL2, uint8_t *d_outputs, uint16_t width, uint16_t height);
void halfPrecisionLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, float *d_weightsFrequencyL0, 
						 half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint16_t width, 
						 uint16_t height);
void positionEncodingCoopLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, float *d_weightsFrequencyL0, 
						        half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint16_t width, 
						        uint16_t height);
void moreCoopLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, half *d_weightsFrequencyL0, 
					half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint16_t width, 
					uint16_t height);
void lessThreadsFixedIterationsLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, 
									  half *d_weightsFrequencyL0, half *d_weightsPositionL0, half *d_weightsL1, 
									  half *d_weightsL2, uint8_t *d_outputs, uint16_t width, uint16_t height);
void pixelPoolLaunch(Camera *d_camera, float *d_aabb, float *d_embeddings, uint32_t *d_offsets, half *d_weightsFrequencyL0, 
					 half *d_weightsPositionL0, half *d_weightsL1, half *d_weightsL2, uint8_t *d_outputs, uint32_t *d_pixelCounter, 
					 uint16_t width, uint16_t height);

GpuINGPEngine::GpuINGPEngine(const std::array<Matrix<float>, 3>& weights, const std::vector<float>& embedding, 
                             const std::vector<unsigned>& offsets, const std::array<float, 6>& aabb)

{
    allocateMemory((void **)&_d_camera, sizeof(Camera)); // TODO: pinned
    allocateMemory((void **)&_d_pixelCounter, sizeof(uint32_t));
    allocateAndSetMemory((void **)&_d_aabb, (void *)aabb.data(), 6 * sizeof(float));
    allocateAndSetMemory((void **)&_d_embeddings, (void *)embedding.data(), embedding.size() * sizeof(float));
    allocateAndSetMemory((void **)&_d_offsets, (void *)offsets.data(), offsets.size() * sizeof(unsigned));
    allocateAndSetWeights(weights[0].data, _d_weightsL0);
	allocateAndSetWeights(weights[1].data, _d_weightsL1);
	allocateAndSetWeights(weights[2].data, _d_weightsL2);
        
    // needed for 'baselineLaunch' and 'encodingCoopLaunch'
    /*#if __CUDA_ARCH__ >= 530
		constexpr uint16_t samplesPerIteration = 40'000;
	#else
		constexpr uint16_t samplesPerIteration = 2'500;
	#endif
    allocateMemory((void **)&_d_rays, samplesPerIteration * 3 * sizeof(float));
    allocateMemory((void **)&_d_sampleDistributions, samplesPerIteration * 512 * sizeof(float));
    allocateMemory((void **)&_d_samples, samplesPerIteration * 512 * 6 * sizeof(float));
    allocateMemory((void **)&_d_networkInputs, 71 * samplesPerIteration * 512 * sizeof(float));
    allocateMemory((void **)&_d_networkOutputs, 4 * samplesPerIteration * 512 * sizeof(float));*/

    vector<float> W0{weights[0].data.begin(), weights[0].data.end()};
	for (int8_t i = 64; i >= 1; i--)
	{
		W0.insert(W0.begin() + (i * 71), 0.0f);
	}
	allocateAndSetWeights(W0, _d_weightsPaddedL0);

    vector<float> weightsFrequencyL0;
    vector<float> weightsPositionL0;
    for (uint8_t i = 0; i < 64; i++)
    {
        for (uint8_t j = 0; j < 39; j++)
        {
            weightsFrequencyL0.push_back(weights[0].data[i * 71 + j]);
        }
        weightsFrequencyL0.push_back(0);

        for (uint8_t j = 39; j < 71; j++)
        {
            weightsPositionL0.push_back(weights[0].data[i * 71 + j]);
        }
    }
    allocateAndSetWeights(weightsFrequencyL0, _d_weightsFrequencyL0);
    allocateAndSetWeights(weightsPositionL0, _d_weightsPositionL0);

    vector<half> halfFrequencyW0;
    for (float w : weightsFrequencyL0)
    {
        halfFrequencyW0.push_back(__float2half(w));
    }
    vector<half> halfFrequencyTransposedW0;
    for (int16_t i = 0; i < 20; i++)
	{
		for (int16_t j = 0; j < 128; j++)
		{
			halfFrequencyTransposedW0.push_back(halfFrequencyW0[i + j * 20]);
		}
	}

    vector<half> halfPositionW0;
    for (float w : weightsPositionL0)
    {
        halfPositionW0.push_back(__float2half(w));
    }
    
    vector<half> halfPositionTransposedW0;
	for (int16_t i = 0; i < 32; i++)
	{
		for (int16_t j = 0; j < 64; j++)
		{
			halfPositionTransposedW0.push_back(halfPositionW0[i + j * 32]);
		}
	}

    vector<half> halfW1;
	for (float w : weights[1].data)
	{
		halfW1.push_back(__float2half(w));
	}
    vector<half> halfTransposedW1;
	for (int16_t i = 0; i < 64; i++)
	{
		for (int16_t j = 0; j < 64; j++)
		{
			halfTransposedW1.push_back(halfW1[i + j * 64]);
		}
	}

    vector<half> halfW2;
	for (float w : weights[2].data)
	{
		halfW2.push_back(__float2half(w));
	}
    vector<half> half2WayAddressingW2;
	for (int16_t i = 0; i < 4 * 64; i += 2)
	{
		half2WayAddressingW2.push_back(halfW2[i]);
	}
	for (int16_t i = 1; i < 4 * 64; i += 2)
	{
		half2WayAddressingW2.push_back(halfW2[i]);
	}

    allocateAndSetHalfWeights(halfFrequencyW0, _d_halfWeightsFrequencyL0);
    allocateAndSetHalfWeights(halfFrequencyTransposedW0, _d_halfWeightsFrequencyTransposedL0);
    allocateAndSetHalfWeights(halfPositionW0, _d_halfWeightsPositionL0);
    allocateAndSetHalfWeights(halfPositionTransposedW0, _d_halfWeightsPositionTransposedL0);
    allocateAndSetHalfWeights(halfTransposedW1, _d_halfWeightsTransposedL1);
    allocateAndSetHalfWeights(half2WayAddressingW2, _d_halfWeights2WayAddressingL2);
}

GpuINGPEngine::~GpuINGPEngine()
{
	freeMemoryOnDevice({ _d_rays, _d_camera, _d_aabb, _d_samples, _d_networkInputs, _d_embeddings, _d_offsets, _d_networkOutputs,
                         _d_weightsL0, _d_weightsL1, _d_weightsL2, _d_sampleDistributions, _d_weightsPaddedL0, _d_weightsFrequencyL0,
                         _d_weightsPositionL0, _d_halfWeights2WayAddressingL2, _d_halfWeightsTransposedL1, _d_pixelCounter,
                         _d_halfWeightsFrequencyTransposedL0 });
}

void GpuINGPEngine::generateImage(const Camera& camera, uchar4* d_image)
{
    copyMemoryToDevice(_d_camera, (void *)&camera, sizeof(Camera));
    //baselineLaunch(_d_rays, _d_camera, _d_aabb, _d_samples, _d_sampleDistributions, _d_embeddings, _d_offsets, _d_networkInputs, 
    //               _d_networkOutputs, _d_weightsL0, _d_weightsL1, _d_weightsL2, reinterpret_cast<uint8_t *>(d_image), camera.w, 
    //               camera.h);
    //allInOneLaunch(_d_camera, _d_aabb, _d_sampleDistributions, _d_embeddings, _d_offsets, _d_networkInputs, _d_networkOutputs, 
    //               _d_weightsPaddedL0, _d_weightsL1, _d_weightsL2, reinterpret_cast<uint8_t *>(d_image), camera.w, camera.h);
    //frequencyEncodingCoopLaunch(_d_camera, _d_aabb, _d_embeddings, _d_offsets, _d_networkInputs, _d_networkOutputs, _d_weightsFrequencyL0, 
    //                            _d_weightsPositionL0, _d_weightsL1, _d_weightsL2, reinterpret_cast<uint8_t *>(d_image), camera.w, camera.h);
    //halfPrecisionLaunch(_d_camera, _d_aabb, _d_embeddings, _d_offsets, _d_weightsFrequencyL0, _d_halfWeightsPositionL0, 
    //                    _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, reinterpret_cast<uint8_t *>(d_image), 
    //                    camera.w, camera.h);
    //positionEncodingCoopLaunch(_d_camera, _d_aabb, _d_embeddings, _d_offsets, _d_weightsFrequencyL0, _d_halfWeightsPositionL0, 
    //                           _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, reinterpret_cast<uint8_t *>(d_image), 
    //                           camera.w, camera.h);
    //moreCoopLaunch(_d_camera, _d_aabb, _d_embeddings, _d_offsets, _d_halfWeightsFrequencyL0, _d_halfWeightsPositionTransposedL0, 
    //               _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, reinterpret_cast<uint8_t *>(d_image), 
    //               camera.w, camera.h);
    //lessThreadsFixedIterationsLaunch(_d_camera, _d_aabb, _d_embeddings, _d_offsets, _d_halfWeightsFrequencyL0, 
    //                                 _d_halfWeightsPositionTransposedL0, _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, 
    //                                 reinterpret_cast<uint8_t *>(d_image), camera.w, camera.h);
    pixelPoolLaunch(_d_camera, _d_aabb, _d_embeddings, _d_offsets, _d_halfWeightsFrequencyTransposedL0, 
                    _d_halfWeightsPositionTransposedL0, _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, 
                    reinterpret_cast<uint8_t *>(d_image), _d_pixelCounter, camera.w, camera.h);
}

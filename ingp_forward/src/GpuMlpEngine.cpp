#include "GpuMlpEngine.h"

using namespace std;

void allocateAndSetWeights(vector<float> weights, float *&d_weights);
void allocateAndSetHalfWeights(vector<half> weights, half *&d_weights);
void freeMemoryOnDevice(initializer_list<void *> d_memoryList);
void allocateMemory(float *&d_data, uint32_t size);
void baselineLaunch(const float* d_inputs, float* d_outputs, float* d_intermidiateOutputsL0, float* d_intermidiateOutputsL1, 
                    float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, int numberOfInputs);
void fusedGlobalMemoryLaunch(const float* d_inputs, float* d_outputs, float* d_intermidiateOutputsL0, float* d_intermidiateOutputsL1, 
							 float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, int numberOfInputs);
void fusedSharedMemLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
						  int numberOfInputs);
void wholeNetInSharedMemLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
							   int numberOfInputs);
void netInSharedMemAndRegsLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
							     int numberOfInputs);
void lastLayerCoopLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
						 int numberOfInputs);
void wholeNetInRegsLaunch(const float* d_inputs, float* d_outputs, float *d_weightsL0, float *d_weightsL1, float *d_weightsL2, 
						  int numberOfInputs);
void wholeNetInRegsHalfLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						      int numberOfInputs);
void reducedBankConflictsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						        int numberOfInputs);
void coalescedWeightsReadsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						        int numberOfInputs);
void inputsLoadHidingLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						          int numberOfInputs);
void betterOccupancyLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
						   int numberOfInputs);
void bestKernelsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2, 
					   int numberOfInputs);
void bestKernelsLaunch(const half* d_inputs, float* d_outputs, half *d_weightsL0, half *d_weightsL1, half *d_weightsL2For5x5, 
					   half *d_weightsL2For100x100, int numberOfInputs);

GpuMlpEngine::~GpuMlpEngine()
{
	freeMemoryOnDevice({ _d_weightsL0, _d_weightsL1, _d_weightsL2, _d_weightsPaddedL0, _d_intermidiateOutputsL0, _d_intermidiateOutputsL1,
						 _d_halfWeightsL0, _d_halfWeightsL1, _d_halfWeightsL2, _d_halfWeightsTransposedL0, 
						 _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, _d_halfWeightsTransposedL2, 
						 _d_halfWeights8WayAddressingL2, _d_halfWeights16WayAddressingL2, _d_halfWeights32WayAddressingL2 });
}

void GpuMlpEngine::setWeights(const Matrix<float> weights[3])
{
	allocateAndSetWeights(weights[0].data, _d_weightsL0);
	allocateAndSetWeights(weights[1].data, _d_weightsL1);
	allocateAndSetWeights(weights[2].data, _d_weightsL2);

	vector<float> W0{weights[0].data.begin(), weights[0].data.end()};
	for (int16_t i = 64; i >= 1; i--)
	{
		W0.insert(W0.begin() + (i * 71), 0.0f);
	}
	allocateAndSetWeights(W0, _d_weightsPaddedL0);

	vector<half> halfW0;
	for (float w : W0)
	{
		halfW0.push_back(__float2half(w));
	}
	vector<half> halfW1;
	for (float w : weights[1].data)
	{
		halfW1.push_back(__float2half(w));
	}
	vector<half> halfW2;
	for (float w : weights[2].data)
	{
		halfW2.push_back(__float2half(w));
	}
	allocateAndSetHalfWeights(halfW0, _d_halfWeightsL0);
	allocateAndSetHalfWeights(halfW1, _d_halfWeightsL1);
	allocateAndSetHalfWeights(halfW2, _d_halfWeightsL2);

	vector<half> halfTransposedW0;
	for (int16_t i = 0; i < 72; i++)
	{
		for (int16_t j = 0; j < 64; j++)
		{
			halfTransposedW0.push_back(halfW0[i + j * 72]);
		}
	}
	vector<half> halfTransposedW1;
	for (int16_t i = 0; i < 64; i++)
	{
		for (int16_t j = 0; j < 64; j++)
		{
			halfTransposedW1.push_back(halfW1[i + j * 64]);
		}
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
	vector<half> halfTransposedW2;
	for (int16_t i = 0; i < 4; i++)
	{
		for (int16_t j = 0; j < 64; j++)
		{
			halfTransposedW2.push_back(halfW2[i + j * 4]);
		}
	}
	vector<half> half8WayAddressingW2;
	for (int16_t i = 0; i < 8; i++)
	{
		for (int16_t j = 0; j < 32; j++)
		{
			half8WayAddressingW2.push_back(halfW2[i + j * 8]);
		}
	}
	vector<half> half16WayAddressingW2;
	for (int16_t i = 0; i < 16; i++)
	{
		for (int16_t j = 0; j < 16; j++)
		{
			half16WayAddressingW2.push_back(halfW2[i + j * 16]);
		}
		for (int16_t j = 0; j < 16; j++)
		{
			half16WayAddressingW2.push_back(halfW2[i + j * 16]);
		}
	}
	vector<half> half32WayAddressingW2;
	for (int16_t i = 0; i < 32; i++)
	{
		for (int16_t j = 0; j < 8; j++)
		{
			half32WayAddressingW2.push_back(halfW2[i + j * 32]);
		}
		for (int16_t j = 0; j < 8; j++)
		{
			half32WayAddressingW2.push_back(halfW2[i + j * 32]);
		}
		for (int16_t j = 0; j < 8; j++)
		{
			half32WayAddressingW2.push_back(halfW2[i + j * 32]);
		}
		for (int16_t j = 0; j < 8; j++)
		{
			half32WayAddressingW2.push_back(halfW2[i + j * 32]);
		}
	}

	allocateAndSetHalfWeights(halfTransposedW0, _d_halfWeightsTransposedL0);
	allocateAndSetHalfWeights(halfTransposedW1, _d_halfWeightsTransposedL1);
	allocateAndSetHalfWeights(half2WayAddressingW2, _d_halfWeights2WayAddressingL2);
	allocateAndSetHalfWeights(halfTransposedW2, _d_halfWeightsTransposedL2);
	allocateAndSetHalfWeights(half8WayAddressingW2, _d_halfWeights8WayAddressingL2);
	allocateAndSetHalfWeights(half16WayAddressingW2, _d_halfWeights16WayAddressingL2);
	allocateAndSetHalfWeights(half32WayAddressingW2, _d_halfWeights32WayAddressingL2);

	// uncomment for 'baselineLaunch' and 'fusedGlobalMemoryLaunch'
	//allocateMemory(_d_intermidiateOutputsL0, 2'097'152 * 64);
	//allocateMemory(_d_intermidiateOutputsL1, 2'097'152 * 64);
}

void GpuMlpEngine::runWithDevicePointers(float* d_outputs, const float* d_inputs, const half* d_inputs_half, int numberOfInputs)
{
	// choose an implementation to run

	//baselineLaunch(d_inputs, d_outputs, _d_intermidiateOutputsL0, _d_intermidiateOutputsL1, _d_weightsL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//fusedGlobalMemoryLaunch(d_inputs, d_outputs, _d_intermidiateOutputsL0, _d_intermidiateOutputsL1, _d_weightsL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//fusedSharedMemLaunch(d_inputs, d_outputs, _d_weightsL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//wholeNetInSharedMemLaunch(d_inputs, d_outputs, _d_weightsL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//netInSharedMemAndRegsLaunch(d_inputs, d_outputs, _d_weightsL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//lastLayerCoopLaunch(d_inputs, d_outputs, _d_weightsL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//wholeNetInRegsLaunch(d_inputs, d_outputs, _d_weightsPaddedL0, _d_weightsL1, _d_weightsL2, numberOfInputs);
	//wholeNetInRegsHalfLaunch(d_inputs_half, d_outputs, _d_halfWeightsL0, _d_halfWeightsL1, _d_halfWeightsL2, numberOfInputs);
	//reducedBankConflictsLaunch(d_inputs_half, d_outputs, _d_halfWeightsL0, _d_halfWeightsL1, _d_halfWeightsL2, numberOfInputs);
	//coalescedWeightsReadsLaunch(d_inputs_half, d_outputs, _d_halfWeightsTransposedL0, _d_halfWeightsTransposedL1, _d_halfWeights2WayAddressingL2, numberOfInputs);
	//inputsLoadHidingLaunch(d_inputs_half, d_outputs, _d_halfWeightsTransposedL0, _d_halfWeightsTransposedL1, _d_halfWeightsTransposedL2, numberOfInputs);
	//betterOccupancyLaunch(d_inputs_half, d_outputs, _d_halfWeightsTransposedL0, _d_halfWeightsTransposedL1, _d_halfWeights16WayAddressingL2, numberOfInputs);
	bestKernelsLaunch(d_inputs_half, d_outputs, _d_halfWeightsTransposedL0, _d_halfWeightsTransposedL1, _d_halfWeights32WayAddressingL2, _d_halfWeights8WayAddressingL2, numberOfInputs);
}

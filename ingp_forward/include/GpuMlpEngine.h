#pragma once

#include "DataLoader.h"
#include <iostream>

class GpuMlpEngine
{
private:
	float *_d_weightsL0 = nullptr;
	float *_d_weightsL1 = nullptr;
	float *_d_weightsL2 = nullptr;
	float *_d_weightsPaddedL0 = nullptr;
	__half *_d_halfWeightsL0 = nullptr;
	__half *_d_halfWeightsL1 = nullptr;
	__half *_d_halfWeightsL2 = nullptr;
	__half *_d_halfWeightsTransposedL0 = nullptr;
	__half *_d_halfWeightsTransposedL1 = nullptr;
	__half *_d_halfWeights2WayAddressingL2 = nullptr;
	__half *_d_halfWeightsTransposedL2 = nullptr;
	__half *_d_halfWeights8WayAddressingL2 = nullptr;
 	__half *_d_halfWeights16WayAddressingL2 = nullptr;
	__half *_d_halfWeights32WayAddressingL2 = nullptr;

	float *_d_intermidiateOutputsL0 = nullptr;
	float *_d_intermidiateOutputsL1 = nullptr;

public:
	~GpuMlpEngine();
	/// <summary>
	/// called once to setup the weight matices of all three layers
	/// weights[layer].dims[0] is the number of outputs 'layer' = n_outputs
	/// weights[layer].dims[1] is the number of inputs of 'layer' = n_inputs
	/// 
	/// weights[layer].dims[1] successive elements provide the weights for one output neuron:
	/// output[0] = w[0]*input[0] + w[1]*input[1] ... w[n_inputs-1]*input[n_inputs-1]
	/// output[1] = w[n_inputs]*input[0] + w[n_inputs+1]*input[1] ... w[n_inputs+n_inputs-1]*input[n_inputs-1]
	/// ...
	/// output[n] = w[n*n_inputs]*input[0] + w[n*n_inputs+1]*input[1] ... w[n*n_inputs+n_inputs-1]*input[n_inputs-1]
	/// 
	/// note that there are no biases, the weights always have the same shape
	/// [64x71, 64x64, 4x64]
	/// </summary>
	/// <param name="weights">note that the weight matrices are on the CPU</param>
	void setWeights(const Matrix<float> weights[3]);

	/// <summary>
	/// called multiple times and measured in terms of performance
	/// must evaluate the three layer network defined by the weights for all numberOfInputs
	/// each of the numberOfInputs corresponds to 71 inputs - one full network evaluation
	/// numberOfInputs can become relatively large
	/// </summary>
	/// <param name="d_outputs">linear array expecting the outputs, where each 4 outputs are placed consecutively in memory, in sequence for all numberOfInputs</param>
	/// <param name="d_inputs">linear array holding the input data (71*numberOfInputs) elements, where all 71 inputs to one network evaluation are next to another in memory</param>
	/// <param name="d_inputs_half">linear array holding the input data (72*numberOfInputs) elements, where all 71 inputs to one network evaluation are next to another in 
	/// memory but each 71 inputs are padded to 72, so each new network input is 4 byte aligned </param>
	/// <param name="numberOfInputs">number of overall network eveluations</param>
	void runWithDevicePointers(float* d_outputs, const float* d_inputs, const __half* d_inputs_half, int numberOfInputs);
};
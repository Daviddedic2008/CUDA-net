#pragma once
#include "math_functions.h"
#include <cmath>
#include "vectorcu.cuh"
#include <vector>

enum class Activator {
	sigmoid10,
	sigmoid,
	bit,
	reLU,
};

__device__ inline float sigmoid10f(const float input) {
	// may be faster depending on implementation of exp10f
	return 1.0f / (1.0f + exp10f(input));
}
__device__ inline float sigmoidf(const float input) {
	return 1.0f / (1.0f + expf(input));
}

__device__ inline float bitActivator(const float input) {
	return input > 0.0f;
}

__device__ inline float reLU(const float input) {
	return (input > 0) * input;
}

struct Neuron {
	
	float output;

	__host__ __device__ Neuron() {
		output = 0.0f;
	}

	__device__ inline void activate(const Activator activatorType) {
		switch (activatorType) {
		case Activator::bit:
			output = bitActivator(output);
			break;
		case Activator::reLU:
			output = reLU(output);
			break;
		case Activator::sigmoid:
			output = sigmoidf(output);
			break;
		case Activator::sigmoid10:
			output = sigmoid10f(output);
			break;
		}
	}
};

struct Layer {
	vectorCU<Neuron> neurons;
	vectorCU<float> weights;
	Activator activatorType;
	unsigned int nextNeurons;
	float bias;

	__host__ Layer(const int numNeurons, const Activator act, const int numNextNeurons, const float bias = 0.0f) : bias(bias){
		// should destruct/free previous mem properly
		nextNeurons = numNeurons;
		neurons = vectorCU<Neuron>(numNeurons);
		for (int i = 0; i < numNeurons; i++) {
			neurons.push_back(Neuron());
			for (int i2 = 0; i2 < numNextNeurons; i2++) {
				weights.push_back(0);
			}
		}
	}

	__host__ ~Layer() {
		neurons.free();
		weights.free();
		// for some odd reason, destructors tagged with host prohibit structs from being passed into kernels :(
		// had to free vectors here instead of giving them a proper destructor
	}

	__host__ void activateLayerImmediate(const Layer prevLayer, const int maxThreadsPerBlock) {
		const int totalThreadCount = this->neurons.size * prevLayer.neurons.size;
		activationKernel << <prevLayer.neurons.size, this->neurons.size, cudaStreamFireAndForget>> > (*this, prevLayer); // less boilerplate to launch
		cudaDeviceSynchronize();
		// maaaaybe could do with stream shenanigans without 2 separate host invocations, idk
		plugKernel << <maxThreadsPerBlock, this->neurons.size / maxThreadsPerBlock + 1, cudaStreamFireAndForget >> > (*this);
		cudaDeviceSynchronize();
	}
};

__global__ void activationKernel(const Layer layer, const Layer previousLayer) {

	__shared__ float SHAREDADD = 0.0f;
	// each block contains enough threads to add to one output neuron
	__syncthreads();

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= layer.neurons.size * previousLayer.neurons.size) { return; } // wohoo branching

	const int layerIdx = idx / previousLayer.neurons.size;
	const int inputIdx = idx % previousLayer.neurons.size;

	Neuron* out = layer.neurons.data;
	// could tail launch child kernel, but the overhead may be too big
	atomicAdd(&SHAREDADD, previousLayer.neurons[inputIdx].output * previousLayer.weights[layerIdx * inputIdx]);

	__syncthreads(); // block-wide access to global memory, accesses to shared memory are faster
	if (threadIdx.x == 0) {
		// can't do without branching, as this would result in race conditons and lots of bad stuff
		atomicAdd(&out[layerIdx].output, SHAREDADD);
	}
}

__global__ void plugKernel(const Layer layer) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= layer.neurons.size) { return; } // wohoo branching

	Neuron* out = layer.neurons.data;
	out[idx].activate(layer.activatorType);
	out[idx].output += layer.bias;
}

struct Model {
	std::vector<Layer> layers;
	Layer inputLayer;
	Layer outputLayer;

	inline Layer& getLayer(const int idx) {
		return idx == 0 ? inputLayer : (idx == layers.size() + 1 ? outputLayer : layers[idx]);
	}

	void evaluate(const int maxThreadsPerBlock) {
		for (int layer = 1; layer < layers.size()+2; layer++) {
			getLayer(layer).activateLayerImmediate(getLayer(layer - 1), maxThreadsPerBlock);
		}
	}

	float cost(const std::vector<const float>& predictedOutput) {
		// mean square cost function
		float err = 0;
		float* outputs = new float[predictedOutput.size()];
		cudaMemcpy(outputs, this->outputLayer.neurons.data, sizeof(float) * predictedOutput.size(), cudaMemcpyDeviceToHost);
		// copy outs over

		for (int idx = 0; idx < outputLayer.neurons.size; idx++) {
			err += (outputs[idx] - predictedOutput[idx]) * (outputs[idx] - predictedOutput[idx]);
		}

		delete outputs;

		err /= predictedOutput.size();
		return err;
	}
};
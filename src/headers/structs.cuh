#pragma once
#include "math_functions.h"
#include <cmath>
#include "vectorcu.cuh"

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

	__host__ Layer(const int numNeurons, const Activator act, const int numNextNeurons) {
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

	__host__ void activateLayer(const Layer& prevLayer) {

	}
};

__global__ void activationSubkernel(Layer layer, Layer previousLayer, const int neuronId) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	Neuron* out = layer.neurons.data; // because cuda wont let me reference a non-lvalue passed into kernel, i must do this....
	atomicAdd(&out[neuronId].output, previousLayer.neurons[idx].output * previousLayer.weights[neuronId * idx]);
}

__global__ void activationKernel(const Layer layer, const Layer previousLayer) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	activationSubkernel << <32, 1, 0, cudaStreamTailLaunch>> > (layer, previousLayer, idx); // tmp call
	Neuron* out = layer.neurons.data;

	out[idx].activate(layer.activatorType);
}
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <cmath>

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
	Activator activatorType;
	float output;

	__device__ Neuron(Activator activator) {
		activatorType = activator;
	}

	__device__ inline void activate(const float input) {
		switch (activatorType) {
		case Activator::bit:
			output = bitActivator(input);
			break;
		case Activator::reLU:
			output = reLU(input);
			break;
		case Activator::sigmoid:
			output = sigmoidf(input);
			break;
		case Activator::sigmoid10:
			output = sigmoid10f(input);
			break;
		}
	}
};

struct Layer {

};
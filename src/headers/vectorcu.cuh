#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <cmath>
#include <stdio.h>

template<typename T>
struct vectorCU {
	unsigned int size;
	T* data;
	unsigned int range;

	__host__ vectorCU(const int sz = 20) : size(sz), range(-1) {
		cudaMalloc(&data, sizeof(T) * sz);
	}

	__host__ ~vectorCU() {
		cudaFree(data);
	}

	inline __device__ T operator[](const unsigned int idx) const{
		return data[idx];
	}

	__host__ void push_back(const T& v) {
		if (range+1 < size) {
			cudaMemcpy(&data[range+1], &v, sizeof(T), cudaMemcpyHostToDevice);
			range++;
			return;
		}

		size = size == 0 ? 1 : size * 2;

		T* tmp;

		cudaMalloc(&tmp, sizeof(T) * size);
		range++;

		if (range > 0) {
			cudaMemcpy(tmp, data, sizeof(T) * (range), cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(&tmp[range], &v, sizeof(T), cudaMemcpyHostToDevice);

		cudaFree(data);
		
		data = tmp;
	}

	__host__ void pop_back() {
		if (range >= size / 2) {
			range--;
			return;
		}

		T* tmp;
		cudaMalloc(&tmp, sizeof(T) * size / 2);
		cudaMemcpy(tmp, data, sizeof(T) * (size/2), cudaMemcpyDeviceToDevice);
		range = size / 2 - 1;
		cudaFree(data);
		data = tmp;
	}
};

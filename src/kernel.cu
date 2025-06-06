#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "headers/kernel.h"
#include "headers/vectorcu.cuh"

#include <stdio.h>

__global__ void testKernel() {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	printf("ovr id: %d\n", idx);
}

__global__ void testVectorKernel(const vectorCU<int> vec) {
	printf("%d\n", vec[1]);
}

extern "C" void launchKernel(const unsigned int numBlocks, const unsigned int numThreads) {
	testKernel << <numThreads, numBlocks >> > ();
	cudaDeviceSynchronize();
}

extern "C" void testVector() {
	vectorCU<int> vec(1);
	vec.push_back(1);
	vec.push_back(2);
	vec.push_back(3);
	printf("add: %s\n", cudaGetErrorString(cudaGetLastError()));
	testVectorKernel << <1, 1 >> > (vec);
	printf("kernel: % s\n", cudaGetErrorString(cudaGetLastError()));
	cudaDeviceSynchronize();
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdio.h>

__global__ void testKernel() {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	printf("ovr id: %d\n", idx);
}

extern "C" void launchKernel(const unsigned int numBlocks, const unsigned int numThreads) {
	testKernel << <numThreads, numBlocks >> > ();
	cudaDeviceSynchronize();
}
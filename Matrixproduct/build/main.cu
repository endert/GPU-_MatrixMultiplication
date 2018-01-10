#include <cstdlib>
#include <iostream>
#include "cuda_util.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include "Matrix.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> tpoint;

__global__ void matrixMultiplication(float* matrix1, float* matrix2, float* result, int m1_cols, int m1_rows, int m2_cols, int m2_rows)
{
	if (m1_cols != m2_rows)
	{
		throw std::length_error("False Matrix size! Can't mulitply.");
	}

	int res_X = blockDim.x * blockIdx.x + threadIdx.x;
	int res_Y = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = threadIdx.x % m2_cols;

	for (int k = 0; k < m1_cols; ++k)
	{
		result[threadIdx.x] += matrix1[res_X + k] * matrix2[offset + m2_rows * k];
	}
}


bool initDevice(int& device_handle, int& max_threads_per_block) {

	int deviceCount = 0;
	checkErrorsCuda(cudaGetDeviceCount(&deviceCount));

	if (0 == deviceCount) {
		std::cerr << "initDevice() : No CUDA device found." << std::endl;
		return false;
	}

	// one could implement more complex logic here to find the fastest device
	if (deviceCount > 1) {
		std::cerr << "initDevice() : Multiple CUDA devices found. Using first one." << std::endl;
	}

	// set the device
	checkErrorsCuda(cudaSetDevice(device_handle));

	cudaDeviceProp device_props;
	checkErrorsCuda(cudaGetDeviceProperties(&device_props, device_handle));
	max_threads_per_block = device_props.maxThreadsPerBlock;

	return true;
}


int main (int /*argc*/, char** /*argv*/)
{
	int i = 3, j = 3, k = 3;

	Matrix<float> matrix1_host(i, j);
	Matrix<float> matrix2_host(j, k);
	Matrix<float> result_host(matrix1_host.getRows(), matrix2_host.getCols());

	matrix1_host.fillMatrix();
	matrix1_host.printMatrix();

	matrix2_host.fillMatrix();
	matrix2_host.printMatrix();

	// check execution environment
	int device_handle = 0;
	int max_threads_per_block = 0;
	if (!initDevice(device_handle, max_threads_per_block)) {
		return EXIT_FAILURE;
	}

	// initialize memory
	float* result_device = nullptr;
	float* matrix1_device = nullptr;
	float* matrix2_device = nullptr;

	// allocate device memory
	checkErrorsCuda(cudaMalloc((void **)&result_device, sizeof(float) * result_host.getTotalSize()));
	checkErrorsCuda(cudaMalloc((void **)&matrix1_device, sizeof(float) * matrix1_host.getTotalSize()));
	checkErrorsCuda(cudaMalloc((void **)&matrix2_device, sizeof(float) * matrix2_host.getTotalSize()));

	// copy device memory
	checkErrorsCuda(cudaMemcpy((void*)matrix1_device, &matrix1_host.m_ptValues, sizeof(float) * matrix1_host.getTotalSize(),
		cudaMemcpyHostToDevice));
	checkErrorsCuda(cudaMemcpy((void*)matrix2_device, &matrix2_host.m_ptValues, sizeof(float) * matrix2_host.getTotalSize(),
		cudaMemcpyHostToDevice));

	// determine thread layout
	dim3 num_threads_per_block(1, 1, 1);
	dim3 num_blocks(1, 1, 1);

	int max_threads_per_block_sqrt = (int)std::sqrt((double)max_threads_per_block);
	assert(32 == max_threads_per_block_sqrt);

	num_blocks.x = result_host.getCols() / max_threads_per_block_sqrt;
	if (0 != result_host.getCols() % max_threads_per_block_sqrt)
	{
		num_blocks.x++;
	}

	num_blocks.y = result_host.getRows() / max_threads_per_block_sqrt;
	if (0 != result_host.getRows() % max_threads_per_block_sqrt)
	{
		num_blocks.y++;
	}

	num_threads_per_block.x = max_threads_per_block_sqrt;
	num_threads_per_block.y = max_threads_per_block_sqrt;

	// run kernel
	tpoint t_start = std::chrono::high_resolution_clock::now();
	//convSeparable<kernel_supp_half> << < num_blocks, num_threads_per_block >> >(kernel_device, image_device, image_conv_device, image.n_rows);
	matrixMultiplication << <num_blocks, num_threads_per_block >> > (matrix1_device, matrix2_device, result_device, matrix1_host.getCols(), matrix1_host.getRows(), matrix2_host.getCols(), matrix2_host.getRows());

	tpoint t_end = std::chrono::high_resolution_clock::now();
	double wall_clock = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	std::cerr << "Execution time: " << wall_clock << " ms." << std::endl;

	checkLastCudaError("Kernel execution failed");
	cudaDeviceSynchronize();

	// copy result back to host
	checkErrorsCuda(cudaMemcpy(&result_host.m_ptValues, result_device,	sizeof(float) * result_host.getTotalSize(),
		cudaMemcpyDeviceToHost));
}


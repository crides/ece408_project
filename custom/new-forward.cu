#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

#define cuda_check(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            std::cout << "Failed to run stmt " << #stmt << std::endl;                       \
            std::cout << "Got CUDA error ...  " << cudaGetErrorString(err) << std::endl;    \
            exit(-1);                                                        \
        }                                                                     \
    } while(0)

#define y4d(y, i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(x, i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(k, i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define for_in(v, max) for (int v = 0; v < max; v++)

#define INT_FACTOR 4096
#define INT_TYPE int

#define TILE_WIDTH 32
__global__ void conv_forward_kernel(float *y, INT_TYPE *temp, const INT_TYPE *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int w_grid = (int) ceil(((float) W_out) / TILE_WIDTH);

	// Recover the blockidx in x and y dim using Z
	const int H_bi = blockIdx.z / w_grid;
	const int W_bi = blockIdx.z % w_grid;
	
	// h and w refers to both the output and input index(some of the threads will be turned off during output)
    const int h = H_bi * TILE_WIDTH + threadIdx.y;
    const int w = W_bi * TILE_WIDTH + threadIdx.x;
    const int b = blockIdx.x;
    const int m = blockIdx.y;

    if (h < H_out && w < W_out) {
        INT_TYPE acc = 0;
        for_in(c, C) {
            for_in(p, K) {
                for_in(q, K) {
                    acc += x4d(temp, b, c, h + p, w + q) * k4d(k, m, c, p, q) / INT_FACTOR;
                }
            }
        }
        y4d(y, b, m, h, w) = __int2float_rd(acc) / INT_FACTOR;
    }
}

__global__ void input_to_fixed_point(const float *input, INT_TYPE *output, const int B, const int M, const int C, const int H, const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int w_grid = (int) ceil(((float) W_out) / TILE_WIDTH);

	// Recover the blockidx in x and y dim using Z
	const int H_bi = blockIdx.z / w_grid;
	const int W_bi = blockIdx.z % w_grid;
	
	// h and w refers to both the output and input index(some of the threads will be turned off during output)
    const int h = H_bi * TILE_WIDTH + threadIdx.y;
    const int w = W_bi * TILE_WIDTH + threadIdx.x;
    const int b = blockIdx.x;

    if (h < H && w < W) {
        for_in(c, C) {
            float in = x4d(input, b, c, h, w);
            x4d(output, b, c, h, w) = __float2int_rd(in * INT_FACTOR);
        }
    }
}

__global__ void kernel_to_fixed_point(const float *input, INT_TYPE *output, const int B, const int M, const int C, const int H, const int W, const int K) {
    int m = blockIdx.x, c = blockIdx.y, p = threadIdx.x, q = threadIdx.y;
    if (m < M && c < C && p < K && q < K) {
        k4d(output, m, c, p, q) = __float2int_rd(k4d(input, m, c, p, q) * INT_FACTOR);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *dev_x, *dev_y, *dev_k;
    INT_TYPE *temp, *dev_k_16;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate memory and copy over the relevant data structures to the GPU
    size_t x_len = B * C * H * W;
    size_t y_size = B * M * H_out * W_out * sizeof(float);
    size_t k_len = M * C * K * K;
    cuda_check(cudaMalloc(&dev_x, x_len * sizeof(float)));
    cuda_check(cudaMalloc(&temp, x_len * sizeof(INT_TYPE)));
    cuda_check(cudaMalloc(&dev_y, y_size));
    cuda_check(cudaMalloc(&dev_k, k_len * sizeof(float)));
    cuda_check(cudaMalloc(&dev_k_16, k_len * sizeof(INT_TYPE)));
    cuda_check(cudaMemcpy(dev_k, host_k, k_len * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dim_grid_kernel(M, C, 1);
    dim3 dim_block_kernel(K, K, 1);
    kernel_to_fixed_point<<<dim_grid_kernel, dim_block_kernel>>>(dev_k, dev_k_16, B, M, C, H, W, K);

    const int H_grid = ceil((float) H_out / TILE_WIDTH);
    const int W_grid = ceil((float) W_out / TILE_WIDTH);
    const int Z = H_grid * W_grid;
    dim3 dim_grid(B, M, Z);
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    const int Z_input = ceil((float) H / TILE_WIDTH) * ceil((float) W / TILE_WIDTH);
    dim3 dim_grid_conv_input(B, M, Z_input);

    cuda_check(cudaMemcpy(dev_x, host_x, x_len * sizeof(float), cudaMemcpyHostToDevice));
    input_to_fixed_point<<<dim_grid_conv_input, dim_block>>>(dev_x, temp, B, M, C, H, W, K);
    conv_forward_kernel<<<dim_grid, dim_block>>>(dev_y, temp, dev_k_16, B, M, C, H, W, K);
    cuda_check(cudaMemcpy(host_y, dev_y, y_size, cudaMemcpyDeviceToHost));

    // Free device memory
    cuda_check(cudaFree(dev_x));
    cuda_check(cudaFree(dev_k_16));
    cuda_check(cudaFree(temp));
    cuda_check(cudaFree(dev_y));
    cuda_check(cudaFree(dev_k));
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define cuda_check(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            std::cout << "Failed to run stmt " << #stmt << std::endl;                       \
            std::cout << "Got CUDA error ...  " << cudaGetErrorString(err) << std::endl;    \
            exit(-1);                                                        \
        }                                                                     \
    } while(0)

#define TILE_WIDTH 32
__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int w_grid = (int) ceil(((float) W_out) / TILE_WIDTH);

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define for_in(v, max) for (int v = 0; v < max; v++)

    int m = blockIdx.z;
    int h = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (h < H_out && w < W_out) {
        float acc = 0;
        for_in(c, C) {
            for_in(p, K) {
                for_in(q, K) {
                    acc += x4d(0, c, h + p, w + q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(0, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *dev_x, *dev_y, *dev_k;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate memory and copy over the relevant data structures to the GPU
    size_t x_size = C * H * W * sizeof(float);
    size_t y_size = M * H_out * W_out * sizeof(float);
    size_t k_size = M * C * K * K * sizeof(float);
    cuda_check(cudaMalloc(&dev_x, x_size));
    cuda_check(cudaMalloc(&dev_y, y_size));
    cuda_check(cudaMalloc(&dev_k, k_size));
    cuda_check(cudaMemcpy(dev_k, host_k, k_size, cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel TODO
    dim3 dim_grid(ceil((float) W_out / TILE_WIDTH), ceil((float) H_out / TILE_WIDTH), M);
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);

    for_in(b, B) {
        cuda_check(cudaMemcpy(dev_x, &host_x[b * C * H * W], x_size, cudaMemcpyHostToDevice));
        conv_forward_kernel<<<dim_grid, dim_block>>>(dev_y, dev_x, dev_k, M, C, H, W, K);
        cuda_check(cudaMemcpy(&host_y[b * M * H_out * W_out], dev_y, y_size, cudaMemcpyDeviceToHost));
    }

    // Free device memory
    cuda_check(cudaFree(dev_x));
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

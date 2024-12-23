extern "C" {
#include "hostFE.h"
}
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

float *d_filter, *d_inputImage;

__global__ void convolutionKernel(const float *d_inputImage, const int H, const int W, const float *d_filter,
                                  const int filterWidth, float *d_outputImage)
{
    // Dynamically Allocated Shared Memory
    extern __shared__ float s_filter[];
    for (int i = 0; i < filterWidth * filterWidth; i++) {
        s_filter[i] = d_filter[i];
    }
    __syncthreads();

    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx >= W || y_idx >= H)
        return;

    int halfWidth = filterWidth >> 1;

    float sum = 0;
    int offset_y = -min(halfWidth, y_idx), offset_x = -min(halfWidth, x_idx);
    for (int h = y_idx + offset_y, f_h = halfWidth + offset_y; h <= y_idx + min(halfWidth, H - y_idx - 1); h++, f_h++) {
        for (int w = x_idx + offset_x, f_w = halfWidth + offset_x; w <= x_idx + min(halfWidth, W - x_idx - 1);
             w++, f_w++) {
            sum += d_inputImage[h * W + w] * s_filter[f_h * filterWidth + f_w];
        }
    }

    d_outputImage[y_idx * W + x_idx] = sum;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage,
            cl_device_id *device, cl_context *context, cl_program *program)
{
    // printf("This is CUDA version\n");
    static int t = 0;

    // Allocate device memory
    int filterSize = filterWidth * filterWidth, imageSize = imageHeight * imageWidth;

    if (t == 0) {
        t++;
        cudaMalloc(&d_filter, filterSize * sizeof(float));
        cudaMemcpy(d_filter, filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_inputImage, imageSize * sizeof(float));
        cudaMemcpy(d_inputImage, inputImage, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    }
    float *d_outputImage;
    cudaMalloc(&d_outputImage, imageSize * sizeof(float));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(imageWidth / block.x, imageHeight / block.y);
    convolutionKernel<<<grid, block, filterSize * sizeof(float)>>>(d_inputImage, imageHeight, imageWidth, d_filter,
                                                                   filterWidth, d_outputImage);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage, d_outputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
}

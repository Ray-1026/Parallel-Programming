#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define GROUP_SIZE 4

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_img, int resX,
                             int maxIterations, size_t pitch)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE, thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float c_re, c_im = lowerY + thisY * stepY;

    int *row = (int *)((char *)d_img + thisY * pitch);
    for (int g = 0; g < GROUP_SIZE; g++) {
        c_re = lowerX + thisX * stepX;
        float z_re = c_re, z_im = c_im;

        int i;
        for (i = 0; i < maxIterations; i++) {
            float z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
            if (z_re_sq + z_im_sq > 4.0f)
                break;

            z_im = c_im + (2.0f * z_re * z_im);
            z_re = c_re + (z_re_sq - z_im_sq);
        }

        row[thisX++] = i;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY, *h_img, *d_img;
    size_t pitch;

    cudaHostAlloc(&h_img, size * sizeof(int), cudaHostAllocDefault);
    cudaMallocPitch(&d_img, &pitch, resX * sizeof(int), resY);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(resX / (BLOCK_SIZE * GROUP_SIZE), resY / BLOCK_SIZE);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, d_img, resX, maxIterations, pitch);

    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_img, size * sizeof(int));

    cudaFree(d_img);
    cudaFreeHost(h_img);
}

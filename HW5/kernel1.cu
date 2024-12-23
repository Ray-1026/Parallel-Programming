#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_img, int resX,
                             int maxIterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x, thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float c_re = lowerX + thisX * stepX, c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im;

    int i;
    for (i = 0; i < maxIterations; i++) {
        float z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
        if (z_re_sq + z_im_sq > 4.0f)
            break;

        z_im = c_im + (2.0f * z_re * z_im);
        z_re = c_re + (z_re_sq - z_im_sq);
    }

    d_img[thisY * resX + thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY;
    int *h_img = (int *)malloc(size * sizeof(int));
    int *d_img;
    cudaMalloc(&d_img, size * sizeof(int));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, d_img, resX, maxIterations);

    cudaMemcpy(h_img, d_img, size * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, h_img, size * sizeof(int));

    free(h_img);
    cudaFree(d_img);
}

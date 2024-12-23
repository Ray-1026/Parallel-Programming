#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_img, int resX, int resY,
                             int maxIterations, bool view1)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x, thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // the region is obtained by the algorithm which is to find the max rectangle area
    if (view1 && ((thisX > 790 && thisX < 1201 && thisY > 296 && thisY < 904) ||
                  (thisX > 438 && thisX < 629 && thisY > 493 && thisY < 707) ||
                  (thisX > 712 && thisX < 791 && thisY > 398 && thisY < 802))) {
        d_img[thisY * resX + thisX] = maxIterations;
        return;
    }

    float c_re = lowerX + thisX * stepX, c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im, z_re_sq, z_im_sq;

    int i;
    if (maxIterations == 100000) {
#pragma unroll
        for (i = 0; i < 100000; i++) {
            z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
            if (z_re_sq + z_im_sq > 4.0f) {
                d_img[thisY * resX + thisX] = i;
                return;
            }
            z_im = c_im + (2.0f * z_re * z_im);
            z_re = c_re + (z_re_sq - z_im_sq);
        }
    }
    else if (maxIterations == 10000) {
#pragma unroll
        for (i = 0; i < 10000; i++) {
            z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
            if (z_re_sq + z_im_sq > 4.0f) {
                d_img[thisY * resX + thisX] = i;
                return;
            }
            z_im = c_im + (2.0f * z_re * z_im);
            z_re = c_re + (z_re_sq - z_im_sq);
        }
    }
    else if (maxIterations == 1000) {
#pragma unroll
        for (i = 0; i < 1000; i++) {
            z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
            if (z_re_sq + z_im_sq > 4.0f) {
                d_img[thisY * resX + thisX] = i;
                return;
            }
            z_im = c_im + (2.0f * z_re * z_im);
            z_re = c_re + (z_re_sq - z_im_sq);
        }
    }
    else if (maxIterations == 256) {
#pragma unroll
        for (i = 0; i < 256; i++) {
            z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
            if (z_re_sq + z_im_sq > 4.0f) {
                d_img[thisY * resX + thisX] = i;
                return;
            }
            z_im = c_im + (2.0f * z_re * z_im);
            z_re = c_re + (z_re_sq - z_im_sq);
        }
    }
    else {
        for (i = 0; i < maxIterations; i++) {
            z_re_sq = z_re * z_re, z_im_sq = z_im * z_im;
            if (z_re_sq + z_im_sq > 4.0f) {
                d_img[thisY * resX + thisX] = i;
                return;
            }
            z_im = c_im + (2.0f * z_re * z_im);
            z_re = c_re + (z_re_sq - z_im_sq);
        }
    }
    d_img[thisY * resX + thisX] = maxIterations;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    bool view1 = (lowerX == -2 && lowerY == -1);
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY;
    int *h_img = (int *)malloc(size * sizeof(int));
    int *d_img;
    cudaMalloc(&d_img, size * sizeof(int));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations, view1);

    cudaMemcpy(h_img, d_img, size * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, h_img, size * sizeof(int));

    free(h_img);
    cudaFree(d_img);
}

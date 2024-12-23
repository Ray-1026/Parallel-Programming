#include "hostFE.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

cl_mem d_filter, d_inputImage;

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage,
            cl_device_id *device, cl_context *context, cl_program *program)
{
    cl_int status;
    static int t = 0;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);

    if (t == 0) {
        t++;
        d_filter = clCreateBuffer(*context, 0, filterSize * sizeof(float), NULL, &status);
        d_inputImage = clCreateBuffer(*context, 0, imageSize * sizeof(float), NULL, &status);

        clEnqueueWriteBuffer(queue, d_filter, CL_MEM_READ_ONLY, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_inputImage, CL_MEM_READ_ONLY, 0, imageSize * sizeof(float), inputImage, 0, NULL,
                             NULL);
    }
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, &status);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_filter);
    status = clSetKernelArg(kernel, 1, sizeof(int), &filterWidth);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_inputImage);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_outputImage);

    size_t global_worksize[2] = {imageWidth, imageHeight};
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_worksize, NULL, 0, NULL, NULL);

    status =
        clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);
}
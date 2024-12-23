__kernel void convolution(__constant float *filter,
                            const int filterWidth,
                            __global const float *inputImage,
                            __global float *outputImage)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int W = get_global_size(0);
    int H = get_global_size(1);
    int halfWidth = filterWidth >> 1;
    float sum = 0;

    for (int h = -halfWidth; h <= halfWidth; h++) {
        for (int w = -halfWidth; w <= halfWidth; w++) {
            int row = y + h, col = x + w;

            if (row >= 0 && row < H && col >= 0 && col < W)
                sum = mad(inputImage[row * W + col], filter[(h + halfWidth) * filterWidth + w + halfWidth], sum);
        }
    }
    outputImage[y * W + x] = sum;
}
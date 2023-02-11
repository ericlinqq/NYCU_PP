__kernel void convolution(int filterWidth, 
                          __constant float *filter,
                          __read_only __global const float *inputImage, 
                          __write_only __global float *outputImage) 
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    int halfFilterSize = filterWidth / 2;

    float sum = 0.f;
    for (int row = -halfFilterSize; row <= halfFilterSize; row++) {
        for (int col = -halfFilterSize; col <= halfFilterSize; col++) {
            if (filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize] != 0 && 
                ix + col >= 0 && ix + col < imageWidth &&
                iy + row >= 0 && iy + row < imageHeight) {
                    sum += inputImage[(iy + row) * imageWidth + ix + col] * 
                    filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize];
            }
        }
    }
    outputImage[iy * imageWidth + ix] = sum;
}

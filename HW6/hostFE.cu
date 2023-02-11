#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "hostFE.h"
}

#define BLOCK_SIZE 25

__global__ void convolutionKernel(int filterWidth,
                           float *filter,
                           int imageHeight,
                           int imageWidth,
                           float *inputImage,
                           float *outputImage) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int halfFilterSize = filterWidth / 2;
    float sum = 0.f;
    for (int row = -halfFilterSize; row <= halfFilterSize; row++) {
        for (int col = -halfFilterSize; col <= halfFilterSize; col++) {
            if (filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize] != 0 &&
                iy + row >= 0 && iy + row < imageHeight &&
                ix + col >= 0 && ix + col < imageWidth) {
                sum += inputImage[(iy + row) * imageWidth + ix + col] *
                       filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize];
            }
        }
    }
    outputImage[iy * imageWidth + ix] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    // Allocate device memory
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageWidth * imageHeight * sizeof(float);
    float *filterBuffer, *inputImageBuffer, *outputImageBuffer;
    cudaMalloc(&filterBuffer, filterSize);
    cudaMalloc(&inputImageBuffer, imageSize);
    cudaMalloc(&outputImageBuffer, imageSize);

    // lock area pages
    // cudaHostRegister(outputImage, imageSize, cudaHostRegisterPortable);

    // Copy filter and inputImage from host to device
    cudaMemcpy(filterBuffer, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(inputImageBuffer, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(imageWidth / BLOCK_SIZE, imageHeight / BLOCK_SIZE);
    convolutionKernel<<<grid, block>>>(filterWidth, filterBuffer, imageHeight, imageWidth, 
                                       inputImageBuffer, outputImageBuffer);

    cudaMemcpy(outputImage, outputImageBuffer, imageSize, cudaMemcpyDeviceToHost);

    // cudaHostUnregister(outputImage);
    cudaFree(filterBuffer);
    cudaFree(inputImageBuffer);
    cudaFree(outputImageBuffer);
}
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__device__ int mandel(float2 c, int count)
{
    // .x => re
    // .y => im
    float2 z = c;
    float2 new_z;
    int i;
    for (i = 0; i < count; i++) 
    {

        if (z.x * z.x + z.y * z.y > 4.f)
            break;

        new_z.x = z.x * z.x - z.y * z.y;
        new_z.y = 2.f * z.x * z.y;
        z.x = c.x + new_z.x;
        z.y = c.y + new_z.y;
    }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, int *img, float stepX, float stepY, int maxIterations, int width) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int2 coord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    float2 c = make_float2(lowerX + coord.x * stepX, lowerY + coord.y * stepY);

    img[coord.x + coord.y * width] = mandel(c, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    // lock area pages
    cudaHostRegister(img, size, cudaHostRegisterPortable);

    int *d_img;
    cudaMalloc((void **) &d_img, size);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, d_img, stepX, stepY, maxIterations, resX);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    // unlock area pages
    cudaHostUnregister(img);
}
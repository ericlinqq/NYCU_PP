#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, int* img, float stepX, float stepY, int maxIterations, int pitch, int height) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + stepX * thisX;
    float y = lowerY + stepY * thisY;

    int *ptr = (int *)((char *) img + thisY * pitch) + thisX;
    *ptr = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);

    int *h_img;
    cudaHostAlloc((void **) &h_img, size, cudaHostAllocMapped);

    int *d_img;
    size_t pitch;
    cudaMallocPitch((void **) &d_img, &pitch, sizeof(int) * resX, resY);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, d_img, stepX, stepY, maxIterations, pitch, resY);

    cudaMemcpy2D(h_img, sizeof(int) * resX, d_img, pitch, sizeof(int) * resX, resY, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    
    memcpy(img, h_img, size);
    cudaFreeHost(h_img);
}
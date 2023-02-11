#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define LOCAL_SIZE 25

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{ 
    cl_int status; 
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageWidth * imageHeight * sizeof(float);
    
    // Create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, &status);
    
    // Create buffers on device
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_COPY_HOST_PTR, filterSize, filter, &status);
    cl_mem inputImageBuffer = clCreateBuffer(*context, CL_MEM_COPY_HOST_PTR, imageSize, inputImage, &status);
    cl_mem outputImageBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status); 
               
    // Use the "convolution" function as the kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // Set argumentsBuffer
    clSetKernelArg(kernel, 0, sizeof(int), (void *) &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &filterBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &inputImageBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &outputImageBuffer);

    // Set global and local workgroup sizes
    size_t localws[2] = {LOCAL_SIZE, LOCAL_SIZE};
    size_t globalws[2] = {imageWidth, imageHeight};

    // Execute kernel
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);

    // Copy results from device back to host
    clEnqueueReadBuffer(commandQueue, outputImageBuffer, CL_TRUE, 0, imageSize, (void *) outputImage, 0, NULL, NULL);
    
    // Release object
    clReleaseCommandQueue(commandQueue);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(inputImageBuffer);
    clReleaseMemObject(outputImageBuffer);
    clReleaseKernel(kernel);
}
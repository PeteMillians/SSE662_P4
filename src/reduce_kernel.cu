#include "reduce_kernel.cuh"

__global__ void reduceKernel(float *inputArray, float *output, int arraySize) {
    /*
    Sums together all elements of the input array and updates the output value with that sum

    Arguments:
        inputArray (*float): pointer to an array of **arraySize** float values which will be summed
        output (*float): pointer to CPU memeory which will be updated by the sum
        arraySize (int): the size of the input array
    */

    // Shared sharedSum array between threads
    extern __shared__ float sharedSum[];

    // Compute the index and stride of this kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
   
    //////////////////////////////////////////
    //  THREAD COMPUTES ITS OWN PARTIAL SUM //
    //////////////////////////////////////////

    float partialSum = 0.0f;
    while (idx < arraySize) {
        partialSum += inputArray[idx];
        idx += stride;
    }
    
    // Update the current sharedSum index with the running sum
    sharedSum[threadIdx.x] = partialSum;

    __syncthreads(); // Synchronize threads within the block

    ////////////////////////////////////////////
    //  BLOCK COMPUTES SUM ACROSS ALL THREADS //
    ////////////////////////////////////////////
    
    // Block-level reduction 
    // Continuously splits the block sums in half, adding the sum of the half, and then repeating until there is only one sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) { 
        if (threadIdx.x < offset) { 
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + offset]; 
        } 

        __syncthreads(); 
    }

    // Only add all partiam sums together once per block
    if (threadIdx.x == 0) {
        atomicAdd(output, sharedSum[0]);
    }
}

void launchReduceKernel(float *d_input, float *d_output, int arraySize, int blockSize, int numBlocks) {
    /*
    Method to call the reduceKernel kernel

    Arguments:
        d_input (*float): pointer to an array of float values which will be summed
        d_output (*float): pointer to device memory where the output will be stored
        arraySize (int): the size of the input array that we will use to sum
        blockSize (int): the size of the blocks we will use for the kernel
        numBlocks (int): the number of blocks that we will use for the kernel
    */

    // Call the reduceKernel kernel
    reduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, arraySize);

    // Synchronize the GPU, blocking until kernel is completed
    cudaDeviceSynchronize();
}
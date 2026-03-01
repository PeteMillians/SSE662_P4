# CUDA Kernel Reduction Design

This document describes the design of the CUDA Kernel Reduction module, which uses a simple CUDA kernel to compute the sum of an array of random integers. The objective of this module is to analyze this kernel using a varying amount of blocks and threads per block. However, because the kernel must be a standalone method which implements over several blocks and threads, it must utilize built-in CUDA methods to synchronize threads together.

## Requirements
- Reduction Kernel with ability to handle variable block count and threads per block number
- Sums up array values to produce a total sum output
![Blocks](/docs/blocks.png)
    - The number of blocks in the above example is 4, with 8 threads per block.
    - Each thread will compute all *block index* summation
        - For instance, thread 2 will take block 0 threadIdx 2, block 1 threadIdx 2, block 2 threadIdx 3, etc.
- Index cannot exceed array size
    - If the grid is not an even multiple of num threads * num blocks, we need to make sure the index doesn't go out of bounds
- Synchronization of kernel across threads

## Methods
__global__ void reduceKernel(float *inputArray, float *output, int arraySize) {}
 
    Sums together all elements of the input array and updates the output value with that sum

    Arguments:
        inputArray (*float): pointer to an array of **arraySize** float values which will be summed
        output (*float): pointer to CPU memeory which will be updated by the sum
        arraySize (int): the size of the input array

void launchReduceKernel(float *d_input, float *d_output, int arraySize, int blockSize, int numBlocks) {}

    Method to call the reduceKernel kernel

    Arguments:
        d_input (*float): pointer to an array of float values which will be summed
        d_output (*float): pointer to device memory where the output will be stored
        arraySize (int): the size of the input array that we will use to sum
        blockSize (int): the size of the blocks we will use for the kernel
        numBlocks (int): the number of blocks that we will use for the kernel
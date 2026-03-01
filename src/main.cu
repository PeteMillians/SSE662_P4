#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include "reduce_kernel.cuh"

// Constants for the system
const int MAX = 1000;
const size_t ARRAY_SIZE = 1e7;

// Prototypes
float cpuReduce(float *inputArray, int ARRAY_SIZE);
float* generateInputArray(size_t length, uint64_t seed1, uint64_t seed2);
uint64_t xoroshiro128plus(uint64_t s[2]);
void reduceVaryingBlockSize(float *d_inputArr, float *inputArr, float *d_output, float actualSum);
void reduceVaryingNumBlocks(float *d_inputArr, float *inputArr, float *d_output, float actualSum);

int main(void) {
    /*
    Calls subsequent methods to find the kernel reduction solution
    */
    
    // Allocate space for device-copy arrays
    float *d_inputArr; 
    float *d_output;
    cudaMalloc(&d_inputArr, ARRAY_SIZE * sizeof(float)); 
    cudaMalloc(&d_output, sizeof(float));
    
    // Generate the input array
    float *inputArr = generateInputArray(ARRAY_SIZE, 12345ULL, 67890ULL);
    
    // Compute the accurate sum of the array
    float actualSum = cpuReduce(inputArr, ARRAY_SIZE);
   
    // Print table header once 
    std::cout << "| Number of Blocks | Block Size | Accuracy (%) | Time (us) |\n"; 
    std::cout << "|------------------|------------|--------------|-----------|\n";

    // First, try fixed block size, varying grid size
    reduceVaryingBlockSize(d_inputArr, inputArr, d_output, actualSum);
    std::cout << "|------------------|------------|--------------|-----------|\n";
    
    // Now, compute the time and accuracy if we have varying block size
    reduceVaryingNumBlocks(d_inputArr, inputArr, d_output, actualSum);

    // Free up the allocated memory
    free(inputArr);
    cudaFree(d_inputArr);
    cudaFree(d_output);

    return 0;
}

void reduceVaryingBlockSize(float *d_inputArr, float *inputArr, float *d_output, float actualSum) {
    int blockSize = 256;
    std::vector<int> numBlocks;

    // Create array from 1 - minimum number of blocks needed to cover the array
    int minBlocks = (ARRAY_SIZE + blockSize - 1) / blockSize; 
    int nb = 1; 
    while (nb < minBlocks) { 
        numBlocks.push_back(nb); 
        nb *= 2; 
    } 

    // Need to push back one more time to get that minimum value
    numBlocks.push_back(nb); 
    
    
    // Iterate through each block number combination to find accuracy and time
    for (int i = 0; i < numBlocks.size(); i++) {
        
        float output = 0.0f;
        
        // Copy the input and output array values to the GPU
        cudaMemcpy(d_inputArr, inputArr, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        float zero = 0.0f;
        cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
        
        // Compute the total time it takes to reduce the compute the SUM
        auto startTime = std::chrono::high_resolution_clock::now();
        launchReduceKernel(d_inputArr, d_output, ARRAY_SIZE, blockSize, numBlocks[i]);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        // Copy the GPU output array back to the CPU
        cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute results
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float accuracy = 100.0f - fabs(actualSum - output) / actualSum * 100.0f;
        
        // Print row 
        std::cout << "| " << std::setw(16) << numBlocks[i] 
        << " | " << std::setw(10) << blockSize 
        << " | " << std::setw(12) << accuracy 
        << " | " << std::setw(9) << duration << " |\n";
    }
    
}

void reduceVaryingNumBlocks(float *d_inputArr, float *inputArr, float *d_output, float actualSum) {
    
    // Get the maximum number of threads per block on our device
    cudaDeviceProp prop; 
    cudaGetDeviceProperties(&prop, 0); 
    int maxBlockSize = prop.maxThreadsPerBlock; 

    // Create block sizes based on the max block size
    std::vector<int> blockSizes; 
    for (int bs = 128; bs <= maxBlockSize; bs *= 2) { 
        blockSizes.push_back(bs); 
    }

    // Try the kernel for each block size
    for (int blockSize : blockSizes) {

        // Compute the number of blocks we can have given the block size
        int numBlocks = (ARRAY_SIZE + blockSize - 1) / blockSize;    

        float output = 0.0f;

        // Copy the input and output array values to the GPU
        cudaMemcpy(d_inputArr, inputArr, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        float zero = 0.0f;
        cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
        
        // Compute the total time it takes to reduce the compute the SUM
        auto startTime = std::chrono::high_resolution_clock::now();
        launchReduceKernel(d_inputArr, d_output, ARRAY_SIZE, blockSize, numBlocks);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        // Copy the GPU output array back to the CPU
        cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        // Compute results
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float accuracy = 100.0f - fabs(actualSum - output) / actualSum * 100.0f;

        // Print row 
        std::cout << "| " << std::setw(16) << numBlocks 
        << " | " << std::setw(10) << blockSize 
        << " | " << std::setw(12) << accuracy 
        << " | " << std::setw(9) << duration << " |\n";
    }
}

float cpuReduce(float *inputArray, int ARRAY_SIZE){
    /*
    Iterates through the input array to determine the sum from the host side

    Arguments:
        inputArray (*float): pointer to an array of **ARRAY_SIZE** float values which will be summed
        output (*float): pointer to CPU memeory which will be updated by the sum
        ARRAY_SIZE (int): the size of the input array
    */

    // Initialize output to 0
    float sum = 0.0f;

    // Iterate through each index in input array, adding the value to the output
    for (int i = 0; i < ARRAY_SIZE; i++) {

        // Add the input array value to the output
        sum += inputArray[i];
    }

    // Update output to be the sum
    return sum;
}

float* generateInputArray(size_t length, uint64_t seed1, uint64_t seed2) {
    /*
    Calls the given xoroshiro128plus method to generate a random array of floats of a specific size

    Arguments:
        length (size_t): the size of the array
        seed1 (uint64_t): the first seed for the random number generator
        seed2 (uint64_t): the second seed for the random number generator
    Returns:
        a pointer to an array of floats
    */

    // Allocate memory for the size of the array we want
    float* arr = (float*)malloc(length * sizeof(float));
    
    // Create the seed array to pass into the PRNG 
    uint64_t state[2] = { seed1, seed2 }; 

    // Create the array of random floats using the generator
    for (size_t i = 0; i < length; i++) { 
        uint64_t r = xoroshiro128plus(state); 
        
        // Convert to float in [0,1) 
        float x = (r >> 11) * (1.0f / 9007199254740992.0f); 

        // Multitply by our max value to get it into our determined range
        x *= MAX;

        // Add the float to the array
        arr[i] = x; 
    } 
    
    return arr;
}

uint64_t xoroshiro128plus(uint64_t s[2]) {
    /*
    Creates a random float given random seeds

    Arguments:
        s (List[uint64_t]): the input seeds
    Returns:
        a random float value
    */

    uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    s[1] = (s1 << 36) | (s1 >> 28);
    return result;
}
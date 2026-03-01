# Main Design

This document outlines the design of the main CUDA file, which will generate the random numbers in the input array and run the Kernel Reduction algorithm with varying number of blocks and block sizes. 

## Requiremets
- Can generate an array of random float values of a given array size
- Can iteratively call the Kernel Reduction methods using various pre-determined numbers of blocks and block sizes
- Contains a method to compute the sum of the float values to compare to the GPU sum
- Measure execution time of the kernel call
- Can print the outputs in the following format:

| Number of Blocks | Block Size | Accuracy (%) | Time (ms) |
|------------------|------------|--------------|-----------|

## Methods
void main() {}

    Calls subsequent methods to find the kernel reduction solution

void cpuReduce(float *inputArray, float *output, int arraySize){}

    Iterates through the input array to determine the sum from the host side

    Arguments:
        inputArray (*float): pointer to an array of **arraySize** float values which will be summed
        output (*float): pointer to CPU memeory which will be updated by the sum
        arraySize (int): the size of the input array

float* generateInputArray(size_t length, uint64_t seed1, uint64_t seed2)

    Calls the given xoroshiro128plus method to generate a random array of floats of a specific size

    Arguments:
        length (size_t): the size of the array
        seed1 (uint64_t): the first seed for the random number generator
        seed2 (uint64_t): the second seed for the random number generator
    Returns:
        a pointer to an array of floats

uint64_t xoroshiro128plus(uint64_t s[2]) {}

    Creates a random float given random seeds

    Arguments:
        s (List[uint64_t]): the input seeds
    Returns:
        a random float value
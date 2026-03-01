#pragma once

void launchReduceKernel(float *d_input, float *d_output, int arraySize, int blockSize, int numBlocks);
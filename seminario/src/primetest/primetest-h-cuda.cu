#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAXNUM 100
#define MAXSTR 50
#define MAXRESULTS 1000
#define MAXTHREADS 512

void cudaCheckError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n",cudaGetErrorString(err));
        cudaDeviceReset();
        exit(1);
    }
}

__device__
int isprime(long int value)
{
    int isPrime = 1;
    long int root = sqrt((double)value);

    if (value % 2 == 0)
        return (value == 2);

    for (int factor = 3; factor<=root && isPrime; factor += 2) {
        isPrime = fmod((double)value, (double) factor) > 0.0;
    }

    return isPrime;
}

__global__
void gpu_check_primes(long int *primes, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
      return;
 
    long int primeToTest = primes[i];
    primes[i] = isprime(primeToTest);
}

void runGPU(int start, int end, int numPrimes, int *numResults,
        char firstHalf[MAXNUM][MAXSTR], char secondHalf[MAXNUM][MAXSTR], char strToTest[MAXSTR])
{
    long int memSize;
    long int primeToTest;
    long int *h_primes;
    long int *d_primes;

    memSize = sizeof(long int) * (end-start) * numPrimes;
    h_primes = (long int*) malloc(memSize);
    cudaMalloc((void**)&d_primes, memSize);

    int k = 0;
    for (int i = start; i < end; i++)
        for (int j = 0; j < numPrimes; j++)
        {
            strcpy(strToTest, firstHalf[i]);
            strcat(strToTest, secondHalf[j]);
            primeToTest = atol(strToTest);
            h_primes[k++] = primeToTest;
        }
    cudaMemcpy(d_primes, h_primes, memSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
 
    // Chamada do kernel
    int blocks = ceil( (double)(k+1) / MAXTHREADS);
    gpu_check_primes<<< blocks, MAXTHREADS>>>(d_primes, k);
    cudaDeviceSynchronize();
 
    cudaMemcpy(h_primes, d_primes, memSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; i++)
      *numResults += (h_primes[i] > 0 ? 1 : 0);

    free(h_primes);
    cudaFree(d_primes);
    cudaDeviceReset();
}

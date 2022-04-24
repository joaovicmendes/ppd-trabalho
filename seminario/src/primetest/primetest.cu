#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "quicksort.c"

// #define DEBUG

#define MAXNUM 100
#define MAXSTR 50
#define MAXRESULTS 1000
#define MAXTHREADS 512

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

int main(int argc,char** argv)
{
    char firstHalf[MAXNUM][MAXSTR];
    char secondHalf[MAXNUM][MAXSTR];
    char strToTest[MAXSTR];
    int numPrimes = 0;
    int numResults = 0;
    long int memSize = 0;
    long int primeToTest;
    long int *h_primes;
    long int *d_primes;
    FILE *primesFile = stdin;

    // Leitura da entrada
    fscanf(primesFile, "%d\n", &numPrimes);
    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", firstHalf[i]);

    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", secondHalf[i]);
    fclose(primesFile);
    
    // Definindo os números para serem testados
    memSize = sizeof(long int) * numPrimes * numPrimes;
    h_primes = (long int*) malloc(memSize);
    cudaMalloc((void**)&d_primes, memSize);
 
    int k = 0;
    for (int i = 0; i < numPrimes; i++)
        for (int j = 0; j < numPrimes; j++)
        {
            strcpy(strToTest, firstHalf[i]);
            strcat(strToTest, secondHalf[j]);
            primeToTest = atol(strToTest);
            h_primes[k++] = primeToTest;
        }
    cudaMemcpy(d_primes, h_primes, memSize, cudaMemcpyHostToDevice);

    // Chamada do kernel
    int blocks = ceil( (double)(k+1) / MAXTHREADS);
    gpu_check_primes<<< blocks, MAXTHREADS>>>(d_primes, k);
    cudaDeviceSynchronize();
 
    cudaMemcpy(h_primes, d_primes, memSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; i++)
      numResults += h_primes[i];
 
    free(h_primes);
    cudaFree(d_primes);
    cudaDeviceReset();

    printf("%d\n", numResults);

    return 0;
}
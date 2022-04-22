%%writefile primetest.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "quicksort.c"

// #define DEBUG

#define MAXNUM 100
#define MAXSTR 50
#define MAXRESULTS 1000

__device__
int isprime(long int value)
{
    long int root = sqrt((double)value);
    int prime = 1;

    if (value % 2 == 0)
        return 0;

    for (int factor = 3; factor <= root; factor += 2) {
        int isPrime = fmod((double)value, (double) factor) > 0.0;
        prime = prime ? isPrime : prime;
    }

    return prime;
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
    long int primeToTest;
    long int *primes;
    FILE *primesFile = stdin;

    // Leitura da entrada
    fscanf(primesFile, "%d\n", &numPrimes);
    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", firstHalf[i]);

    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", secondHalf[i]);
    fclose(primesFile);
    
    // Definindo os números para serem testados
    cudaMallocManaged(&primes, sizeof(long int) * numPrimes * numPrimes);
    int k = 0;
    for (int i = 0; i < numPrimes; i++)
        for (int j = 0; j < numPrimes; j++)
        {
            strcpy(strToTest, firstHalf[i]);
            strcat(strToTest, secondHalf[j]);
            primeToTest = atol(strToTest);
            primes[k++] = primeToTest;
        }

    // Chamada do kernel
    gpu_check_primes<<<1, k>>>(primes, k);
    cudaDeviceSynchronize();
 
    for (int i = 0; i < k; i++)
      numResults += primes[i];
 
    cudaFree(primes);
    cudaDeviceReset();

    printf("%d\n", numResults);

    return 0;
}

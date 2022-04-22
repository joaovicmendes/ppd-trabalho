%%writefile sieve.cu
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;
#define IS_PRIME 1
#define NOT_PRIME 0
#define MAX_THREADS 1024


__global__
void gpu_sieve_run(uint *primes, uint n, uint p, uint k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n && i >= k && (i % p == 0))
        primes[i] = NOT_PRIME;
}

uint sieve(uint n)
{
    // Inicializando variáveis
    uint *primes;
    cudaMallocManaged(&primes, (n+1)*sizeof(uint));
 
    for (uint i = 0; i <= n; i++)
      primes[i] = IS_PRIME;
 
    // Sieve
    uint blocks = ceil( (double)(n+1) / MAX_THREADS);
    dim3 dimGrid = {blocks, 1, 1};
    dim3 dimBlock = {MAX_THREADS, 1, 1};

    for (uint p=3; p <= sqrt(n); p++) {
        if (primes[p] == IS_PRIME && p*p <= n) {
            gpu_sieve_run<<< dimGrid, dimBlock >>>(primes, n, p, p*p); 
            cudaDeviceSynchronize(); 
        }
     }
 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n",cudaGetErrorString(err));
        cudaDeviceReset();
        exit(1);
    }
  
    // Contando resultados
    uint count = 1;
    for (uint i = 3; i < n+1; i += 2)
        if (primes[i] == IS_PRIME)
            count++;

    cudaFree(primes);
    cudaDeviceReset();
    return count;
}

int main()
{
    uint N, result;

    int read = scanf("%d", &N);
    if (!read) {
        exit(1);
    } 
    result = sieve(N);
    printf("%d\n", result);
    return 0;
}

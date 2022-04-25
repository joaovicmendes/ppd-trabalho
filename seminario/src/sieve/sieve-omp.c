#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// #define DEBUG /* comment this line to avoid debbuging prints */

typedef unsigned int uint;
#define IS_PRIME 1
#define NOT_PRIME 0

uint floor_sqrt(uint x)
{
    if (x == 0 || x == 1)
    return x;
 
    uint i = 1, result = 1;
    while (result <= x)
    {
      i++;
      result = i * i;
    }
    return i - 1;
}

uint sieve(uint n)
{
    uint sqrt_of_n = floor_sqrt(n);
    uint *primes = malloc( (n+1) * sizeof(uint) );

    omp_set_num_threads(omp_get_num_procs()*2);

    #pragma omp parallel for
    for (uint i=2; i < n+1; i++)
        primes[i] = IS_PRIME;

    for (uint p=3; p <= sqrt_of_n; p += 2)
        if (primes[p] == IS_PRIME)
            #pragma omp parallel for
            for (uint mult = p*p; mult < n+1; mult += p)
                primes[mult] = NOT_PRIME;

    uint count = 1;
    #pragma omp parallel for reduction(+:count)
    for (uint i = 3; i < n+1; i += 2)
            count += primes[i];

    free(primes);
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

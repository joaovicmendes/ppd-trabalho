#include <stdio.h>
#include <stdlib.h>

// #define DEBUG /* comment this line to avoid debbuging prints */

typedef unsigned int uint;
#define IS_PRIME 1
#define NOT_PRIME 0

void print(uint *primes, uint size)
{
    for (uint i=2; i < size; i++)
        if (primes[i])
            printf("%d\n", i);
}

uint sieve(uint n)
{
    uint *primes = malloc( (n+1) * sizeof(uint) );
    for (uint i=2; i < n+1; i++)
        primes[i] = IS_PRIME;

    for (uint p=2; p*p <= n; p++)
        if (primes[p] == IS_PRIME)
            for (uint mult = p*p; mult < n+1; mult += p)
                primes[mult] = NOT_PRIME;

    uint count = 0;
    for (uint i = 2; i < n+1; i++)
        if (primes[i] == IS_PRIME)
            count++;
    
#ifdef DEBUG
    print(primes, n+1);
#endif

    free(primes);
    return count;
}

int main()
{
    uint N, result;

    scanf("%d", &N);
    result = sieve(N);
    printf("%d\n", result);
    return 0;
}

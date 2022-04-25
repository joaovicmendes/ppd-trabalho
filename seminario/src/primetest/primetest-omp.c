#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "quicksort.c"

// #define DEBUG

#define MAXNUM 100
#define MAXSTR 50
#define MAXRESULTS 1000

char firstHalf[MAXNUM][MAXSTR];
char secondHalf[MAXNUM][MAXSTR];
char strToTest[MAXSTR];
long int result[MAXRESULTS];
int numPrimes = 0;

int isprime(long int value)
{
    long int root;
    long int factor=3;
    int prime=1;
    root = sqrtl(value);
    if (value % 2 == 0) {
        return value == 2;
    } 
    while ((factor<=root) && (prime))
    {
        prime = fmod((double)value, (double) factor) > 0.0;
        factor += 2;
    }
    return prime;
}

int main(int argc,char** argv)
{
    FILE *primesFile = stdin;
    int numResults = 0;
    long int primeToTest;

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // Leitura da entrada
    fscanf(primesFile, "%d\n", &numPrimes);
    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", firstHalf[i]);

    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", secondHalf[i]);

    fclose(primesFile);

    // Formando combinações e testando se são primos ou não
    #pragma omp parallel for private(strToTest, primeToTest)
    for (int i = 0; i < numPrimes; i++)
        for (int j = 0; j < numPrimes; j++)
        {
            strcpy(strToTest, firstHalf[i]);
            strcat(strToTest, secondHalf[j]);
            primeToTest = atol(strToTest);
            if (isprime(primeToTest)) {
                omp_set_lock(&writelock);
                result[numResults] = primeToTest;
                numResults++;
                omp_unset_lock(&writelock);
            }
        }
    
#ifdef DEBUG
    // Ordenando resultado e imprimindo
    quicksort(result,0,numResults-1);
    for (int i = 0; i < numResults; i++)
        printf("%ld\n",result[i]);
#else
    printf("%d\n", numResults);
#endif

    return 0;
}

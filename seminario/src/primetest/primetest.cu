%%writefile primetest.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
    long int root = sqrtl(value);
    int prime = 1;

    if (value % 2 == 0)
        return 0;

    for (int factor = 3; factor <= root; factor += 2) {
        int isPrime = fmod((double)value, (double) factor) > 0.0;
        prime = prime ? isPrime : prime;
    }

    return prime;
}

void quicksort(long int *primes,int first,int last)
{
    int i, j, pivot;
    long int temp;

    if (first < last)
    {
        pivot = first;
        i = first;
        j = last;
        while (i < j)
        {
            while (primes[i] <= primes[pivot] && i<last)
                i++;
            while(primes[j] > primes[pivot])
                j--;

            if (i < j)
            {
                temp = primes[i];
                primes[i] = primes[j];
                primes[j] = temp;
            }
        }

        temp = primes[pivot];
        primes[pivot] = primes[j];
        primes[j] = temp;

        quicksort(primes,first,j-1);
        quicksort(primes,j+1,last);
    }
}


int main(int argc,char** argv)
{
    FILE *primesFile = stdin;
    int numResults = 0;
    long int primeToTest;

    // Leitura da entrada
    fscanf(primesFile, "%d\n", &numPrimes);
    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", firstHalf[i]);

    for (int i = 0; i < numPrimes; i++)
        fscanf(primesFile,"%s\n", secondHalf[i]);

    fclose(primesFile);

    // Formando combinações e testando se são primos ou não
    for (int i = 0; i < numPrimes; i++)
        for (int j = 0; j < numPrimes; j++)
        {
            strcpy(strToTest, firstHalf[i]);
            strcat(strToTest, secondHalf[j]);
            primeToTest = atol(strToTest);
            if (isprime(primeToTest)) {
                result[numResults] = primeToTest;
                numResults++;
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

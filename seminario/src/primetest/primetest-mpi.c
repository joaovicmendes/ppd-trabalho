#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
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
    int totalResults = 0;
    long int primeToTest;
    int numTasks;
    int taskId;
 
    // Inicializando ambiente MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskId);

    // A main lê a entrada e repassa para os demais
    if (taskId == 0) {
        fscanf(primesFile, "%d\n", &numPrimes);
        for (int i = 0; i < numPrimes; i++)
            fscanf(primesFile,"%s\n", firstHalf[i]);

        for (int i = 0; i < numPrimes; i++)
            fscanf(primesFile,"%s\n", secondHalf[i]);

        fclose(primesFile);
    }
    // Main repassa o tamanho dos vetores para trabalhadores e o conteúdo
    MPI_Bcast(&numPrimes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < numPrimes; i++) {
        MPI_Bcast(firstHalf[i], MAXSTR, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(secondHalf[i], MAXSTR, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Formando combinações e testando se são primos ou não
    int start = (numPrimes/numTasks) * taskId;
    int end = (numPrimes/numTasks) * (taskId+1);
    if (taskId == numTasks-1)
      end = numPrimes;
 
    for (int i = start; i < end; i++)
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
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Todos os trabalhadores enviam seu numResults para a main
    MPI_Reduce(&numResults, &totalResults, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (taskId == 0) {
        printf("%d\n", totalResults);
    }
 
    MPI_Finalize();
    return 0;
}

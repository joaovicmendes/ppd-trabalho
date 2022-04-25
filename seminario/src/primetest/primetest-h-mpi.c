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

void runGPU(int start, int end, int numPrimes, int *numResults,
            char firstHalf[MAXNUM][MAXSTR],
            char secondHalf[MAXNUM][MAXSTR],
            char strToTest[MAXSTR]);

int main(int argc,char** argv)
{
    char firstHalf[MAXNUM][MAXSTR];
    char secondHalf[MAXNUM][MAXSTR];
    char strToTest[MAXSTR];
    int numTasks;
    int taskId;
    int numPrimes = 0;
    int numResults = 0;
    int totalResults = 0;
    long int memSize = 0;
    long int primeToTest;
    long int *h_primes;
    long int *d_primes;
    FILE *primesFile = stdin;
 
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

    // Main repassa o tamanho dos vetores e o conteúdo para cada trabalhador
    MPI_Bcast(&numPrimes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < numPrimes; i++) {
        MPI_Bcast(firstHalf[i], MAXSTR, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(secondHalf[i], MAXSTR, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (numTasks > numPrimes) {
        printf("%d processos para apenas %d primos. Diminua o número de processos\n", numTasks, numPrimes);
        MPI_Finalize();
        exit(0);
    }

    // Dividindo blocos
    int start = (numPrimes/numTasks) * taskId;
    int end = (numPrimes/numTasks) * (taskId+1);
    if (taskId == numTasks-1)
      end = numPrimes;
 
    // Chamando a GPU
    runGPU(start, end, numPrimes, &numResults, firstHalf, secondHalf, strToTest);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Todos os trabalhadores enviam seu numResults para a main
    MPI_Reduce(&numResults, &totalResults, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (taskId == 0) {
        printf("%d\n", totalResults);
    }
 
    MPI_Finalize();
    return 0;
}

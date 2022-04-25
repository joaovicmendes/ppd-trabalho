UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	FLAGS=-Xpreprocessor -fopenmp -lomp
else
	FLAGS=-fopenmp 
endif

CC=gcc

all: sieve sieve-omp

sieve: sieve.c
	$(CC) sieve.c -o sieve $(FLAGS)

sieve-omp: sieve-omp.c
	$(CC) sieve-omp.c -o sieve-omp $(FLAGS)

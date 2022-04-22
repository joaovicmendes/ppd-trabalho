UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	OMP_FLAGS=-Xpreprocessor -fopenmp -lomp
else
	OMP_FLAGS=-fopenmp 
endif

FLAGS=-O3 -Wall


all: primetest primetest-omp

primetest: primetest.c
	$(CC) primetest.c $(FLAGS) -lm -o primetest

primetest-omp: primetest-omp.c
	$(CC) primetest-omp.c $(FLAGS) $(OMP_FLAGS) -lm -o primetest-omp

clean:
	rm -rf primetest primetest-omp

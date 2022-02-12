# OSTYPE=darwin21.0
# OSTYPE=linux-gnu
# uname Darwin
# uname Linux

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	CFLAGS=-Xpreprocessor -fopenmp 
	LFLAGS=-lomp
else
	CFLAGS=-fopenmp 
	LFLAGS=
endif

CC=gcc

FLAGS=-O3

all: matriz matriz-omp

matriz: matriz.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) matriz.c -o matriz 

matriz-omp: matriz-omp.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) matriz-omp.c -o matriz-omp $(CFLAGS) $(LFLAGS)

clean:
	rm -f matriz matriz-omp


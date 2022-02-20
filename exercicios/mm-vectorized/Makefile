UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	OMP_FLAGS=-Xpreprocessor -fopenmp -lomp
	VEC_FLAGS=-O3
else
	OMP_FLAGS=-fopenmp
	VEC_FLAGS=-O3 -fopt-info-optall-optimized -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-note
endif

CC=gcc

all: mm-base mm-vec mm-omp mm-omp-vec

mm-base: mm.c
	$(CC) mm.c -o mm-base 

mm-vec: mm.c
	$(CC) mm.c -o mm-vec $(VEC_FLAGS)

mm-omp: mm.c
	$(CC) mm.c -o mm-omp $(OMP_FLAGS)

mm-omp-vec: mm.c
	$(CC) mm.c -o mm-omp-vec -O3 $(OMP_FLAGS) $(VEC_FLAGS)

clean:
	rm -f mm-base mm-vec mm-omp mm-omp-vec

test:
	make all
	time ./mm-base
	time ./mm-vec
	time ./mm-omp
	time ./mm-omp-vec

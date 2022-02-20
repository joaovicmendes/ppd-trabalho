UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	CFLAGS=-Xpreprocessor -fopenmp 
	LFLAGS=-lomp
else
	CFLAGS=-fopenmp 
	LFLAGS=
endif

CC=gcc

DEBUG_FLAGS=-ftree-vectorize

all: mm optimized omp omp-optimized

mm: mm.c
	@echo Compiling for $(UNAME)
	$(CC) mm.c -o mm 

optimized: mm.c
	@echo Compiling for $(UNAME)
	$(CC) -O3 $(DEBUG_FLAGS) mm.c -o optimized 

omp: mm.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) mm.c -o omp $(CFLAGS) $(LFLAGS)

omp-optimized: mm.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) $(DEBUG_FLAGS) mm.c -O3 -o omp-optimized $(CFLAGS) $(LFLAGS)

clean:
	rm -f mm optimized omp omp-optimized

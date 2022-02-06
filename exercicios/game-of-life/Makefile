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

all: life gol gol-parallel

life: life.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) life.c -o life 

gol: gol.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) gol.c -o gol $(CFLAGS) $(LFLAGS)

gol-parallel: gol-parallel.c
	@echo Compiling for $(UNAME)
	$(CC) $(FLAGS) gol-parallel.c -o gol-parallel -pthread $(CFLAGS) $(LFLAGS)

clean:
	rm -f life gol gol-parallel


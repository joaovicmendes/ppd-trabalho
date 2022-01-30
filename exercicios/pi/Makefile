SAMPLE=10000000000

build:
	gcc -O2 pi.c -o pi.out
	gcc -O2 -pthreads pi-parallel.c -o pi-parallel.out

clean: 
	rm *.out

run:
	time ./pi.out 1 $(SAMPLE)
	time ./pi-parallel.out 1 $(SAMPLE)
	time ./pi-parallel.out 4 $(SAMPLE)
	time ./pi-parallel.out 8 $(SAMPLE)
	time ./pi-parallel.out 64 $(SAMPLE)
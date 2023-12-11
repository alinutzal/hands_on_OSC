CCOMP = gcc
CFLAGS = -c -std=c99

run: vector_addition.o
	$(CCOMP) vector_addition.o -o run

vector_addition.o: vector_addition.c
	$(CCOMP) $(CFLAGS) -c vector_addition.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o add_vec_cpu*

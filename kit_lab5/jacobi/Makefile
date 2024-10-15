CC = g++
CFLAGS = -g -O3 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../

jacobi: jacobi.cpp
	$(CC) $(CFLAGS) jacobi.cpp -o jacobi

clean:
	rm -f jacobi


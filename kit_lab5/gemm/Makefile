CC = g++
CFLAGS = -g -O3 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../

gemm: gemm.cpp
	$(CC) $(CFLAGS) gemm.cpp -o gemm

clean:
	rm -f gemm


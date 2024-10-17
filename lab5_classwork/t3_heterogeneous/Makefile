CC = g++
CFLAGS = -g -O3 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../
APP = t3

$(APP): $(APP).cpp
	$(CC) $(CFLAGS) $(APP).cpp -o $(APP)

clean:
	rm -f $(APP)


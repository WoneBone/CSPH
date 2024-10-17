CC = g++
CFLAGS = -g -O1 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../
APP = t1

$(APP): $(APP).cpp $(APP).hpp
	$(CC) $(CFLAGS) -std=c++11 $(APP).cpp ../common/graph.cpp -o $(APP)

clean:
	rm -f $(APP)

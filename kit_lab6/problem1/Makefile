
all: problem1.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 problem1.cpp -o frobenius

clean:
	rm -rf *~ frobenius

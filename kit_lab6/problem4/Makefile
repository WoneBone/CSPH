
all: problem4.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 problem4.cpp -o covmat

clean:
	rm -rf *~ covmat

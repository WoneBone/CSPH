
all: problem2.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 problem2.cpp -o histogram

clean:
	rm -rf *~ histogram

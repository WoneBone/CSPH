
all: problem3.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 problem3.cpp -o kmeans

clean:
	rm -rf *~ kmeans

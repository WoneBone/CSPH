
all: eval1.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 eval1.cpp -o image

clean:
	rm -rf *~ image

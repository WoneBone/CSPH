
all: eval3.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 eval3.cpp -o distance

clean:
	rm -rf *~ distance

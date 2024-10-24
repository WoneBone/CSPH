
all: eval2.cpp CycleTimer.h
	icpx -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -O3 eval2.cpp -o csort

clean:
	rm -rf *~ csort

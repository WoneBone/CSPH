#include <stdio.h>
#include <thread>


void thread_func(int thread_id, int groupnum) {
    int count = 0, init = groupnum*thread_id;
    while(1){
        if(init <= 1) break;
        else{
            if(init%2 == 0) init /= 2;
            else init = 3*init + 1;

            count += 1;
        }
    }
    printf("Thread ID: %d; Count: %d\n", thread_id, count);
}


int main(int argc, char** argv) {

    const int num_threads = 7;

    std::thread problem1_threads[num_threads];

    for (int i=1 ; i< num_threads; i++){
        problem1_threads[i]= std::thread(thread_func, i, 21);
    }
    thread_func(0,21);
    for (int i = 0; i < num_threads; i++)
    {
        problem1_threads[i].join();
    }
    
    
    return 0;
}

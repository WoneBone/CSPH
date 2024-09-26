#include <math.h>
#include <stdio.h>
#include <stdlib.h>


void evalSerial(int N, float values[],float output[])
{
    for (int i=0; i<N; i++) {

        int count = 10*values[i];
        float x1 = 1.1, x2 = 1.1, result = 0;

        for(int j = 0; j < count; j++)
            result += x1*x2;

        output[i] = result*count;
    }
}


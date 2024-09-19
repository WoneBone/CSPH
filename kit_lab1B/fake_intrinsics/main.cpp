#include "fake_intrinsics.h"

void clampedExp(float* values, int* exponents, float* output, int N){

    for (int i=0; i<N; i+=VECTOR_LENGTH){

		__vfloat x = _vload(&values[i]);
		__vint exp = _vload(&exponents[i]);
		__vfloat c = _vbcast(1.0f);
		__vint sub =  _vbcast(1);
		__vint zero = _vbcast(0);
		__vbool mask = _vgt(exp, zero);
		
		//exponent
		while(_vpopcnt(mask) > 0){
			//multiply those whose exponent is greater then 1
			c = _vmul(c, x, mask);
			exp = _vsub(exp, sub);
			mask = _vgt(exp, zero);
		}
		
		__vfloat nine = _vbcast(9.999999f);
		mask = _vgt(c, nine);
		c = _vcopy(c, nine, mask);

		_vstore(&output[i], c);

	}
}



int main() 
{
	float values[] = {3.0, 2.0, 2.5, 1.25, 5.5, 0.5, 10.1, 3.15, 1.75, 6.55, 1.63, 1.5, 4.33, 0.15, 1.95, 2.83};
	int exponents[] = {0, 2, 3, 10, 0, 4, 0, 3, 5, 1, 4, 9, 0, 5, 0, 3};
	float output[N] = {0};

	clampedExp(values, exponents, output, N);

	for(int i = 0; i < 16; i++){
		printf("%f  ", output[i]);
	}
	printf("\n");
    printStats();


    
    return 0;
}

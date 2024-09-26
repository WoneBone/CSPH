
export void eval_ispc(uniform int N,uniform float values[],uniform float output[])
{
    foreach (i = 0 ... N) {

        int count = 10*values[i];
        float x1 = 1.1, x2 = 1.1, result = 0;

        for(int j = 0; j < count; j++)
            result += x1*x2;

        output[i] = result*count;
        
    }
}

task void eval_ispc_task(uniform int N,uniform int span,uniform float values[],uniform float output[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);
    
    foreach (i = indexStart ... indexEnd) {

        int count = 10*values[i];
        float x1 = 1.1, x2 = 1.1, result = 0;

        for(int j = 0; j < count; j++)
            result += x1*x2;

        output[i] = result*count;
    
    }
}

export void eval_ispc_withtasks(uniform int N,uniform float values[],uniform float output[])
{

    uniform int span = N / 64;  // 64 tasks

    launch[N/span] eval_ispc_task(N, span, values, output);
}

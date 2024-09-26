static inline int eval(int count)
{
    float z_re = 1.1, z_im = 1.1;
    int result = 0;

    int i = 0;
    for (i = 0; i < count; ++i) {
        result += z_re*z_im;
    }

    return result;
}

export void image_ispc(uniform int width, uniform int height, uniform int data[], uniform int output[])
{
    foreach (j = 0 ... height, i = 0 ... width) {
            int index = j * width + i;
            if(data[index] == 0) output[index] = eval(1);
            else output[index] = eval(1000);  
    }
}

// slightly different kernel to support tasking
task void image_ispc_task(uniform int width, uniform int height,uniform int rowsPerTask,uniform int data[],uniform int output[])
{
    // taskIndex is an ISPC built-in
    
    uniform int ystart = taskIndex * rowsPerTask;
    uniform int yend = ystart + rowsPerTask;
    
    foreach (j = ystart ... yend, i = 0 ... width) {
            int index = j * width + i;
            if(data[index] == 0) output[index] = eval(1);
            else output[index] = eval(1000);  
    }
}

export void image_ispc_withtasks(uniform int width, uniform int height,uniform int data[],uniform int output[])
{
    uniform int rowsPerTask = height / 2;

    // create 2 tasks
    launch[2] image_ispc_task(width, height,rowsPerTask,data,output); 
}

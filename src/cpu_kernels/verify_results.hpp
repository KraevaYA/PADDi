#pragma once
#include <stdlib.h>

template <typename T>
void verify_arrays(T *cpu_array, T *gpu_array, unsigned int size)
{

    double epsilon = 1.0E-3;
    
    for (unsigned int i = 0; i < size; i++)
    {
        if (abs(cpu_array[i] - gpu_array[i]) > epsilon)
        {
            printf("Arrays do not match!\n");
            //printf("host %5.2f gpu %5.2f at current %d\n", cpu_array[i], gpu_array[i], i);
            printf("host %d, gpu %d at current %d\n", cpu_array[i], gpu_array[i], i);
            //break;
        }
    }
    
    /*for (int i = 0; i < size; i++)
    {
        printf("cpu_array[%d] = %f, gpu_array[%d] = %f\n", i, cpu_array[i], i, gpu_array[i]);
    }*/
    
    return;
}







































































// #ifndef __KERNELS_CUH__
// #define __KERNELS_CUH__

// __global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n);

// #endif

#pragma once

#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

#include "parameters.hpp"


__global__ void gpu_compute_statistics(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, unsigned int N, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    DATA_TYPE mean, std;

    while (thread_idx < N)
    {
        mean = 0;
        std = 0;

        for (int i = 0; i < m; i++)
        {
	    mean += d_T[thread_idx+i];
	    std += d_T[thread_idx+i]*d_T[thread_idx+i];
        }

	std = std/m;
        mean = mean/m;

        d_mean[thread_idx] = mean;
        d_std[thread_idx] = sqrt(std - mean*mean);

	//printf("d_std[%d] = %f, d_std[%d] = %f\n", thread_idx, d_mean[thread_idx], thread_idx, d_std[thread_idx]);
	
        thread_idx += blockDim.x*gridDim.x;

    }
}


__global__ void gpu_update_statistics(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, unsigned int N, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    volatile float mean;
    volatile float std;

    while (thread_idx < N)
    {
        mean = 0;
        std = 0;

        mean = (((float)(m-1)/m)*d_mean[thread_idx] + d_T[thread_idx+m-1]/m);
        std = sqrt(((float)(m-1)/m)*(d_std[thread_idx]*d_std[thread_idx]+(d_mean[thread_idx]-d_T[thread_idx+m-1])*(d_mean[thread_idx]-d_T[thread_idx+m-1])/m));

        d_mean[thread_idx] = mean;
        d_std[thread_idx] = std;

        thread_idx += blockDim.x*gridDim.x;
    }
}


void compute_statistics(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, unsigned int N, unsigned int m, cudaStream_t stream)
{
    unsigned int num_blocks = (int)ceil(N/(float)BLOCK_SIZE);
    dim3 blockDim = dim3(BLOCK_SIZE, 1, 1);
    dim3 gridDim = dim3(num_blocks, 1, 1);

    gpu_compute_statistics<<<gridDim, blockDim, 0, stream>>>(d_T, d_mean, d_std, N, m);

}


void update_statistics(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, unsigned int N, unsigned int m, cudaStream_t stream)
{
    unsigned int num_blocks = (int)ceil(N/(float)BLOCK_SIZE);
    dim3 blockDim = dim3(BLOCK_SIZE, 1, 1);
    dim3 gridDim = dim3(num_blocks, 1, 1);

    gpu_update_statistics<<<gridDim, blockDim, 0, stream>>>(d_T, d_mean, d_std, N, m);
}

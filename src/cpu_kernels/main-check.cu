#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <cuda_device_runtime_api.h>
#include <string.h>

#include "PD3.cuh"
#include "preprocessing.cuh"
#include "PD3_types.hpp"
#include "parameters.hpp"
#include "IOdata.hpp"
#include "timer.cuh"
#include "PD3_kernels.cuh"
#include "cpu_kernels.hpp"
#include "verify_results.hpp"
#include "common.h"


using namespace std;

int define_N_with_pad(unsigned int n, unsigned int m)
{
    unsigned int N_pad = 0;
    unsigned int pad = 0;

    unsigned int N = n - m + 1;

    if (N % SEGMENT_N != 0)
        pad = ceil(N/(float)SEGMENT_N)*SEGMENT_N + 2*m - 2 - n;
    else
        pad = m - 1;

    N_pad = N + pad;

    return N_pad;
}


int main(int argc, char *argv[])
{
    char *file_name = argv[1]; // name of input file with time series
    unsigned int n = atoi(argv[2]); // length of time series
    unsigned int m = atoi(argv[3]); // length of subsequence

    float r = 2*sqrt(m)/5.0;
    unsigned int w = m; // window
    unsigned int N = n - m + 1; // count of subsequences in time series

    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("CUDA-capable devices: %i\n", ngpus);

    unsigned int N_local = ceil(N/(float)ngpus);
    unsigned int n_local = N_local + m - 1;

    unsigned int N_local_pad = 0;
    unsigned int n_local_pad = 0;

    N_local_pad = define_N_with_pad(n_local, m);
    n_local_pad = N_local_pad + m - 1;

    printf("N_local_pad = %d\n", N_local_pad);

    // size_t size = n_local_pad * sizeof(float);

    // allocate pageable memory on host
    float **h_T = (float **)malloc(ngpus * sizeof(float *));
    int **h_cand = (int **)malloc(ngpus * sizeof(int *));
    int **h_neighbor = (int **)malloc(ngpus * sizeof(int *));
    float **h_nnDist = (float **)malloc(ngpus * sizeof(float *));

    float **d_T = (float **)malloc(ngpus * sizeof(float *));
    float **d_mean = (float **)malloc(ngpus * sizeof(float *));
    float **d_std = (float **)malloc(ngpus * sizeof(float *));
    int **d_cand = (int **)malloc(ngpus * sizeof(int *));
    int **d_neighbor = (int **)malloc(ngpus * sizeof(int *));
    float **d_nnDist = (float **)malloc(ngpus * sizeof(float *));


#ifdef _DEBUG_
    float **gpu_mean = (float **)malloc(ngpus * sizeof(float *));
    float **gpu_std = (float **)malloc(ngpus * sizeof(float *));

    float **cpu_mean = (float **)malloc(ngpus * sizeof(float *));
    float **cpu_std = (float **)malloc(ngpus * sizeof(float *));
    int **cpu_cand = (int **)malloc(ngpus * sizeof(int *));
    float **cpu_nnDist = (float **)malloc(ngpus * sizeof(float *));
#endif

    cudaStream_t *streams = (cudaStream_t *)malloc(ngpus * sizeof(cudaStream_t));

    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        // allocate pinned memory on host
        CHECK(cudaMallocHost((void **)&h_T[i], n_local_pad * sizeof(float)));
        CHECK(cudaMallocHost((void **)&h_cand[i], N_local_pad * sizeof(int)));
        CHECK(cudaMallocHost((void **)&h_neighbor[i], N_local_pad * sizeof(int)));
        CHECK(cudaMallocHost((void **)&h_nnDist[i], N_local_pad * sizeof(float)));

#ifdef _DEBUG_

	CHECK(cudaMallocHost((void **)&gpu_mean[i], N_local_pad * sizeof(float)));
        CHECK(cudaMallocHost((void **)&gpu_std[i], N_local_pad * sizeof(float)));

        CHECK(cudaMallocHost((void **)&cpu_mean[i], N_local_pad * sizeof(float)));
        CHECK(cudaMallocHost((void **)&cpu_std[i], N_local_pad * sizeof(float)));
        CHECK(cudaMallocHost((void **)&cpu_cand[i], N_local_pad * sizeof(int)));
        CHECK(cudaMallocHost((void **)&cpu_nnDist[i], N_local_pad * sizeof(float)));
#endif

        // allocate global memory on device
        CHECK(cudaMalloc((void **)&d_T[i], n_local_pad * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_mean[i], N_local_pad * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_std[i], N_local_pad * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_cand[i], N_local_pad * sizeof(int)));
        CHECK(cudaMalloc((void **)&d_neighbor[i], N_local_pad * sizeof(int)));
        CHECK(cudaMalloc((void **)&d_nnDist[i], N_local_pad * sizeof(float)));

        // create streams for timing and synchronizing
        CHECK(cudaStreamCreate(&streams[i]));
    }

    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        int len_file_name = strlen(file_name);
        file_name[len_file_name - 8] = (char)('0' + i / 1000);
        file_name[len_file_name - 7] = (char)('0' + (i / 100) % 10);
        file_name[len_file_name - 6] = (char)('0' + (i / 10) % 10);
        file_name[len_file_name - 5] = (char)('0' + i % 10);
        printf(file_name);

        read_ts(file_name, h_T[i], n_local_pad);

        for (int j = 0; j < N_local_pad; j++)
        {
            h_cand[i][j] = 1;
            h_neighbor[i][j] = 1;
            h_nnDist[i][j] = FLT_MAX;
#ifdef _DEBUG_
	    cpu_cand[i][j] = 1;
            cpu_nnDist[i][j] = FLT_MAX;
#endif
        }
    }


    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK(cudaMemcpyAsync(d_T[i], h_T[i], n_local_pad * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_cand[i], h_cand[i], N_local_pad * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_neighbor[i], h_neighbor[i], N_local_pad * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_nnDist[i], h_nnDist[i], N_local_pad * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        cuda_compute_statistics(d_T[i], d_mean[i], d_std[i], N_local_pad, m, streams[i]);

        do_PD3(d_T[i], d_mean[i], d_std[i], d_cand[i], d_neighbor[i], d_nnDist[i], N_local_pad, m, w, r*r, streams[i]);

	CHECK(cudaMemcpyAsync(h_cand[i], d_cand[i], N_local_pad * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        CHECK(cudaMemcpyAsync(h_nnDist[i], d_nnDist[i], N_local_pad * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));

    }

    // synchronize streams
    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(streams[i]));
    }


#ifdef _DEBUG_
    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

	CHECK(cudaMemcpyAsync(gpu_mean[i], d_mean[i], N_local_pad * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
	CHECK(cudaMemcpyAsync(gpu_std[i], d_std[i], N_local_pad * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));

	compute_statistics_cpu(h_T[i], cpu_mean[i], cpu_std[i], N_local_pad, m);

	//verify_arrays<float>(cpu_mean[i], gpu_mean[i], N_local_pad);
	//verify_arrays<float>(cpu_std[i], gpu_std[i], N_local_pad);

	if (i == 0)
	    find_discords_cpu(h_T[i], cpu_mean[i], cpu_std[i], cpu_cand[i], cpu_nnDist[i], N_local, r*r, m);
	else
	    find_discords_cpu(h_T[i], cpu_mean[i], cpu_std[i], cpu_cand[i], cpu_nnDist[i], N-N_local, r*r, m);

	verify_arrays<int>(cpu_cand[i], h_cand[i], N_local_pad);
	//verify_arrays(cpu_nnDist[i], h_nnDist[i], N_local_pad);
    }
#endif

    return 0;

    // make candidate matrix (union candidates from GPUs)
    //CHECK(cudaSetDevice(0));
    int Cand_matr_size = 0;
    for (int i = 0; i < ngpus; i++)
    {
        for (int j = 0; j < N_local; j++)
        {
            if (h_cand[i][j])
                Cand_matr_size++;
        }
    }

    float *h_cand_matr;
    float **d_cand_matr = (float **)malloc(ngpus * sizeof(float *));
    float *h_nnDist_cand;
    float **d_nnDist_cand = (float **)malloc(ngpus * sizeof(float *));
    float **h_nnDist_discords = (float **)malloc(ngpus * sizeof(float *));

    CHECK(cudaMallocHost((void **)&h_cand_matr, (m*Cand_matr_size) * sizeof(float)));
    CHECK(cudaMallocHost((void **)&h_nnDist_cand, Cand_matr_size * sizeof(float)));

    int num_cand = 0;
    for (unsigned int i = 0; i < ngpus; i++)
    {
        for (unsigned int j = 0; j < N_local; j++)
        {
            if (h_cand[i][j])
            {
                for (unsigned k = 0; k < m; k++)
                    h_cand_matr[Cand_matr_size*k+num_cand] = h_T[i][j+k];
                h_nnDist_cand[num_cand] = h_nnDist[i][j];
                num_cand++;
            }
        }
    }


    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
	CHECK(cudaMallocHost((void **)&h_nnDist_discords[i], Cand_matr_size * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_nnDist_cand[i], Cand_matr_size * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_cand_matr[i], (m*Cand_matr_size) * sizeof(float)));	
    }


    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK(cudaMemcpyAsync(d_cand_matr[i], h_cand_matr, (m*Cand_matr_size) * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_nnDist_cand[i], h_nnDist_cand, Cand_matr_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        dim3 blockDim = dim3(BLOCK_DIM, BLOCK_DIM, 1);
        //dim3 gridDim = dim3((int)ceil(N_local/(float)BLOCK_DIM), (int)ceil(Cand_matr_size/(float)BLOCK_DIM), 1);
	dim3 gridDim = dim3((int)ceil(Cand_matr_size/(float)BLOCK_DIM), (int)ceil(N_local/(float)BLOCK_DIM), 1);

        gpu_global_discords_refine<<<gridDim, blockDim, 0, streams[i]>>>(d_T[i], d_mean[i], d_std[i], d_cand_matr[i], d_nnDist_cand[i], N_local, Cand_matr_size, m, r*r);

        CHECK(cudaMemcpyAsync(h_nnDist_discords[i], d_nnDist_cand[i], Cand_matr_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    // synchronize streams
    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(streams[i]));
    }
 

    // Find top-1 discord
    float top1_discord_nnDist = -FLT_MAX;
    int top1_discord_ind;
    float min_nnDist;

    for (unsigned int i = 0; i < Cand_matr_size; i++)
    {
        min_nnDist = FLT_MAX;
	for (unsigned j = 0; j < ngpus; j++)
	{
	    if (min_nnDist > h_nnDist_discords[j][i])
		min_nnDist = h_nnDist_discords[j][i];
	}

	printf("i = %d, nnDist = %.2f\n", i, min_nnDist);

	if ((min_nnDist != -FLT_MAX) && (min_nnDist > top1_discord_nnDist))
        {
            top1_discord_nnDist = min_nnDist;
            top1_discord_ind = i;
        }
    }

    printf("Top-1 discord index = %d\n", top1_discord_ind);
    printf("Top-1 discord distance = %.2f\n", top1_discord_nnDist);

    // cleanup and shutdown
    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK(cudaFreeHost(h_T[i]));
        CHECK(cudaFreeHost(h_cand[i]));
        CHECK(cudaFreeHost(h_neighbor[i]));
        CHECK(cudaFreeHost(h_nnDist[i]));
	CHECK(cudaFreeHost(h_nnDist_discords[i]));

        CHECK(cudaFree(d_T[i]));
        CHECK(cudaFree(d_mean[i]));
        CHECK(cudaFree(d_std[i]));
        CHECK(cudaFree(d_cand[i]));
        CHECK(cudaFree(d_neighbor[i]));
        CHECK(cudaFree(d_nnDist[i]));
	CHECK(cudaFree(d_cand_matr[i]));
        CHECK(cudaFree(d_nnDist_cand[i]));

        CHECK(cudaStreamDestroy(streams[i]));

        CHECK(cudaDeviceReset());
    }

    free(h_T);
    free(h_cand);
    free(h_neighbor);
    free(h_nnDist);
    free(h_nnDist_discords);

    free(d_T);
    free(d_mean);
    free(d_std);
    free(d_cand);
    free(d_neighbor);
    free(d_nnDist);
    free(d_nnDist_cand);
    free(d_cand_matr);

    free(streams);

   // CHECK(cudaSetDevice(0));
   // CHECK(cudaFreeHost(h_cand_matr));
    //CHECK(cudaFreeHost(h_multi_gpus_cand));
    //CHECK(cudaFreeHost(h_nnDist_cand));

    return 0;
}


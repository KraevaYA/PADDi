#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
//#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <string.h>
#include <vector>
#include <mpi.h>

#include "data_preparation.cuh"
#include "preprocessing.cuh"
#include "cuda_streams.cuh"
#include "PALMAD_types.hpp"
#include "parameters.hpp"
#include "timer.cuh"
#include "common.h"
#include "PALMAD_one_length.cuh"


using namespace std;


int main(int argc, char *argv[])
{
    char *file_name = argv[1]; // name of input file with time series
    unsigned int n = atoi(argv[2]); // length of time series
    unsigned int minL = atoi(argv[3]); // minimum length of subsequence
    unsigned int maxL = atoi(argv[4]); // maximum length of subsequence
    char *discords_file_name = argv[5]; // name of output file for top-1 discords
    char *times_file_name = argv[6]; // name of output file for times table

    unsigned int num_lengths = maxL-minL+1;

    //DATA_TYPE r = 2*sqrt(minL)/5.0;
    DATA_TYPE r = 2*sqrt(minL);
    unsigned int w = minL; // window
    int m;

    int nproc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));

    ngpus = 1;
    printf("CUDA-capable devices: %i\n", ngpus);

    // allocate pageable memory on host
    DATA_TYPE **h_T = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **h_local_nnDist = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **h_local_discords_nnDist = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    int **h_local_is_cand = (int **)malloc(ngpus * sizeof(int *));
    int **h_local_is_neighbor = (int **)malloc(ngpus * sizeof(int *));
    DATA_TYPE *h_node_C, *h_node_cand_nnDist;
    int *h_node_cand_idx;

    // device data
    DATA_TYPE **d_T = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **d_mean = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **d_std = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **d_nnDist = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **d_C = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    DATA_TYPE **d_cand_nnDist = (DATA_TYPE **)malloc(ngpus * sizeof(DATA_TYPE *));
    int **d_is_cand = (int **)malloc(ngpus * sizeof(int *));
    int **d_is_neighbor = (int **)malloc(ngpus * sizeof(int *));
    int **d_cand_idx = (int **)malloc(ngpus * sizeof(int *));
   
    int **N_segments = (int **)malloc(ngpus * sizeof(int*));
    int **N_segments_pad = (int **)malloc(ngpus * sizeof(int*));
    for (int i = 0; i < ngpus; i++)
    {
	N_segments[i] = (int *)malloc(num_lengths * sizeof(int));
	N_segments_pad[i] = (int *)malloc(num_lengths * sizeof(int));
    }
    int *n_segments_pad_max = (int *)malloc(ngpus * sizeof(int)); 
    int *N_segments_pad_max = (int *)malloc(ngpus * sizeof(int));
    int N_segment, n_fragment;

    Discord *top1_discords = (Discord *)malloc(num_lengths * sizeof(Discord));
    for (int i = 0; i < num_lengths; i++)
        top1_discords[i] = {-1, -FLT_MAX};


    vector<vector<double>> times_tbl;
    vector<double> times_one_length;
	

    cudaStream_t *streams = (cudaStream_t *)malloc(ngpus * sizeof(cudaStream_t));

    setup_data_sizes(N_segments, N_segments_pad, n_segments_pad_max, N_segments_pad_max, n, &n_fragment, &N_segment, minL, maxL, num_lengths, nproc, ngpus, rank);

    CHECK(cudaMallocHost((void **)&h_node_C, (maxL * (n_fragment-maxL+1)) * sizeof(DATA_TYPE)));
    CHECK(cudaMallocHost((void **)&h_node_cand_nnDist, (n_fragment-minL+1) * sizeof(DATA_TYPE)));
    CHECK(cudaMallocHost((void **)&h_node_cand_idx, (n_fragment-minL+1) * sizeof(int)));

    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        // allocate pinned memory on host
        CHECK(cudaMallocHost((void **)&h_T[i], n_segments_pad_max[i] * sizeof(DATA_TYPE)));
        CHECK(cudaMallocHost((void **)&h_local_nnDist[i], N_segments_pad_max[i] * sizeof(DATA_TYPE)));
	CHECK(cudaMallocHost((void **)&h_local_discords_nnDist[i], (n_fragment-minL+1) * sizeof(DATA_TYPE)));
        CHECK(cudaMallocHost((void **)&h_local_is_cand[i], N_segments_pad_max[i] * sizeof(int)));
        CHECK(cudaMallocHost((void **)&h_local_is_neighbor[i], N_segments_pad_max[i] * sizeof(int)));

        // allocate global memory on device
        CHECK(cudaMalloc((void **)&d_T[i], n_segments_pad_max[i] * sizeof(DATA_TYPE)));
        CHECK(cudaMalloc((void **)&d_mean[i], N_segments_pad_max[i] * sizeof(DATA_TYPE)));
        CHECK(cudaMalloc((void **)&d_std[i], N_segments_pad_max[i] * sizeof(DATA_TYPE)));
        CHECK(cudaMalloc((void **)&d_nnDist[i], N_segments_pad_max[i] * sizeof(DATA_TYPE)));
	CHECK(cudaMalloc((void **)&d_C[i], (maxL * (n_fragment-maxL+1)) * sizeof(DATA_TYPE)));
	CHECK(cudaMalloc((void **)&d_cand_nnDist[i], (n_fragment-minL+1) * sizeof(DATA_TYPE)));
        CHECK(cudaMalloc((void **)&d_is_cand[i], N_segments_pad_max[i] * sizeof(int)));
        CHECK(cudaMalloc((void **)&d_is_neighbor[i], N_segments_pad_max[i] * sizeof(int)));	
	CHECK(cudaMalloc((void **)&d_cand_idx[i], (n_fragment-minL+1) * sizeof(int)));

        // create streams for timing and synchronizing
        CHECK(cudaStreamCreate(&streams[i]));
    }

    initialize_data(h_T, h_local_is_cand, h_local_is_neighbor, h_local_nnDist, n_segments_pad_max, N_segments_pad_max, n, N_segment, maxL, nproc, ngpus, rank, file_name);

    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpyAsync(d_T[i], h_T[i], n_segments_pad_max[i] * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[i]));
    }


    while (top1_discords[0].dist < 0)
    //while (D_size_non_overlap < K)
    {

        printf("%d;", minL);

	w = minL;

	top1_discords[0] = do_PALMAD_one_length(h_T, h_local_nnDist, h_local_discords_nnDist, h_local_is_cand, h_local_is_neighbor, h_node_C, h_node_cand_nnDist, h_node_cand_idx, d_T, d_mean, d_std, d_nnDist, d_C, d_cand_nnDist, d_is_cand, d_is_neighbor, d_cand_idx, N_segments, N_segments_pad, N_segment, minL, 0, r*r, w, rank, ngpus, nproc, streams, times_one_length);

	if (rank == 0)
	{
	    times_tbl.push_back(times_one_length);
	    times_one_length.clear();
	}

	printf("Top-1 discord index = %d\n", top1_discords[0].ind);
        printf("Top-1 discord distance = %.2f\n", top1_discords[0].dist);

	r = 0.5*r;	
    }
    

    //--------------------------------------------------

    for (int j = 1; j < 5; j++)
    {
	if (j == num_lengths)
	    break;

        m = minL + j;
        w = m;

	printf("%d;", m);

        r = sqrt(top1_discords[j-1].dist)*0.99;

	while (top1_discords[j].dist < 0)
	{

	    top1_discords[j] = do_PALMAD_one_length(h_T, h_local_nnDist, h_local_discords_nnDist, h_local_is_cand, h_local_is_neighbor, h_node_C, h_node_cand_nnDist, h_node_cand_idx, d_T, d_mean, d_std, d_nnDist, d_C, d_cand_nnDist, d_is_cand, d_is_neighbor, d_cand_idx, N_segments, N_segments_pad, N_segment, m, j, r*r, w, rank, ngpus, nproc, streams, times_one_length);

	    if (rank == 0)
	    {
		times_tbl.push_back(times_one_length);
		times_one_length.clear();
	    }

	    printf("Top-1 discord index = %d\n", top1_discords[j].ind);
            printf("Top-1 discord distance = %.2f\n", top1_discords[j].dist);

	    r = 0.99*r;
	}

    }


    //--------------------------------------------------

    if (num_lengths > 5)
    {
        for (int j = 5; j < num_lengths; j++)
        {
            m = minL + j;
            w = m;
	
            DATA_TYPE M = 0;
            DATA_TYPE S = 0;
        
            for (int k = j-5; k < j; k++)
                M += sqrt(top1_discords[k].dist);
        
            M = M*0.2;
        
            for (int k = j-5; k < j; k++)
                S += pow((sqrt(top1_discords[k].dist) - M), 2);
        
            S = sqrt(S*0.2);
            r = M - 2*S;

	    while (top1_discords[j].dist < 0)
	    {

                top1_discords[j] = do_PALMAD_one_length(h_T, h_local_nnDist, h_local_discords_nnDist, h_local_is_cand, h_local_is_neighbor, h_node_C, h_node_cand_nnDist, h_node_cand_idx, d_T, d_mean, d_std, d_nnDist, d_C, d_cand_nnDist, d_is_cand, d_is_neighbor, d_cand_idx, N_segments, N_segments_pad, N_segment, m, j, r*r, w, rank, ngpus, nproc, streams, times_one_length);

		if (rank == 0)
		{
		    times_tbl.push_back(times_one_length);
		    times_one_length.clear();
		}

	        printf("Top-1 discord index = %d\n", top1_discords[j].ind);
                printf("Top-1 discord distance = %.2f\n", top1_discords[j].dist);

	        r = r - S;
	    }
         }
    }
	

    if (rank == 0)
    {
	write_discords(discords_file_name, top1_discords, num_lengths, minL);
	write_times_tbl(times_file_name, times_tbl);
    }


    // cleanup and shutdown
    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK(cudaFreeHost(h_T[i]));
        CHECK(cudaFreeHost(h_local_is_cand[i]));
        CHECK(cudaFreeHost(h_local_is_neighbor[i]));
        CHECK(cudaFreeHost(h_local_nnDist[i]));
	CHECK(cudaFreeHost(h_local_discords_nnDist[i]));

        CHECK(cudaFree(d_T[i]));
        CHECK(cudaFree(d_mean[i]));
        CHECK(cudaFree(d_std[i]));
        CHECK(cudaFree(d_is_cand[i]));
        CHECK(cudaFree(d_is_neighbor[i]));
        CHECK(cudaFree(d_nnDist[i]));
	CHECK(cudaFree(d_C[i]));
        CHECK(cudaFree(d_cand_nnDist[i]));
	CHECK(cudaFree(d_cand_idx[i]));

        CHECK(cudaStreamDestroy(streams[i]));

        CHECK(cudaDeviceReset());
    }

    free(h_T);
    free(h_local_is_cand);
    free(h_local_is_neighbor);
    free(h_local_nnDist);
    free(h_local_discords_nnDist);

    free(d_T);
    free(d_mean);
    free(d_std);
    free(d_is_cand);
    free(d_is_neighbor);
    free(d_nnDist);
    free(d_C);
    free(d_cand_nnDist);
    free(d_cand_idx);

    free(streams);

    MPI_Finalize();

    return 0;
}


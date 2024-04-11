#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "PD3_kernels.cuh"
#include "parameters.hpp"
#include "timer.cuh"


using namespace std;


void do_PD3(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, int *d_is_cand, int *d_is_neighbor, DATA_TYPE *d_nnDist, unsigned int N, unsigned int m, unsigned int w, DATA_TYPE r, cudaStream_t stream, int rank, int N_real)
{

        // Phase 1. Candidate Selection Algorithm
        unsigned int num_blocks = (N-m+1)/SEGMENT_N;
        dim3 blockDim = dim3(SEGMENT_N, 1, 1);
        dim3 gridDim = dim3(num_blocks, 1, 1);

        gpu_local_candidate_select<<<gridDim, blockDim, (2*SEGMENT_N+2*m-2)*sizeof(DATA_TYPE), stream>>>(d_T, d_mean, d_std, d_is_cand, d_is_neighbor, d_nnDist, N, N_real, r, m, rank);

        num_blocks = ceil(N/(float)DEFINE_CAND_BLOCK_SIZE);
        blockDim = dim3(DEFINE_CAND_BLOCK_SIZE, 1, 1);
        gridDim = dim3(num_blocks, 1, 1);

        gpu_local_define_candidates<<<gridDim, blockDim, 0, stream>>>(d_is_cand, d_is_neighbor, N, rank);

        // Phase 2. Discord Refinement Algorithm
        num_blocks = (N-m+1)/SEGMENT_N;
        blockDim = dim3(SEGMENT_N, 1, 1);
        gridDim = dim3(num_blocks, 1, 1);
        printf("blockDim = %d, gridDim = %d\n", SEGMENT_N, num_blocks);

        gpu_local_discords_refine<<<gridDim, blockDim, (2*SEGMENT_N+2*m-2)*sizeof(DATA_TYPE), stream>>>(d_T, d_mean, d_std, d_is_cand, d_nnDist, N, m, r);

}


void node_discords_refine(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, DATA_TYPE *d_C, int *d_cand_idx, DATA_TYPE *d_cand_nnDist, int N_segment_i, int common_N_segment, int node_cand_num, int m, DATA_TYPE r, int gpu_i, cudaStream_t stream)
{

    dim3 blockDim = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 gridDim = dim3((int)ceil(node_cand_num/(float)BLOCK_DIM), (int)ceil(N_segment_i/(float)BLOCK_DIM), 1);

    gpu_global_discords_refine<<<gridDim, blockDim, 0, stream>>>(d_T, d_mean, d_std, d_C, d_cand_idx, d_cand_nnDist, N_segment_i, common_N_segment, node_cand_num, m, r, gpu_i);

}


void global_discords_refine(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, DATA_TYPE *d_global_C, int *d_global_cand_idx, DATA_TYPE *d_global_cand_nnDist, int N_segment_i, int common_N_segment, int global_cand_num, int m, DATA_TYPE r, int rank, int num_segm, cudaStream_t stream) 
{
    
    dim3 blockDim = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 gridDim = dim3((int)ceil(global_cand_num/(float)BLOCK_DIM), (int)ceil(N_segment_i/(float)BLOCK_DIM), 1);

    gpu_multigpus_discords_refine<<<gridDim, blockDim, 0, stream>>>(d_T, d_mean, d_std, d_global_C, d_global_cand_idx, d_global_cand_nnDist, N_segment_i, common_N_segment, global_cand_num, m, r, rank, num_segm);

}


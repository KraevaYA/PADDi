 #pragma once

#include <math.h>
#include <stdio.h>
#include <float.h>

#include "parameters.hpp"

__global__ void gpu_local_candidate_select(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, int *d_cand, int *d_neighbor, DATA_TYPE *d_nnDist, unsigned int N, int N_real, DATA_TYPE r, unsigned int m, int rank)
{
    unsigned int tid = threadIdx.x; 
    unsigned int blockSize = blockDim.x;
    unsigned int segment_ind = blockIdx.x*blockSize;
    unsigned int chunk_ind = segment_ind + m - 1;
    DATA_TYPE nnDist = FLT_MAX;
    DATA_TYPE min_nnDist = FLT_MAX;
    bool non_overlap;
	

    extern __shared__ DATA_TYPE dynamicMem[];
    DATA_TYPE *segment = dynamicMem;
    DATA_TYPE *chunk = (DATA_TYPE*)&segment[SEGMENT_N+m-1];

    __shared__ int cand[SEGMENT_N];
    __shared__ DATA_TYPE dot_col[SEGMENT_N];
    __shared__ DATA_TYPE dot_row[SEGMENT_N];
    __shared__ DATA_TYPE dot_inter[SEGMENT_N];
    __shared__ int all_rej[1];

    cand[tid] = 1;

    if (tid == 0)
	all_rej[0] = 1;

    int ind = tid;
    int segment_len = SEGMENT_N+m-1;

    while (ind < segment_len)
    {
	segment[ind] = d_T[segment_ind+ind];
	chunk[ind] = d_T[chunk_ind+ind];
	ind += blockSize;
    }

    dot_col[tid] = 0;
    dot_row[tid] = 0;

    __syncthreads();

    // calculate dot for the first column and row (the first chunk)
    for (int j = 0; j < m; j++)
    {
        dot_col[tid] += segment[j]*chunk[j+tid];
        dot_row[tid] += segment[j+tid]*chunk[j];
    }

    __syncthreads();

    if ((segment_ind+tid < N_real) && (chunk_ind < N_real))
    {
    	nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

    	if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
	    nnDist = 2.0*m;
    }
    else
	if ((segment_ind+tid >= N_real) && (chunk_ind >= N_real))
	    nnDist = -FLT_MAX;

    non_overlap = (abs((int)(segment_ind+tid-chunk_ind)) < (m-1)) ? 0 : 1;

    if (non_overlap) 
    {
        if (nnDist < r)
        {
            cand[tid] = 0;
            atomicMin(d_neighbor+chunk_ind, 0);
        }
        else
	    min_nnDist = min(min_nnDist, nnDist);
    }

    // calculate dot for rows from second to last (the first chunk)
    for (int j = 0; j < blockSize-1; j++)
    {

	if (tid > 0)
	    dot_inter[tid] = dot_row[tid-1];

        __syncthreads();

        if (tid > 0)
            dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

        __syncthreads();

        if (tid == 0)
            dot_row[tid] = dot_col[j+1];

	if ((segment_ind+tid < N_real) && (chunk_ind+j+1 < N_real))
        {
            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

            if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
	        nnDist = 2.0*m;

	    if (std::isnan(nnDist))
	        nnDist = -FLT_MAX;
	}
	else
	    if ((segment_ind+tid >= N_real) && (chunk_ind+j+1 >= N_real))
	        nnDist = -FLT_MAX;

	non_overlap = (abs((int)(segment_ind+tid-chunk_ind-j-1)) < (m-1)) ? 0 : 1;

        if (non_overlap) 
	{
            if (nnDist < r)
            {
                cand[tid] = 0;
                atomicMin(d_neighbor+chunk_ind+j+1, 0);
            }
            else
		min_nnDist = min(min_nnDist, nnDist);
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        all_rej[0] = 0;
        for (int k = 0; k < blockSize; k++)
            if (cand[k] == 1)
	    {
	        all_rej[0] = cand[k];
		break;
	    }
    }

    __syncthreads();

    chunk_ind += blockSize;

    // process chunks from the second to last
    while ((chunk_ind < N) && (all_rej[0] != 0))
    {
        dot_col[tid] = 0;
	ind = tid;

	while (ind < segment_len)
	{
	    chunk[ind] = d_T[chunk_ind+ind];
	    ind += blockSize;
        }
 	
	__syncthreads();
	
        for (int j = 0; j < m; j++)
	{
            dot_col[tid] += segment[j]*chunk[j+tid];
        }

	if (tid > 0)
            dot_inter[tid] = dot_row[tid-1];

	__syncthreads();

	if (tid > 0)
            dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m-1] - segment[tid-1]*d_T[chunk_ind-1];
	else
            dot_row[tid] = dot_col[0];

	__syncthreads();


	if ((segment_ind+tid < N_real) && (chunk_ind < N_real))
        {
            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

            if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                nnDist = 2.0*m;
	}
	else
	    if ((segment_ind+tid >= N_real) && (chunk_ind >= N_real))
	        nnDist = -FLT_MAX;
	

        if (cand[tid] != 0) { //!!!
            if (nnDist < r)
            {
                cand[tid] = 0;
                atomicMin(d_neighbor+chunk_ind, 0);
            }
            else
               min_nnDist = min(min_nnDist, nnDist);
        }

        for (int j = 0; j < blockSize-1; j++)
        {
	    if (tid > 0)
                dot_inter[tid] = dot_row[tid-1];

            __syncthreads();

            if (tid > 0)
		dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];
            if (tid == 0)
                dot_row[tid] = dot_col[j+1];

	    if ((segment_ind+tid < N_real) && (chunk_ind+j+1 < N_real))
            {
                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

                if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
	            nnDist = 2.0*m;
	    }
	    else
	 	if ((segment_ind+tid >= N_real) && (chunk_ind+j+1 >= N_real))
	            nnDist = -FLT_MAX;

            if (cand[tid] != 0)
            {
                if (nnDist < r)
                {
                    cand[tid] = 0;
                    atomicMin(d_neighbor+chunk_ind+j+1, 0);
                }
                else
                    min_nnDist = min(min_nnDist, nnDist);
            }
        }

        __syncthreads();

        if (tid == 0)
        {
            all_rej[0] = 0;
            for (int k = 0; k < blockSize; k++)
                if (cand[k] == 1)
                {
                    all_rej[0] = cand[k];
                    break;
                }
        }

        __syncthreads();

        chunk_ind += blockSize;

    }

    if (segment_ind+tid < N)
    {
        d_cand[segment_ind+tid] = cand[tid];
	d_nnDist[segment_ind+tid] = min_nnDist;
	//printf("rank = %d: idx = %d, min_nnDist = %f\n", rank, segment_ind+tid, min_nnDist);
    }
}

__global__ void gpu_local_define_candidates(int *d_is_cand, int *d_is_neighbor, unsigned int N, int rank)
{
    unsigned int tid = threadIdx.x; 
    unsigned int thread_id = blockIdx.x*blockDim.x+tid;

    if (thread_id < N) 
    {
	//printf("rank = %d: d_is_cand[%d] = %d, d_is_neighbor[%d] = %d\n", rank, thread_id, d_is_cand[thread_id], thread_id, d_is_neighbor[thread_id]);

	d_is_cand[thread_id] = d_is_cand[thread_id] * d_is_neighbor[thread_id];
        //thread_id += blockDim.x*gridDim.x;
    }

}

__global__ void gpu_local_discords_refine(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, int *d_cand, DATA_TYPE *d_nnDist, unsigned int N, unsigned int m, DATA_TYPE r)
{
    unsigned int tid = threadIdx.x; 
    int blockSize = blockDim.x;
    int segment_ind = blockIdx.x*blockSize;
    int chunk_ind = 0;
    DATA_TYPE nnDist = FLT_MAX;
    DATA_TYPE min_nnDist = d_nnDist[segment_ind+tid];
    bool non_overlap;
    int ind = tid;
    int step = 0;
    int segment_len = SEGMENT_N+m-1;

    extern __shared__ DATA_TYPE dynamicMem[];
    DATA_TYPE *segment = dynamicMem;
    DATA_TYPE *chunk = (DATA_TYPE*)&segment[SEGMENT_N+m-1];

    __shared__ int cand[SEGMENT_N];
    __shared__ DATA_TYPE dot_col[SEGMENT_N];
    __shared__ DATA_TYPE dot_row[SEGMENT_N];
    __shared__ DATA_TYPE dot_inter[SEGMENT_N];
    __shared__ int all_rej[1];

    cand[tid] = d_cand[segment_ind+tid];

    __syncthreads();

    if (tid == 0)
    {
        all_rej[0] = 0;
        for (int k = 0; k < blockSize; k++)
            if (cand[k] == 1)
            {
                all_rej[0] = cand[k];
                break;
            }
    }

    __syncthreads();

    if (all_rej[0] != 0) 
    {
        while (ind < segment_len)
        {
	    segment[ind] = d_T[segment_ind+ind];
	    ind += blockSize;
        }

	while ((chunk_ind < segment_ind-blockSize) && (all_rej[0] != 0))
        {
            dot_col[tid] = 0;
	    
            ind = tid;
	    
            while (ind < segment_len)
            {
                chunk[ind] = d_T[chunk_ind+ind];
	        ind += blockSize;
            }

	    __syncthreads();

	    if (step == 0) 
	    {
	        dot_row[tid] = 0;

                for (int j = 0; j < m; j++) {
                    dot_col[tid] += segment[j]*chunk[j+tid];
                    dot_row[tid] += segment[j+tid]*chunk[j];
                }
	    }
            else
            {
                for (int j = 0; j < m; j++)
                    dot_col[tid] += segment[j]*chunk[j+tid];

                if (tid > 0)
                     dot_inter[tid] = dot_row[tid-1];

		__syncthreads();
	
		if (tid > 0)
            	    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m-1] - segment[tid-1]*d_T[chunk_ind-1];
		else
            	    dot_row[tid] = dot_col[0];
	    }

            __syncthreads();

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

            if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                nnDist = 2.0*m;

            if (cand[tid] != 0)
            {
                if (nnDist < r)
                {
                    cand[tid] = 0;
                    min_nnDist = -FLT_MAX;
                }
                else
                    min_nnDist = min(min_nnDist, nnDist);
            }

            for (int j = 0; j < blockSize-1; j++)
            {
                if (tid > 0)
                    dot_inter[tid] = dot_row[tid-1];

                __syncthreads();

                if (tid > 0)
                    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

                __syncthreads();

                if (tid == 0)
                    dot_row[tid] = dot_col[j+1];

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

                if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                    nnDist = 2.0*m;

                if (cand[tid] != 0) {
                    if (nnDist < r)
                    {
                        cand[tid] = 0;
                        min_nnDist = -FLT_MAX;
                    }
                    else
                        min_nnDist = min(min_nnDist, nnDist);
                }
            }

            __syncthreads();

            if (tid == 0)
            {
                all_rej[0] = 0;
                for (int k = 0; k < blockSize; k++)
                    if (cand[k] == 1)
                    {
                        all_rej[0] = cand[k];
                        break;
                    }
            }

            __syncthreads();

            chunk_ind += blockSize;
            step++;
        }

        while ((chunk_ind < segment_ind) && (all_rej[0] != 0))
        {
            dot_row[tid] = 0;
            dot_col[tid] = 0;
            ind = tid;

            while (ind < segment_len)
            {
                chunk[ind] = d_T[chunk_ind+ind];
                ind += blockSize;
            }

            __syncthreads();

            for (int j = 0; j < m; j++)
            {
                dot_col[tid] += segment[j]*chunk[j+tid];
                dot_row[tid] += segment[j+tid]*chunk[j];
            }

            __syncthreads();

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

            if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                nnDist = 2.0*m;

            non_overlap = (abs((int)(segment_ind+tid-chunk_ind)) < (m-1)) ? 0 : 1;

            if ((non_overlap) && (cand[tid] != 0))
            {
                if (nnDist < r)
                {
                    cand[tid] = 0;
                    min_nnDist = -FLT_MAX;
                }
                else
                    min_nnDist = min(min_nnDist, nnDist);
            }

            for (int j = 0; j < blockSize-1; j++)
            {
                if (tid > 0)
                    dot_inter[tid] = dot_row[tid-1];

                __syncthreads();

                if (tid > 0)
                    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

                __syncthreads();

                if (tid == 0)
                    dot_row[tid] = dot_col[j+1];

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

                if (std::isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                    nnDist = 2.0*m;

                non_overlap = (abs((int)(segment_ind+tid-chunk_ind-j-1)) < (m-1)) ? 0 : 1;

                if ((non_overlap) && (cand[tid] != 0)) {
                    if (nnDist < r)
                    {
                        cand[tid] = 0;
                        min_nnDist = -FLT_MAX;
                    }
                    else
                        min_nnDist = min(min_nnDist, nnDist);
                }
            }

            __syncthreads();

            if (tid == 0)
            {
                all_rej[0] = 0;
                for (int k = 0; k < blockSize; k++)
                    if (cand[k] == 1)
                    {
                        all_rej[0] = cand[k];
                        break;
                    }
            }

            __syncthreads();

            chunk_ind += blockSize;
        }

        if (segment_ind+tid < N)
        {
            d_cand[segment_ind+tid] *= cand[tid];
            d_nnDist[segment_ind+tid] = min_nnDist;
        }
    }
} // end void


template <typename T>
__device__ void AtomicMin(T * const address, const T value)
{
    AtomicMin(address, value);
}


template <>
__device__ void AtomicMin(float* const address, const float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    //return __int_as_float(old);
}


template <>
__device__ void AtomicMin(double* const address, const double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    //return __longlong_as_double(old);
}


__device__ float floatAtomicMin(float* const address, const float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}


__global__ void gpu_global_discords_refine(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, DATA_TYPE *d_Cand_matr, int *d_Cand_ind, DATA_TYPE *d_global_nnDist, int N, int N_segment, int Cand_matr_size, unsigned int m, DATA_TYPE r, int gpu)
{

// 2*(BLOCK_DIM*BLOCK_DIM) + BLOCK_DIM + 4*BLOCK_DIM = shared memory (48 KB)
// 2*BLOCK_DIM^2 + 5*BLOCK_DIM <= 12288 (# floats)
// 2*BLOCK_DIM^2 + 5*BLOCK_DIM - 12288 <= 0
// 2*BLOCK_DIM^2 + 5*BLOCK_DIM - 12288 = 0
// D = b^2 - 4ac = 5^2 + 4*2*12288 = 98329
// x_1 = (-b + sqrt(D)) / 2a = (-5 + 313.57) / 4 = (-5 + 313) / 4 = 77
// x_2 = (-b - sqrt(D)) / 2a = (-5 - 313.57) / 4 = (-5 - 313) / 4 = -79

// BLOCK_DIM \in [0; 77]

    __shared__ DATA_TYPE shared_Cand[BLOCK_DIM][BLOCK_DIM]; // used to store candidates from global memory
    __shared__ DATA_TYPE shared_Neighbor[BLOCK_DIM][BLOCK_DIM]; // used to store subsequences from global memory
    __shared__ DATA_TYPE shared_nnDist[BLOCK_DIM]; // used to store local minimum of distance between candidates and subsequences

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int col = BLOCK_DIM*blockIdx.x + tx; // global column in result matrix
    unsigned int row = BLOCK_DIM*blockIdx.y + ty; // global row in result matrix

    DATA_TYPE dot = 0;
    DATA_TYPE nnDist = 0;
    DATA_TYPE mean = 0, std = 0;

    if (tx+BLOCK_DIM*ty < BLOCK_DIM)
        shared_nnDist[tx] = FLT_MAX;

    for (unsigned int i = 0; i < m; i += BLOCK_DIM)
    {

	shared_Neighbor[ty][tx] = 0;
	shared_Cand[ty][tx] = 0;

	if ((row < N) && (i+tx < m)) 
	    shared_Neighbor[ty][tx] = d_T[row+tx+i];

	if ((col < Cand_matr_size) && (i+ty < m))
	    shared_Cand[ty][tx] = d_Cand_matr[col + (i+ty) * Cand_matr_size];

	__syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; k++)
        {
            dot += shared_Neighbor[ty][k]*shared_Cand[k][tx];
            mean += shared_Cand[k][tx];
            std += shared_Cand[k][tx]*shared_Cand[k][tx];
        }

        __syncthreads();

    }

    //printf("dot = %.2f\n", dot);

    mean = mean/m;
    std = std/m;
    std = sqrt(std - mean*mean);

    nnDist = 2*m*(1-(dot-m*d_mean[row]*mean)/(m*d_std[row]*std));

    // nnDist = (abs((int)(row - ind_cand)) < m) || (dist < r) ? FLT_MAX : dist;
    // nnDist = (nnDist < r) ? -FLT_MAX : nnDist;
    nnDist = (abs((int)((row+gpu*N_segment) - d_Cand_ind[col])) < (m-1)) || (nnDist < r) ? FLT_MAX : nnDist;

    //if ((col < Cand_matr_size) && (row < N))
    if (col < Cand_matr_size)
    {
        AtomicMin<DATA_TYPE>(shared_nnDist+tx, nnDist);
    }

    __syncthreads();


    if (tx+BLOCK_DIM*ty < BLOCK_DIM)
        AtomicMin<DATA_TYPE>(d_global_nnDist +blockIdx.x*BLOCK_DIM+tx, shared_nnDist[tx]);

}


__global__ void gpu_multigpus_discords_refine(DATA_TYPE *d_T, DATA_TYPE *d_mean, DATA_TYPE *d_std, DATA_TYPE *d_Cand_matr, int *d_Cand_ind, DATA_TYPE *d_global_nnDist, int N, int N_segment, int Cand_matr_size, unsigned int m, DATA_TYPE r, int rank, int num_segm)
{

    __shared__ DATA_TYPE shared_Cand[BLOCK_DIM][BLOCK_DIM+1]; // used to store candidates from global memory
    __shared__ DATA_TYPE shared_Neighbor[BLOCK_DIM][BLOCK_DIM]; // used to store subsequences from global memory
    __shared__ DATA_TYPE shared_nnDist[BLOCK_DIM]; // used to store local minimum of distance between candidates and subsequences

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x; 
    int by = blockIdx.y;

    int aBegin = BLOCK_DIM*by;
    int aEnd = aBegin+m-1;

    int atBegin = m*BLOCK_DIM*bx; 

    DATA_TYPE dot = 0;
    DATA_TYPE nnDist = 0;
    DATA_TYPE mean = 0, std = 0;

    if (tx+BLOCK_DIM*ty < BLOCK_DIM)
        shared_nnDist[tx] = FLT_MAX;

    int tile_i = 0;

    for (int ia = aBegin, iat = atBegin; ia <= aEnd; ia += BLOCK_DIM, iat += BLOCK_DIM)
    {

	shared_Neighbor[ty][tx] = 0;
	shared_Cand[ty][tx] = 0;

	if ((tile_i*BLOCK_DIM+tx < m) && (BLOCK_DIM*by+ty < N)) 
	    shared_Neighbor[ty][tx] = d_T[ia+ty+tx];

	if ((tile_i*BLOCK_DIM+tx < m) && (BLOCK_DIM*bx+ty < Cand_matr_size))
	    shared_Cand[ty][tx] = d_Cand_matr[iat+m*ty+tx];

	__syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; k++)
        {
            dot += shared_Neighbor[ty][k]*shared_Cand[tx][k];
            mean += shared_Cand[tx][k];
            std += shared_Cand[tx][k]*shared_Cand[tx][k];
        }

        tile_i++;

        __syncthreads();

    }

    //printf("dot = %.2f\n", dot);

    mean = mean/m;
    std = std/m;
    std = sqrt(std - mean*mean);


    nnDist = 2*m*(1-(dot-m*d_mean[BLOCK_DIM*by+ty]*mean)/(m*d_std[BLOCK_DIM*by+ty]*std));

	
    /*if ((bx==0) && (by==0))
        printf("shared_Neighbor[%d][%d] = %f, shared_Cand[%d][%d] = %f\n", ty, tx, shared_Neighbor[ty][tx], ty, tx, shared_Cand[ty][tx]);*/



    // nnDist = (abs((int)(row - ind_cand)) < m) || (dist < r) ? FLT_MAX : dist;
    // nnDist = (nnDist < r) ? -FLT_MAX : nnDist;
    //nnDist = (abs((int)(((BLOCK_DIM*by+ty)+num_segm*N_segment) - d_Cand_ind[BLOCK_DIM*bx+tx])) < (m-1)) || (nnDist < r) ? FLT_MAX : nnDist;
    nnDist = (abs((int)(((BLOCK_DIM*by+ty)+num_segm*N_segment) - d_Cand_ind[BLOCK_DIM*bx+tx])) < (m-1)) ? FLT_MAX : nnDist;


   //printf("rank = %d: idx = %d, d_Cand_ind = %d, nnDist = %f\n", rank, (BLOCK_DIM*by+ty)+num_segm*N_segment, d_Cand_ind[BLOCK_DIM*bx+tx], nnDist);



    if (((BLOCK_DIM*bx+tx) < Cand_matr_size) && ((BLOCK_DIM*by+ty) < N))
    {
        AtomicMin<DATA_TYPE>(shared_nnDist+tx, nnDist);
	//floatAtomicMin(shared_nnDist+tx, nnDist);
    }

    __syncthreads();


    if (tx+BLOCK_DIM*ty < BLOCK_DIM) {
	//floatAtomicMin(d_global_nnDist + bx*BLOCK_DIM+tx, shared_nnDist[tx]);
        AtomicMin<DATA_TYPE>(d_global_nnDist + bx*BLOCK_DIM+tx, shared_nnDist[tx]);
    }

}




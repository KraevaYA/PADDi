#pragma once

#include <math.h>
#include <stdio.h>
#include <float.h>

#include "IOdata.hpp"
#include "common.h"
#include "parameters.hpp"


int define_N_segment_with_pad(unsigned int n, unsigned int m)
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


void setup_data_sizes(int **N_segments, int **N_segments_pad, int *n_segments_pad_max, int *N_segments_pad_max, int n, int *n_fragment, int *N_segment, int minL, int maxL, int num_lengths, int nproc, int ngpus, int rank)
{
    int N, N_segment_i = 0, n_segment = 0, m;

    N = n - maxL + 1;
    *N_segment = ceil(N/(float)(ngpus*nproc));

    for (int i = 0; i < ngpus; i++)
    {
	if ((i == (ngpus-1)) && (rank == (nproc-1)))
	    N_segment_i = N - (rank*ngpus+i)*(*N_segment) + (maxL-minL);
	else
	    N_segment_i = *N_segment;

	if ((i == (ngpus-1)) && (rank == (nproc-1)))
	    *n_fragment = (*N_segment)*(ngpus-1) + N_segment_i + minL - 1;
	else
	{
	    if ((i == (ngpus-1)) && (rank < (nproc-1)))
	        *n_fragment = N_segment_i*ngpus + maxL - 1;
	}


	N_segments_pad_max[i] = 0;

        for (int j = 0; j < num_lengths; j++)
        {
	    m = minL + j;

            n_segment = N_segment_i + m - 1;
    	    N_segments_pad[i][j] = define_N_segment_with_pad(n_segment, m);

	    if (N_segments_pad[i][j] >= N_segments_pad_max[i]) 
	    {
	    	N_segments_pad_max[i] = N_segments_pad[i][j];  
	    	n_segments_pad_max[i] = N_segments_pad[i][j] + m - 1;   
	    }

	    N_segments[i][j] = N_segment_i;


	    if ((i == (ngpus-1)) && (rank == (nproc-1)))
		N_segment_i = N_segment_i - 1;
	}
    }
}


void initialize_data(DATA_TYPE **h_T, int **h_local_is_cand, int **h_local_is_neighbor, DATA_TYPE **h_local_nnDist, int *n_segments_pad_max, int *N_segments_pad_max, int n, int N_segment, int maxL, int nproc, int ngpus, int rank, char *file_name)
{
    int start_segment;
    int end_segment; 

    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        //read_ts(file_name, h_T[i], n_segment_pad_max[i]);

	start_segment = (rank*ngpus+i)*N_segment;
	if ((i == (ngpus-1)) && (rank == (nproc-1)))
	    end_segment = n;
	else
	    end_segment = (rank*ngpus+i+1)*N_segment + maxL - 1;

	read_ts_segment(file_name, h_T[i], n_segments_pad_max[i], start_segment, end_segment);

        for (int j = 0; j < N_segments_pad_max[i]; j++)
        {
            h_local_is_cand[i][j] = 1;
            h_local_is_neighbor[i][j] = 1;
            h_local_nnDist[i][j] = FLT_MAX;
        }
    }
}


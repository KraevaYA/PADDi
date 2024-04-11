#pragma once

// constants for preprocessing
#define BLOCK_SIZE 1024
#define BLOCK_DIM 32

// constants for the 1st phase Candidate Selection and the 2nd phase Discord Refinement
#ifndef SEGMENT_N
#define SEGMENT_N 512
#endif

#define IS_FLOAT 1

#if IS_FLOAT
    #define DATA_TYPE float
    #define MPI_DATA_TYPE MPI_FLOAT
#else
    #define DATA_TYPE double
    #define MPI_DATA_TYPE MPI_DOUBLE
#endif


// constants for prune off the subsequences
#define DEFINE_CAND_BLOCK_SIZE 1024

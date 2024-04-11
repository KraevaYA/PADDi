#pragma once

#include <math.h>
#include <stdio.h>
#include <float.h>


void synchronize_streams(cudaStream_t *streams, int ngpus)
{
    // synchronize streams
    for (unsigned int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(streams[i]));
    }
}
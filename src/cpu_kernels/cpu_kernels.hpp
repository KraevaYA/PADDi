#pragma once

#include "PD3_types.hpp"

void compute_statistics_cpu(float *T, float *means, float *stds, unsigned int N, unsigned int m);
void find_discords_cpu(float *T, float *means, float *stds, int *cand, float *nnDist, unsigned int N, float r, unsigned int m);

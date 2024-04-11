#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "cpu_kernels.hpp"

void compute_statistics_cpu(float *T, float *means, float *stds, unsigned int N, unsigned int m)
{
    float mean = 0, std = 0;
    
    for (int i = 0; i < N; i++)
    {
        mean = 0;
        std = 0;
        
        for (int j = 0; j < m; j++)
        {
            mean += T[i+j];
            std += T[i+j]*T[i+j];
        }
        
        mean = mean/m;
        std = std/m;
        std = (float)sqrt(std - mean*mean);
        
        means[i] = mean;
        stds[i] = std;
    }
    
    return;
}

void find_discords_cpu(float *T, float *means, float *stds, int *cand, float *nnDist, unsigned int N, float r, unsigned int m)
{
    
    float dot = 0;
    float ED_dist = 0;
    bool not_overlap = 0;
    bool not_pruned = 1;
    int cand_ind = 0, subs_ind = 0;
    
    for (int i = 0; i < N; i++)
    {
        cand_ind = i;
        subs_ind = 0;
        not_pruned = 1;
        
        while ((not_pruned) && (subs_ind < N))
        {
            dot = 0;
            
            for (int j = 0; j < m; j++)
                dot += T[cand_ind+j]*T[subs_ind+j];

            ED_dist = 2*m*(1-(dot-m*means[cand_ind]*means[subs_ind])/(m*stds[cand_ind]*stds[subs_ind]));
        
            not_overlap = (abs((int)(cand_ind - 1 - subs_ind)) < m) ? 0 : 1;
        
            if (not_overlap)
            {
                if (ED_dist > r)
                {
                    if (ED_dist < nnDist[cand_ind])
                        nnDist[cand_ind] = ED_dist;
                    //printf("nnDist[%d] = %.2f\n", cand_ind, nnDist[cand_ind]);
                }
                else
                {
                    //printf("ED_dist[%d] = %.2f\n", cand_ind, ED_dist);
                    cand[cand_ind] = 0;
                    not_pruned = 0;
                }
            }
            subs_ind++;
        }
    }
    return;
}

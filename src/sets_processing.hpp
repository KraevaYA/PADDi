#pragma once

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <iostream>
#include <fstream>

using namespace std;

Discord find_top1_discord(DATA_TYPE* h_discords_nnDist, int* h_cand_ind, unsigned int global_cand_num, int *discords_num, DATA_TYPE r, unsigned int nproc)
{
    Discord top1_discord = {-1, -FLT_MAX};
    DATA_TYPE top1_discord_nnDist = -FLT_MAX;
    
    DATA_TYPE min_nnDist;
    
    for (unsigned int i = 0; i < global_cand_num; i++)
    {
        min_nnDist = FLT_MAX;
        for (unsigned int j = 0; j < nproc; j++)
        {
            if (min_nnDist > h_discords_nnDist[j*global_cand_num+i])
                min_nnDist = h_discords_nnDist[j*global_cand_num+i];
        }
        //printf("min_nnDist = %f\n", min_nnDist);
        
        if ((min_nnDist > top1_discord_nnDist) && (min_nnDist > r))
        {
            top1_discord.dist = min_nnDist;
            top1_discord.ind = h_cand_ind[i];
            top1_discord_nnDist = min_nnDist;
            //(*discords_num)++;
            //printf("curr_top1_dist = %f\n", top1_discord_nnDist);
        }
        
        if (min_nnDist > r)
        {
            (*discords_num)++;
            //printf("curr_top1_dist = %f\n", top1_discord_nnDist);
        }
        
    }
    
    return top1_discord;
}


int get_node_candidates_num(int** h_is_cand, int **N_segments, int m_i, unsigned int ngpus)
{
    
    int cand_num = 0;
    for (int i = 0; i < ngpus; i++)
    {
        for (int j = 0; j < N_segments[i][m_i]; j++)
        {
            if (h_is_cand[i][j])
                cand_num++;
        }
    }
    
    return cand_num;
}


int get_node_discords_num(DATA_TYPE** h_discords_nnDist, int node_cand_num, float r, unsigned int ngpus)
{
    int discords_num = 0;
    int is_discord = 1;
    
    for (unsigned int i = 0; i < node_cand_num; i++)
    {
        is_discord = 1;
        for (unsigned int j = 0; j < ngpus; j++)
        {
            if (r > h_discords_nnDist[j][i])
            {
                is_discord = 0;
                break;
            }
        }
        if (is_discord)
            discords_num++;
    }
    
    return discords_num;
}


void local_candidates_union(DATA_TYPE** h_T, int** h_local_is_cand, DATA_TYPE** h_local_nnDist, float *h_node_C, float* h_node_cand_nnDist, int* h_node_cand_idx, int **N_segments, int N_segment, unsigned int node_cand_num, unsigned int m, int m_i, unsigned int ngpus)
{
    int num_cand = 0;
    int start_segment = 0;
    
    for (unsigned int i = 0; i < ngpus; i++)
    {
        //start_segment = i*N_segments[0][m_i];
        start_segment = i*N_segment;
        
        for (unsigned int j = 0; j < N_segments[i][m_i]; j++)
        {
            if (h_local_is_cand[i][j])
            {
                for (unsigned int k = 0; k < m; k++)
                    h_node_C[node_cand_num*k+num_cand] = h_T[i][j+k];
                h_node_cand_nnDist[num_cand] = h_local_nnDist[i][j];
                h_node_cand_idx[num_cand] = start_segment + j;
                
                num_cand++;
            }
        }
    }
}


void init_global_candidates(DATA_TYPE *h_node_C, DATA_TYPE **h_local_discords_nnDist, int *h_node_cand_idx, DATA_TYPE *h_global_C, int *h_global_cand_idx, int pre_global_cand_num, int node_cand_num, int N_segment, int m, DATA_TYPE r, int rank, int ngpus)
{
    int global_cand_i = 0;
    int is_discord = 1;
    DATA_TYPE min_nnDist = FLT_MAX;
    
    //int start_segment = rank*N_segment;
    int start_segment = rank*ngpus*N_segment;
    
    for (unsigned int i = 0; i < node_cand_num; i++)
    {
        is_discord = 1;
        min_nnDist = FLT_MAX;
        for (unsigned int j = 0; j < ngpus; j++)
        {
            if (r > h_local_discords_nnDist[j][i])
            {
                is_discord = 0;
                break;
            }
            else
                min_nnDist = min(min_nnDist, h_local_discords_nnDist[j][i]);
        }
        if (is_discord)
        {
            h_global_cand_idx[pre_global_cand_num+global_cand_i] = h_node_cand_idx[i]+start_segment;
            
            for (int j = 0; j < m; j++)
                h_global_C[m*(pre_global_cand_num+global_cand_i)+j]= h_node_C[node_cand_num*j+i];
            
            global_cand_i++;
        }
    }
}


void get_node_global_discords(DATA_TYPE **h_global_cand_nnDist, int global_cand_num, int ngpus)
{
    DATA_TYPE min_nndist = FLT_MAX;
    
    for (int i = 0; i < global_cand_num; i++)
    {
        min_nndist = FLT_MAX;
        for (int j = 0; j < ngpus; j++)
            min_nndist = min(min_nndist, h_global_cand_nnDist[j][i]);
        
        h_global_cand_nnDist[0][i] = min_nndist;
    }
}

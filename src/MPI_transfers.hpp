#pragma once

#include <mpi.h>

#include "parameters.hpp"

using namespace std;

void MPI_size_transfer(int *set_sizes, int node_discords_num, int rank, int nproc)
{
    MPI_Status stat;

    for (int i = 0; i < nproc; i++)
    {
        if (i == rank)
        {
            for (int j = 0; j < nproc; j++)
            {
                if (j != rank)
                    MPI_Send(&node_discords_num, 1, MPI_INT, j, 2, MPI_COMM_WORLD);
            }
        }
        else
            MPI_Recv(set_sizes + i, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &stat);
    }
}


void MPI_global_candidates_transfer(DATA_TYPE *h_global_C, int *h_global_cand_idx, int *set_sizes, int local_discords_num, int pre_global_discords_num, int m, int rank, int nproc)
{
    MPI_Status stat;

    int pre_now = 0;
    
    for (int i = 0; i < nproc; i++)
    {
        if (i == rank)
        {
            for (int j = 0; j < nproc; j++)
            {
                if (j != rank)
                {
                    MPI_Send(h_global_cand_idx + pre_global_discords_num, local_discords_num, MPI_INT, j, 3, MPI_COMM_WORLD);
                    MPI_Send(h_global_C + (pre_global_discords_num * m), local_discords_num * m, MPI_DATA_TYPE, j, 4, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            MPI_Recv(h_global_cand_idx + pre_now, set_sizes[i], MPI_INT, i, 3, MPI_COMM_WORLD, &stat);
            MPI_Recv(h_global_C + (pre_now * m), set_sizes[i] * m, MPI_DATA_TYPE, i, 4, MPI_COMM_WORLD, &stat);
        }
        pre_now += set_sizes[i];
    }
}


void MPI_r_transfer(DATA_TYPE *r, int rank, int nproc)
{
    MPI_Status stat;

    if (rank == 0)
    {
        for (int j = 1; j < nproc; j++)
            MPI_Send(r, 1, MPI_DATA_TYPE, j, 5, MPI_COMM_WORLD);
    }
    else
        MPI_Recv(r, 1, MPI_DATA_TYPE, 0, 5, MPI_COMM_WORLD, &stat);
}


void MPI_global_candidates_slave(DATA_TYPE *h_discords_nnDist, int global_cand_num, int rank)
{
    MPI_Send(h_discords_nnDist, global_cand_num, MPI_DATA_TYPE, 0, 7, MPI_COMM_WORLD);
}


void MPI_global_candidates_master(DATA_TYPE *global_cand_nnDist, int global_cand_num, int nproc)
{
    MPI_Status stat;

    for (int j = 1; j < nproc; j++)
        MPI_Recv(global_cand_nnDist+global_cand_num*j, global_cand_num, MPI_DATA_TYPE, j, 7, MPI_COMM_WORLD, &stat);
}


void MPI_top1_discord_transfer(Discord *top1_discord, MPI_Datatype top1_discord_type, int rank, int nproc)
{
    if (rank == 0)
    {
        for (int i = 1; i < nproc; i++)
            MPI_Send(top1_discord, 1, top1_discord_type, i, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(top1_discord, 1, top1_discord_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("In MPI function: top1_discord.ind = %d, top1_discord.dist = %f\n", (*top1_discord).ind, (*top1_discord).dist);
    }
    
}

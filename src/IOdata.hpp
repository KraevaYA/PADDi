#pragma once

#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

#include "parameters.hpp"
#include "PALMAD_types.hpp"

using namespace std;

void read_ts_fscanf(char *file_name, float *data, unsigned int n)
{
    
    FILE * file = fopen(file_name, "rt");
    assert(file != NULL);
    
    int i = 0;
    
    while (!feof(file))
    {
        //fscanf(file, "%lf\n", &data[i]);
        fscanf(file, "%f\n", &data[i]);
        i++;
    }
    
    fclose(file);
    
    for (int j = i; j < n; j++)
        data[j] = FLT_MAX;
}


void read_ts(char *file_name, DATA_TYPE *data, unsigned int n)
{
    
    ifstream infile(file_name); // окрываем файл для чтения
    DATA_TYPE ts_value;
    int i = 0;
    
    if (infile.is_open())
    {
        while (infile >> ts_value)
        {
            data[i] = ts_value;
            i++;
        }
    }
        
    infile.close();

    for (int j = i; j < n; j++)
        data[j] = FLT_MAX;
    
}


void read_ts_segment(char *file_name, DATA_TYPE *data, unsigned int n, int start_segment, int end_segment)
{
    
    ifstream infile(file_name); // окрываем файл для чтения
    DATA_TYPE ts_value;
    int ts_value_i = 0, line_number = 0;
    
    if (infile.is_open())
    {
        while ((infile >> ts_value) && (line_number < end_segment))
        {
            if ((line_number >= start_segment) && (line_number < end_segment))
            {
                data[ts_value_i] = ts_value;
                ts_value_i++;
            }
            line_number++;
        }
    }
  
    infile.close();
    
    for (int j = ts_value_i; j < n; j++)
        data[j] = FLT_MAX;
    
}

void write_range_discord_profile(char *outfile_name, float *profile, unsigned int N, unsigned int m)
{
    // open for output in append mode (create a new file only if the file does not exist)
    ofstream outfile(outfile_name, ios::app);
    
    outfile << m << ";";
    
    // Send data to the stream
    for (int i = 0; i < N; i++)
    {
        if (i != N-1)
            outfile << profile[i] << ";";
        else
            outfile << profile[i] << "\n";
    }
    
    outfile.close();
    
}


void write_discords(char *outfile_name, Discord *top_1_discords, unsigned int num_length, int minL)
{
    vector<string> column_names = {"m", "top-1 discord idx", "top-1 discord nnDist"};
    
    // open for output in append mode (create a new file only if the file does not exist)
    ofstream outfile(outfile_name, ios::app);
    
    
    for (int i = 0; i < column_names.size(); i++)
    {
        outfile << column_names[i];
        
        if (i != column_names.size() - 1)
            outfile << ";"; // No semicolon at end of line
    }
    outfile << "\n";
    
    
    // Send data to the stream
    for (int i = 0; i < num_length; i++)
    {
        outfile << minL+i << ";" << top_1_discords[i].ind << ";" << top_1_discords[i].dist << "\n";
    }
    
    outfile.close();
    
}


void write_times_tbl(char *outfile_name, vector<vector<double>> times_tbl)
{
    vector<string> column_names = {"m", "Phase1: find of segment's discords (rank = 0), ms", "max statistics among GPUs (kernel), ms", "max PD3 among GPUs (kernel), ms",
        "Phase2: find of fragment's discords (rank = 0), ms", "max MatrixMultiply among GPUs (kernel), ms", "# Candidates",
        "malloc memory, ms", "MPI exchange C, ms", "Phase3: find of global discords (rank = 0), ms", "max MatrixMultiply among GPUs (kernel), ms",
        "MPI exchange D, ms", "# Discords", "Total time PALMAD (rank = 0), ms"};
    
    // open for output in append mode (create a new file only if the file does not exist)
    ofstream outfile(outfile_name, ios::app);
    
    //write the first row with names
    // Send column names to the stream
    for (int i = 0; i < column_names.size(); i++)
    {
        outfile << column_names[i];
        
        if (i != column_names.size() - 1)
            outfile << ";"; // No semicolon at end of line
    }
    outfile << "\n";
    
    
    // Send data to the stream
    for (int i = 0; i < times_tbl.size(); i++)
    {
        for (int j = 0; j < times_tbl[i].size(); j++)
        {
            outfile << times_tbl[i][j];
            if (j != times_tbl[i].size() - 1)
                outfile << ";"; // No semicolon at end of line
        }
        outfile << "\n";
    }
    
    // Close the file
    outfile.close();
    
}


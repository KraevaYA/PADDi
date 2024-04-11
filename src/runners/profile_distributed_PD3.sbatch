#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=../output/slurm-%j.out
 
#/usr/local/cuda/bin/nsys 
#/usr/local/cuda/bin/nvprof -o timeline_short ./main ../datasets/power_demand/input0000.txt 5000 32

/usr/local/cuda/bin/nsys profile -o timeline_time --stats true --trace cuda,nvtx,osrt ./main ../datasets/power_demand/input0000.txt 5000 32

#/usr/local/cuda/bin/ncu --target-processes all -o timeline_compute ./main ../datasets/power_demand/input0000.txt 5000 32
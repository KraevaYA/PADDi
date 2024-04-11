#!/bin/bash

#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out


mpirun ./main ../datasets/DAMP/ECG/ECG250K.txt 250000 64 128 ../results/discords/ECG250K_64_128_32cpu_P100_test1.csv ../results/times/ECG250K_64_128_32cpu_P100_test1.csv


##mpirun ./main ../datasets/MGAB/MGAB_500K.txt 500000 128 256 ../results/discords/MGAB500K_128_256_8cpu_K40.csv ../results/times/MGAB500K_128_256_8cpu_K40.csv

##mpirun ./main ../datasets/MGAB/MGAB_500K.txt 500000 256 512 ../results/discords/MGAB500K_256_512_8cpu_K40.csv ../results/times/MGAB500K_256_512_8cpu_K40.csv
 
##mpirun ./main ../datasets/power_demand/power_demand5000.txt 5000 128 512 ../results/discords/power_demand5000_128_512_4cpu_K40_new_2r.csv ../results/times/power_demand5000_128_512_4cpu_K40_new_2r.csv

##mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk250K.txt 250000 64 128 ../results/discords/RandomWalk1_250K_64_128_2cpu_K40_test.csv ../results/times/RandomWalk1_250K_64_128_2cpu_K40_test.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP250K.txt 250000 256 300 ../results/discords/GAP250K_256_512_2cpu_K40.csv ../results/times/GAP250K_256_512_2cpu_K40.csv

##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512
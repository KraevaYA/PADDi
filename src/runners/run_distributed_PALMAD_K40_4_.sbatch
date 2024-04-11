#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out

##mpirun ./main ../datasets/DAMP/ECG/ECG250K.txt 250000 64 128 ../results/discords/ECG250K_64_128_2cpu_K40.csv ../results/times/ECG250K_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG500K.txt 500000 64 128 ../results/discords/ECG500K_64_128_2cpu_K40.csv ../results/times/ECG500K_64_128_2cpu_K40.csv
 
##mpirun ./main ../datasets/DAMP/ECG/ECG1M.txt 1000000 64 128 ../results/discords/ECG1M_64_128_2cpu_K40.csv ../results/times/ECG1M_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 64 128 ../results/discords/ECG2M_64_128_2cpu_K40.csv ../results/times/ECG2M_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG4M.txt 4000000 64 128 ../results/discords/ECG4M_64_128_2cpu_K40.csv ../results/times/ECG4M_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk250K.txt 250000 64 128 ../results/discords/RandomWalk250K_64_128_2cpu_K40.csv ../results/times/RandomWalk250K_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk500K.txt 500000 64 128 ../results/discords/RandomWalk500K_64_128_2cpu_K40.csv ../results/times/RandomWalk500K_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk1M.txt 1000000 64 128 ../results/discords/RandomWalk1M_64_128_2cpu_K40.csv ../results/times/RandomWalk1M_64_128_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG250K.txt 250000 128 256 ../results/discords/ECG250K_128_256_2cpu_K40.csv ../results/times/ECG250K_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG500K.txt 500000 128 256 ../results/discords/ECG500K_128_256_2cpu_K40.csv ../results/times/ECG500K_128_256_2cpu_K40.csv
 
##mpirun ./main ../datasets/DAMP/ECG/ECG1M.txt 1000000 128 256 ../results/discords/ECG1M_128_256_2cpu_K40.csv ../results/times/ECG1M_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 128 256 ../results/discords/ECG2M_128_256_2cpu_K40.csv ../results/times/ECG2M_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG4M.txt 4000000 128 256 ../results/discords/ECG4M_128_256_2cpu_K40.csv ../results/times/ECG4M_128_256_2cpu_K40.csv




##mpirun ./main ../datasets/VALMOD/GAP/GAP250K.txt 250000 128 256 ../results/discords/GAP250K_128_256_2cpu_K40.csv ../results/times/GAP250K_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP500K.txt 500000 128 256 ../results/discords/GAP500K_128_256_2cpu_K40.csv ../results/times/GAP500K_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP1M.txt 1000000 128 256 ../results/discords/GAP1M_128_256_2cpu_K40.csv ../results/times/GAP1M_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP2M.txt 2000000 128 256 ../results/discords/GAP2M_128_256_2cpu_K40.csv ../results/times/GAP2M_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP250K.txt 250000 256 512 ../results/discords/GAP250K_256_512_2cpu_K40.csv ../results/times/GAP250K_256_512_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP500K.txt 500000 256 512 ../results/discords/GAP500K_256_512_2cpu_K40.csv ../results/times/GAP500K_256_512_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP1M.txt 1000000 256 512 ../results/discords/GAP1M_256_512_2cpu_K40.csv ../results/times/GAP1M_256_512_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/GAP/GAP2M.txt 2000000 256 512 ../results/discords/GAP2M_256_512_2cpu_K40.csv ../results/times/GAP2M_256_512_2cpu_K40.csv




mpirun ./main ../datasets/VALMOD/GAP/GAP250K.txt 250000 64 128 ../results/discords/GAP250K_64_128_4cpu_K40.csv ../results/times/GAP250K_64_128_4cpu_K40.csv

mpirun ./main ../datasets/VALMOD/GAP/GAP500K.txt 500000 64 128 ../results/discords/GAP500K_64_128_4cpu_K40.csv ../results/times/GAP500K_64_128_4cpu_K40.csv

mpirun ./main ../datasets/VALMOD/GAP/GAP1M.txt 1000000 64 128 ../results/discords/GAP1M_64_128_4cpu_K40.csv ../results/times/GAP1M_64_128_4cpu_K40.csv

mpirun ./main ../datasets/VALMOD/GAP/GAP2M.txt 2000000 64 128 ../results/discords/GAP2M_64_128_4cpu_K40.csv ../results/times/GAP2M_64_128_4cpu_K40.csv


mpirun ./main ../datasets/DAMP/ECG/ECG250K.txt 250000 256 512 ../results/discords/ECG250K_256_512_4cpu_K40.csv ../results/times/ECG250K_256_512_4cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG500K.txt 500000 256 512 ../results/discords/ECG500K_256_512_4cpu_K40.csv ../results/times/ECG500K_256_512_4cpu_K40.csv
 
mpirun ./main ../datasets/DAMP/ECG/ECG1M.txt 1000000 256 512 ../results/discords/ECG1M_256_512_4cpu_K40.csv ../results/times/ECG1M_256_512_4cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 256 512 ../results/discords/ECG2M_256_512_4cpu_K40.csv ../results/times/ECG2M_256_512_4cpu_K40.csv



##mpirun ./main ../datasets/VALMOD/ASTRO/ASTRO250K.txt 250000 128 256 ../results/discords/ASTRO250K_128_256_2cpu_K40.csv ../results/times/ASTRO250K_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/ASTRO/ASTRO500K.txt 500000 128 256 ../results/discords/ASTRO500K_128_256_2cpu_K40.csv ../results/times/GAP500K_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/ASTRO/ASTRO1M.txt 1000000 128 256 ../results/discords/ASTRO1M_128_256_2cpu_K40.csv ../results/times/GAP1M_128_256_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/ASTRO/ASTRO250K.txt 250000 256 512 ../results/discords/ASTRO250K_256_512_2cpu_K40.csv ../results/times/GAP250K_256_512_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/ASTRO/ASTRO500K.txt 500000 256 512 ../results/discords/ASTRO500K_256_512_2cpu_K40.csv ../results/times/GAP500K_256_512_2cpu_K40.csv

##mpirun ./main ../datasets/VALMOD/ASTRO/ASTRO1M.txt 1000000 256 512 ../results/discords/ASTRO1M_256_512_2cpu_K40.csv ../results/times/GAP1M_256_512_2cpu_K40.csv
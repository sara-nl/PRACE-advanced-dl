#!/bin/bash

#SBATCH -t 1:50:00
#SBATCH -N 8
#SBATCH --tasks-per-node 2
#SBATCH -p broadwell
#SBATCH --reservation=patc_bdw8
ulimit -a
module load Python/3.6.3-foss-2017b
source hdis/bin/activate

mpirun -np 16 --map-by ppr:1:socket:pe=16 python keras-cifar10-resnet.py

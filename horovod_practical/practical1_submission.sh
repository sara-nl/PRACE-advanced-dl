#!/bin/bash

#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH -p broadwell
#SBATCH --reservation=patc_bdw8
ulimit -a
module load Python/3.6.3-foss-2017b
source hdis/bin/activate

time python practical1_clear_code.py
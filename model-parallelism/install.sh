#!/bin/bash

module purge
module load Python/2.7.14-foss-2017b
module load CUDA/10.0.130
module load cuDNN/7.4.2-CUDA-10.0.130
module load NCCL/2.3.5-CUDA-10.0.130

pip install tensorflow-gpu --user --no-cache-dir
pip install mesh-tensorflow --user --no-cache-dir



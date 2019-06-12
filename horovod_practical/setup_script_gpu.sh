#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -n 2
#SBATCH -p gpu

#############################
##  Environment setup Cartesius ##
#############################
echo "[...] Environment setup Cartesius"
virtualenv_folder="hdisgpu"

#Loads needed modules
module load NCCL/2.3.5-CUDA-10.0.130 cuDNN/7.4.2-CUDA-10.0.130 Python/3.6.3-foss-2017b || exit

#this deletes any existing virtual environments. Take it out if you want to just reuse it
if [ -d ${virtualenv_folder} ]; then
    echo "[...] Removed the ${virtualenv_folder} folder"
    rm -rf ${virtualenv_folder}
fi

virtualenv --no-site-packages ${virtualenv_folder}
source ${virtualenv_folder}/bin/activate || exit
echo "[...] Packages existing in the virtualenv before deploying our own"
pip list
pip install tensorflow-gpu --no-cache
CC=mpicc CXX=mpicxx HOROVOD_NCCL_HOME=$EBROOTNCCL HOROVOD_GPU_ALLREDUCE=NCCL pip install horovod --no-cache
pip install keras --no-cache
echo "[...] Sanity check for horovod"
ranks=2
mpirun -np ${ranks} --map-by ppr:1:socket:pe=8 python -c "import horovod.tensorflow as hvd; hvd.init(); print('[...] Hello world from rank {0}'.format(hvd.rank()))"
echo "[...] The previous command shoud've printed ${ranks} 'Hello world lines'"
echo "[...] Packages existing in the virtualenv after deploying our own"
pip list
echo "[...] In the future you can activate this venv by:"
echo "source $(realpath ${virtualenv_folder}/bin/activate)"
deactivate
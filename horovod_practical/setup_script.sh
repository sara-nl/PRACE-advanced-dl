#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -n 1
#SBATCH -p broadwell

#############################
##  Environment setup Cartesius ##
#############################
echo "[...] Environment setup Cartesius"
virtualenv_folder="hdis"

#Loads needed modules
module purge && module load Python/3.6.3-foss-2017b || exit

#this deletes any existing virtual environments. Take it out if you want to just reuse it
if [ -d ${virtualenv_folder} ]; then
    echo "[...] Removed the ${virtualenv_folder} folder"
    rm -rf ${virtualenv_folder}
fi

virtualenv --no-site-packages ${virtualenv_folder}
source ${virtualenv_folder}/bin/activate || exit
echo "[...] Packages existing in the virtualenv before deploying our own"
pip list
pip install intel-tensorflow --no-cache
CC=mpicc CXX=mpicxx pip install horovod --no-cache
echo "[...] Sanity check for horovod"
ranks=16
mpirun -np ${ranks} python -c "import horovod.tensorflow as hvd; hvd.init(); print('[...] Hello world from rank {0}'.format(hvd.rank()))"
echo "[...] The previous command shoud've printed ${ranks} 'Hello world lines'"
echo "[...] Packages existing in the virtualenv after deploying our own"
pip list
echo "[...] In the future you can activate this venv by:"
echo "source $(realpath ${virtualenv_folder}/bin/activate)"
deactivate
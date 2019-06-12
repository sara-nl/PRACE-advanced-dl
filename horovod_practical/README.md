In this handson session we will use [Tensorflow](https://github.com/tensorflow/tensorflow), [Keras](https://github.com/keras-team/keras) and [Horovod](https://github.com/horovod/horovod).

## Setup Cartesius (est. 5-10 min)

### CPU partition
A script [```setup_script.sh```](setup_script.sh) has been created for setting up your environment. The main commands are:
```bash
module load Python/3.6.3-foss-2017b
virtualenv --no-site-packages hdis
source hdis/bin/activate
pip install intel-tensorflow --no-cache
CC=mpicc CXX=mpicxx pip install horovod --no-cache
```

### GPU partition 
```bash
module load NCCL/2.3.5-CUDA-10.0.130 cuDNN/7.4.2-CUDA-10.0.130 Python/3.6.3-foss-2017b
virtualenv --no-site-packages hdisgpu
source hdisgpu/bin/activate
pip install tensorflow-gpu --no-cache
CC=mpicc CXX=mpicxx HOROVOD_NCCL_HOME=$EBROOTNCCL HOROVOD_GPU_ALLREDUCE=NCCL pip install horovod --no-cache
```

## First exercise (first half hour)
Can be found in [```practical1_clear_code.py```](practical1_clear_code.py). The solution is [```practical1_solution.py```](practical1_solution.py)

A few practical links: [Chain rule](https://en.wikipedia.org/wiki/Chain_rule), [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), [Tensorflow eager execution and automatic differentiation](https://www.tensorflow.org/tutorials/eager/automatic_differentiation)

### Sub-exercise
Take a look at https://github.com/horovod/horovod/blob/master/docs/tensor-fusion.rst

Try to set HOROVOD_FUSION_THRESHOLD to 0 and HOROVOD_CYCLE_TIME to 0. Remember, passing variables to the MPI execution environment is done (in OpenMPI) with the ```-x``` options, i.e. ```-x HOROVOD_FUSION_THRESHOLD```

## Second exercise (second half hour)
The exercise starts from https://github.com/yhgon/horovod-tf-uber/blob/master/examples/keras-cifar10-resnet.py

We first add, as usual, some performance boilerplate:
```
os.environ['KMP_BLOCKTIME'] = str(0)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['OMP_NUM_THREADS'] = str(15)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0’
```

Convergence is poor without adjusting the learning rate schedule. All these are visible in [```practical2.py```](practical2.py)


## Submission scripts
For your convenience, SLURM submission scripts were created for each practical: [```practical1_submission.sh```](practical1_submission.sh) and [```practical2_submission.sh```](practical2_submission.sh)
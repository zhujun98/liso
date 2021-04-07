#!/bin/bash
#SBATCH --partition=all
#SBATCH --job-name=spark-jupyter # create a short name for your job
#SBATCH --nodes=3                # node count
#SBATCH --ntasks-per-node=20     # number of tasks per node
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem 256G               # real memory required per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
unset LD_PRELOAD

TEC=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE * $SLURM_CPUS_PER_TASK))
EM=$(($SLURM_MEM_PER_NODE / $SLURM_NTASKS_PER_NODE / 1024))

source /etc/profile.d/modules.sh
module load maxwell spark/3.0.2
# The Python version used to launch the cluster and to run pyspark must be consistent
conda activate liso

export SPARK_MASTER_URL=spark://$(hostname -f):7077
export SPARK_CONF_DIR=$PWD/conf
export SPARK_LOG_DIR=$PWD/logs
export SPARK_WORKER_DIR=$PWD/logs

bin/spark_start
echo ${SPARK_MASTER_URL} | tee master.txt

sleep infinity
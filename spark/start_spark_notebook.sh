source /etc/profile.d/modules.sh
module load maxwell spark/3.0.2
# The Python version used to launch the cluster and to run pyspark must be consistent
conda activate liso

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --port=8889 --ip=127.0.0.1'
export PYSPARK_PYTHON=`which python`

pyspark --master $(head -n 1 master.txt) --total-executor-cores 120

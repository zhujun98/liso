# LISO with Apache Spark

LISO supports limited data analysis pipeline with Apache Spark on the [Maxwell cluster](https://confluence.desy.de/display/IS/Maxwell).

## Test Spark cluster setup and job submission


```sh
cd spark-cluster-test
# Launch the Spark cluster
sbatch spark_pi.sh
```

You should get the similar output as follows

```
Starting master on spark://max-exfl093.desy.de:7077

starting org.apache.spark.deploy.master.Master, logging to /home/zhujun/BEEGFS/liso/spark/spark-cluster-test/logs/spark-zhujun-org.apache.spark.deploy.master.Master-1-max-exfl093.out

Starting workers

spark://max-exfl093.desy.de:7077
Pi is roughly 3.139720
```

## Running Spark with Jupyter notebook

Create a `Conda` environment named `liso`.

```sh
# Launch the Spark cluster
sbatch spark_with_jupyter.sh
```

SSH to the master node and launch a standalone Jupyter notebook by

```sh
./start_spark_notebook.sh
```

You will need a two-hop local port forwarding if you are outside DESY network

```
ssh -L 8889:localhost:8889 <username>@bastion.desy.de ssh -L 8889:localhost:8889 -N <username>@<master node hostname>
```

Open your notebook at http://localhost:8889.

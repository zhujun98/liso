#!/bin/bash
#SBATCH --partition maxwell
#SBATCH --time    1-00:00:00
#SBATCH --nodes            1
#SBATCH --job-name hostname
#SBATCH --mincpus         72
source /beegfs/desy/user/zhujun/anaconda3/bin/activate
conda activate liso
cd /beegfs/desy/user/zhujun/liso/examples/astra_scan
python astra_scan.py --cluster
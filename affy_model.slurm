#!/bin/bash

#SBATCH --mem=100000
#SBATCH --time=3-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --output=/home/ty.werbicki/slurm_outs/heng_model-%j.out

echo "Job started: "$(date)
echo "CPUS available to job: "$SLURM_JOB_CPUS_PER_NODE

COMPUTE_HOME=$HOME/compute_scripts

cd /work/slater_lab/shared_compute_scripts
cp py_expressionset.py affymetrix_MLP.py feature_selection.py reporting.py $COMPUTE_HOME
cd $COMPUTE_HOME
source /work/slater_lab/anaconda3/bin/activate ml

python heng_model.py

conda deactivate
rm PyExpressionset.py affymetrix_MLP.py feature_selection.py reporting.py
rm -r __pycache__

echo "Job complete: "$(date)

#!/bin/bash
#SBATCH -n 16
#SBATCH -J "DMC Training"
#SBATCH -p long
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:4
# All the gpus

module load python/3.11.6/ao5olvd
module load cuda
module load miniconda3
cud="0,1,2,3"
SECONDS=0

cd ~/rlcard

# conda env setup already done manually
conda activate rlcard

mkdir -p experiments/gin_rummy_dmc_result

echo "Training DMC"

python3 examples/run_dmc.py --env gin-rummy --cuda "" --xpid gin_rummy --save_interval 60 --savedir experiments/gin_rummy_dmc_result

#Display time it took
echo $(($SECONDS/86400))d $(($(($SECONDS - $SECONDS/86400*86400))/3600))h:$(($(($SECONDS - $SECONDS/86400*86400))%3600/60))m:$(($(($SECONDS - $SECONDS/86400*86400))%60))s
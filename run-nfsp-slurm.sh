#!/bin/bash
#SBATCH -n 16
#SBATCH -J "NFSP Training"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:8
# All the gpus

module load python/3.11.6/ao5olvd
module load cuda
module load miniconda3
cud="0,1,2,3,4,5,6,7"
SECONDS=0

cd ~/rlcard

# conda env setup already done manually
conda activate rlcard

rm -rf experiments/gin_rummy_nfsp_result
mkdir -p experiments/gin_rummy_nfsp_result

echo "Beginning NFSP model training"

python3 examples/run_rl.py --env gin-rummy --algorithm nfsp --num_episodes 100000 --cuda $cud --save_every 5000 --log_dir experiments/gin_rummy_nfsp_result

#Display time it took 
echo $(($SECONDS/86400))d $(($(($SECONDS - $SECONDS/86400*86400))/3600))h:$(($(($SECONDS - $SECONDS/86400*86400))%3600/60))m:$(($(($SECONDS - $SECONDS/86400*86400))%60))s
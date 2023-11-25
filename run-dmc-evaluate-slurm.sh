#!/bin/bash
#SBATCH -n 16
#SBATCH -J "DMC Training"
#SBATCH -p long
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
# One GPU

module load python/3.11.6/ao5olvd
module load cuda
module load miniconda3
cud="0"
SECONDS=0

cd ~/rlcard

# conda env setup already done manually
conda activate rlcard

# Use Saved Model for DMC Because it is not done training yet and we have an assignment due
python3 examples/evaluate.py --env gin-rummy --cuda "" --models experiments/gin_rummy_dmc_result/gin_rummy/0_0.pth random > experiments/gin_rummy_dmc_result/vsrandom.txt
python3 examples/evaluate.py --env gin-rummy --cuda "" --models experiments/gin_rummy_dmc_result/gin_rummy/0_0.pth gin-rummy-novice-rule > experiments/gin_rummy_dmc_result/vsrule.txt

# Evaluate DMC (temp saved) vs NFSP (trained) vs DQN (trained)
python3 examples/evaluate.py --env gin-rummy --cuda $cud --models experiments/gin_rummy_dmc_result/gin_rummy/0_0.pth experiments/gin_rummy_nfsp_result/model.pth experiments/gin_rummy_dqn_result/model.pth > experiments/dmc_vs_nfsp_vs_dqn.txt

#Display time it took 
echo $(($SECONDS/86400))d $(($(($SECONDS - $SECONDS/86400*86400))/3600))h:$(($(($SECONDS - $SECONDS/86400*86400))%3600/60))m:$(($(($SECONDS - $SECONDS/86400*86400))%60))s

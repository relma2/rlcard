#!/bin/bash
#SBATCH -n 16
#SBATCH -J "Evaluate Models"
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

python3 examples/evaluate.py --env gin-rummy --cuda $cud --models experiments/gin_rummy_nfsp_result/model.pth random > experiments/gin_rummy_nfsp_result/vsrandom.txt
python3 examples/evaluate.py --env gin-rummy --cuda $cud --models experiments/gin_rummy_dqn_result/model.pth random > experiments/gin_rummy_dqn_result/vsrandom.txt

python3 examples/evaluate.py --env gin-rummy --cuda $cud --models experiments/gin_rummy_nfsp_result/model.pth gin-rummy-novice-rule > experiments/gin_rummy_nfsp_result/vsrule.txt
python3 examples/evaluate.py --env gin-rummy --cuda $cud --models experiments/gin_rummy_dqn_result/model.pth gin-rummy-novice-rule > experiments/gin_rummy_dqn_result/vsrule.txt

python3 examples/evaluate.py --env gin-rummy --cuda $cud --models experiments/gin_rummy_nfsp_result/model.pth experiments/gin_rummy_dqn_result/model.pth > experiments/nfsp_vs_dqn.txt

#Display time it took 
echo $(($SECONDS/86400))d $(($(($SECONDS - $SECONDS/86400*86400))/3600))h:$(($(($SECONDS - $SECONDS/86400*86400))%3600/60))m:$(($(($SECONDS - $SECONDS/86400*86400))%60))s
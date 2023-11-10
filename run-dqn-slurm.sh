#!/bin/bash
#SBATCH -n 16
#SBATCH -J "DQN Training"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:8
# All the gpus

module load python/3.11.6/ao5olvd
module load cuda
module load miniconda3
cud="0,1,2,3,4,5,6,7"

cd ~/rlcard

# conda env setup already done manually
conda activate rlcard

rm -rf experiments/gin_rummy_dqn_result
mkdir -p experiments/gin_rummy_dqn_result

echo "Beginning DQN model training"

python3 examples/run_rl.py --env gin-rummy --algorithm dqn  --num_episodes 10000 --cuda $cud --save_every 2000 --log_dir experiments/gin_rummy_dqn_result

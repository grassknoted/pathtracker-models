#!/bin/bash
#SBATCH --time=300:00:00
#SBATCH -p gpu --gres=gpu:4
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=24G
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
#SBATCH -J PTO64141
## Specify an output file
#SBATCH -o PTO_64_14_1_32dim_2000epochs_180bs_3e-4.out
#SBATCH -e PTO_64_14_1_32dim_2000epochs_180bs_3e-4.err

module load anaconda/3-5.2.0
source activate newtrack
module load gcc/8.3
module load cuda/10.2

export PYTHONPATH=/users/akarkada/axial-positional-embedding:$PYTHONPATH
export PYTHONPATH=/users/akarkada/Pytorch-Correlation-extension:$PYTHONPATH
#python mainclean.py --print-freq 20 --lr 1e-02 --epochs 2000 -b 180 --model ffhgru_soft --name hgru_soft_1e-2 --log --length 64 --speed 1 --dist 14  --parallel
#rm -r results/PT_O_64_32_32_14_dist_int_circuit_bs_180_hgru_dim_32_2000_epochs_3e-4_parallel/
#python -u ../mainclean_edited_1.py --print-freq 20 --lr 3e-4 --epochs 2000 -b 180 --algo rbp --model ffhgru_new --name PT_O_64_32_32_14_dist_int_circuit_bs_180_hgru_dim_32_2000_epochs_3e-4_parallel --log --parallel

python -u ../mainclean.py --print-freq 20 --lr 3e-4 --epochs 2000 -b 180 --model ffhgru --name int_64_14_occ_1 --log --parallel --length 64 --thickness 1 --dist 14


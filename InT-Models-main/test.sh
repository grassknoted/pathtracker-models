#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=24G
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
#SBATCH -J test_all
## Specify an output file
#SBATCH -o test_all_32dim_2000epochs_180bs_3e-4.out
#SBATCH -e test_all_32dim_2000epochs_180bs_3e-4.err

module load anaconda/3-5.2.0
source activate newtrack
module load gcc/8.3
module load cuda/10.2

export PYTHONPATH=/users/akarkada/axial-positional-embedding:$PYTHONPATH
export PYTHONPATH=/users/akarkada/Pytorch-Correlation-extension:$PYTHONPATH

#python -u ../mainclean.py --print-freq 20 --lr 3e-4 --epochs 2000 -b 180 --model ffhgru --name int_64_14_occ_1 --log --parallel --length 64 --thickness 1 --dist 14



# trained and tested on 64
echo "trained and tested on 64 \n"
echo "64 25 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_25_occ_9 --parallel --length=64 --thickness=9 --dist=25 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_25_occ_5 --parallel --length=64 --thickness=5 --dist=25 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_25_occ_3 --parallel --length=64 --thickness=3 --dist=25 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_25_occ_1 --parallel --length=64 --thickness=1 --dist=25 --which_tests=64

echo "64 14 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_14_occ_9 --parallel --length=64 --thickness=9 --dist=14 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_14_occ_5 --parallel --length=64 --thickness=5 --dist=14 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_14_occ_3 --parallel --length=64 --thickness=3 --dist=14 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_14_occ_1 --parallel --length=64 --thickness=1 --dist=14 --which_tests=64

echo "64 0 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_0_occ_9 --parallel --length=64 --thickness=9 --dist=0 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_0_occ_5 --parallel --length=64 --thickness=5 --dist=0 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_0_occ_3 --parallel --length=64 --thickness=3 --dist=0 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_64_0_occ_1 --parallel --length=64 --thickness=1 --dist=0 --which_tests=64


# trained and tested on 128
echo "trained and tested on 128 \n"
echo "128 25 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_9 --parallel --length=128 --thickness=9 --dist=25 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_5 --parallel --length=128 --thickness=5 --dist=25 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_3 --parallel --length=128 --thickness=3 --dist=25 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_1 --parallel --length=128 --thickness=1 --dist=25 --which_tests=128

echo "128 14 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_9 --parallel --length=128 --thickness=9 --dist=14 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_5 --parallel --length=128 --thickness=5 --dist=14 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_3 --parallel --length=128 --thickness=3 --dist=14 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_1 --parallel --length=128 --thickness=1 --dist=14 --which_tests=128

echo "128 0 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_0_occ_9 --parallel --length=128 --thickness=9 --dist=0 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_0_occ_5 --parallel --length=128 --thickness=5 --dist=0 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_0_occ_3 --parallel --length=128 --thickness=3 --dist=0 --which_tests=128
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_0_occ_1 --parallel --length=128 --thickness=1 --dist=0 --which_tests=128



# trained on 128/14, tested on 64
echo "128 14 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_9 --parallel --length=128 --thickness=9 --dist=14 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_5 --parallel --length=128 --thickness=5 --dist=14 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_3 --parallel --length=128 --thickness=3 --dist=14 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_14_occ_1 --parallel --length=128 --thickness=1 --dist=14 --which_tests=64


# trained on 128/25, tested on 64
echo "128 25 dist \n"
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_9 --parallel --length=128 --thickness=9 --dist=25 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_5 --parallel --length=128 --thickness=5 --dist=25 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_3 --parallel --length=128 --thickness=3 --dist=25 --which_tests=64
python -u ../test_model.py --print-freq 20 --lr 3e-4 --epochs 300 --model ffhgru --name int_128_25_occ_1 --parallel --length=128 --thickness=1 --dist=25 --which_tests=64


CUDA_VISIBLE_DEVICES=4,5,6,7 
python mainclean.py --print-freq 20 --lr 3e-4 --epochs 2000 -b 180 --model ffhgru --name int_32_14_occ_5 --log --parallel --length 32 --thickness 3 --dist 14

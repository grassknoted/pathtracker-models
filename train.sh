CUDA_VISIBLE_DEVICES=2,3 
python3 mainclean.py --print-freq 20 --lr 3e-4 --epochs 2000 -b 128 --model ffhgru --name int_64_14_occ_5 --log --parallel --length 64 --thickness 1 --dist 14

#!/bin/bash
#
#SBATCH --job-name=r3d_bsign
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000
#SBATCH --partition=main
#SBATCH --mail-type=END
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/r3d-out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/r3d-err-%j.err  # send stderr to errfile

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
                                                                --dataset=bsign \
                                                                --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
                                                                --shuffle-train \
                                                                --frame-size=112x112 \
                                                                --horizontal-flip \
                                                                --normalize 0 1 \
                                                                --standardize-mean 0.43216 0.394666 0.37645 \
                                                                --standardize-std 0.22803 0.22145 0.216989 \
                                                                --models=r3d_18 \
                                                                --modality=rgb \
                                                                --sampling=equidistant \
                                                                --clip-length=16 \
                                                                --log-level=info \
                                                                --batch-size=32 \
                                                                --num-workers=4 \
                                                                --criterion=cross_entropy \
                                                                --optimizer=adam \
                                                                --learning-rate=1e-3 \
                                                                --epoch=500 \
                                                                --weight-initializer=xavier_normal \
                                                                --pretrained \
                                                                --layers layer2 layer3 layer4 fc
#!/bin/bash
#
#SBATCH --job-name=c3d_bsign
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000
#SBATCH --partition=main
#SBATCH --mail-type=END
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/err-%j.err  # send stderr to errfile

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
                                                                --dataset=bsign \
                                                                --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
                                                                --frame-size=112x112 \
                                                                --shuffle-train \
                                                                --horizontal-flip \
                                                                --normalize "-1" "1" \
                                                                --models=c3d \
                                                                --modality=rgb \
                                                                --sampling=equidistant \
                                                                --clip-length=16 \
                                                                --log-level=info \
                                                                --batch-size=48 \
                                                                --num-workers=4 \
                                                                --criterion=cross_entropy \
                                                                --optimizer=adam \
                                                                --learning-rate=1e-3 \
                                                                --epoch=200 \
                                                                --weight-initializer=xavier_normal \
                                                                --pretrained-weights=/raid/users/oozdemir/code/untitled-slr-project/models/weights/c3d.pickle \
                                                                --crop-mean=/raid/users/oozdemir/code/untitled-slr-project/models/weights/crop_mean_16_bsign.npy

#                                                                --dataset=toydata \
#                                                                --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_150x150/ \
#                                                                --shuffle-train \
#                                                                --models=c3d \
#                                                                --pretrained-weights=/raid/users/oozdemir/code/untitled-slr-project/models/weights/c3d.pickle \
#                                                                --crop-mean=/raid/users/oozdemir/code/untitled-slr-project/models/weights/crop_mean_16_bsign.npy \
#                                                                --modality=rgb \
#                                                                --sampling=random \
#                                                                --clip-length=16 \
#                                                                --random-crop=112 \
#                                                                --horizontal-flip \
#                                                                --normalize \
#                                                                --dropout=0.5 \
#                                                                --batch-size=30 \
#                                                                --num-workers=4 \
#                                                                --criterion=cross_entropy \
#                                                                --optimizer=sgd \
#                                                                --learning-rate=3e-3 \
#                                                                --weight-decay=5e-3 \
#                                                                --epoch=100 \
#                                                                --weight-initializer=xavier_normal \
#                                                                --scheduler=step_lr \
#                                                                --scheduler-step=1200 \
#                                                                --scheduler-factor=0.1 \
#                                                                --log-level=info

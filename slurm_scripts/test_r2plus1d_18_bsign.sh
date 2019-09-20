#!/bin/bash
#
#SBATCH --job-name=test_sampling_user4_r2plus1d_18_bsign
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000
#SBATCH --partition=main
#SBATCH --mail-type=END
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/test_r2plus1d_18-out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/test_r2plus1d_18-err-%j.err  # send stderr to errfile

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=random \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/sampling_experiments/experiment_19.08.2019-00.02.59_bsign_r2plus1d_18_rgb_random_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/sampling_experiments/experiment_20.08.2019-10.31.21_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

####
/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_29.08.2019-12.29.21_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_29.08.2019-12.30.20_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_29.08.2019-12.30.30_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_30.08.2019-18.55.23_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_31.08.2019-14.04.22_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/test.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --shuffle-train \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=r2plus1d_18 \
    --modality=rgb \
    --sampling=equidistant \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --test-mode on \
    --experiment-path /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_01.09.2019-03.24.16_bsign_r2plus1d_18_rgb_equidistant_clip16_batch24_adam_cross_entropy_lr0.001/ \
    --plot-confusion

#!/bin/bash
#
#SBATCH --job-name=feature_extract_mc3_18_bsign
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000
#SBATCH --partition=main
#SBATCH --mail-type=END
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/extract_mc3_18-out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/extract_mc3_18-err-%j.err  # send stderr to errfile

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/extract_features.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/frames_112x112/ \
    --frame-size=112x112 \
    --normalize 0 1 \
    --standardize-mean 0.43216 0.394666 0.37645 \
    --standardize-std 0.22803 0.22145 0.216989 \
    --models=mc3_18 \
    --modality=rgb \
    --sampling=feature16_overlap \
    --clip-length=16 \
    --log-level=info \
    --batch-size=1 \
    --num-workers=4 \
    --experiment-path \
    /raid/users/oozdemir/code/untitled-slr-project/experiments/layer_experiments/experiment_28.08.2019-07.07.08_bsign_mc3_18_rgb_equidistant_clip16_batch32_adam_cross_entropy_lr0.001 \
    --feature-path=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap \
    --feature-layer=avgpool
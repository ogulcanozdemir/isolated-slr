#!/bin/bash
#
#SBATCH --job-name=c3d_general
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000
#SBATCH --partition=main
#SBATCH --mail-type=END
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/err-%j.err  # send stderr to errfile

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train_c3d.py

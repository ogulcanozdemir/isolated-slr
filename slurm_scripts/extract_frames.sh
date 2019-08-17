#!/bin/bash

#SBATCH --job-name=extract_frames
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=main
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/frame-out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/frame-err-%j.err  # send stderr to errfile

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python  /raid/users/oozdemir/code/untitled-slr-project/datasets/extract_frames.py
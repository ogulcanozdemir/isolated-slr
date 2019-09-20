#!/bin/bash
#
#SBATCH --job-name=overlap_seq_bsign4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000
#SBATCH --partition=main
#SBATCH --mail-type=END
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH -o /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/seq-out-%j.out  # send stdout to outfile
#SBATCH -e /raid/users/oozdemir/code/untitled-slr-project/slurm_scripts/outputs/seq-18-err-%j.err  # send stderr to errfile

## Single directional 1x512
/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap_seqlen16/ \
    --shuffle-train \
    --models=rnn \
    --modality=feature_seq \
    --sampling=all \
    --seq-length=16 \
    --log-level=info \
    --batch-size=128 \
    --num-workers=4 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --learning-rate=1e-3 \
    --epoch=300 \
    --weight-initializer=xavier_normal \
    --num-rnn-layers=2 \
    --num-rnn-hidden=512 \
    --input-dim=512

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap_seqlen16/ \
    --shuffle-train \
    --models=lstm \
    --modality=feature_seq \
    --sampling=all \
    --seq-length=16 \
    --log-level=info \
    --batch-size=128 \
    --num-workers=4 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --learning-rate=1e-3 \
    --epoch=300 \
    --weight-initializer=xavier_normal \
    --num-rnn-layers=2 \
    --num-rnn-hidden=512 \
    --input-dim=512

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap_seqlen16/ \
    --shuffle-train \
    --models=gru \
    --modality=feature_seq \
    --sampling=all \
    --seq-length=16 \
    --log-level=info \
    --batch-size=128 \
    --num-workers=4 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --learning-rate=1e-3 \
    --epoch=300 \
    --weight-initializer=xavier_normal \
    --num-rnn-layers=2 \
    --num-rnn-hidden=512 \
    --input-dim=512

## Bidirectional 1x512
/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap_seqlen16/ \
    --shuffle-train \
    --models=rnn \
    --modality=feature_seq \
    --sampling=all \
    --seq-length=16 \
    --log-level=info \
    --batch-size=128 \
    --num-workers=4 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --learning-rate=1e-3 \
    --epoch=300 \
    --weight-initializer=xavier_normal \
    --num-rnn-layers=2 \
    --num-rnn-hidden=512 \
    --input-dim=512 \
    --bidirectional

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap_seqlen16/ \
    --shuffle-train \
    --models=lstm \
    --modality=feature_seq \
    --sampling=all \
    --seq-length=16 \
    --log-level=info \
    --batch-size=128 \
    --num-workers=4 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --learning-rate=1e-3 \
    --epoch=300 \
    --weight-initializer=xavier_normal \
    --num-rnn-layers=2 \
    --num-rnn-hidden=512 \
    --input-dim=512 \
    --bidirectional

/raid/users/oozdemir/anaconda3/envs/pytorch/bin/python3.6 /raid/users/oozdemir/code/untitled-slr-project/train.py \
    --dataset=bsign4 \
    --dataset-dir=/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap_seqlen16/ \
    --shuffle-train \
    --models=gru \
    --modality=feature_seq \
    --sampling=all \
    --seq-length=16 \
    --log-level=info \
    --batch-size=128 \
    --num-workers=4 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --learning-rate=1e-3 \
    --epoch=300 \
    --weight-initializer=xavier_normal \
    --num-rnn-layers=2 \
    --num-rnn-hidden=512 \
    --input-dim=512 \
    --bidirectional
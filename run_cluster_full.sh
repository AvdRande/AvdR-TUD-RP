#!/bin/sh
#
#SBATCH --job-name="avdr-RP-train-all"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G

srun python recommender/classify_all.py $1 $2
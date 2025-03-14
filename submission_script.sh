#!/bin/bash
#SBATCH -A $ALLOC
#SBATCH -p gengpu
#SBATCH --gres=gpu:100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem=64G
#SBATCH --job-name=wastewater_analysis
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source .venv/bin/activate
uv run main.py

#!/bin/bash
#SBATCH --job-name=mhw_composite
#SBATCH --output=/work5/stb/MHW-gfdl/logs/composite_%x_%j.out 
#SBATCH --error=/work5/stb/MHW-gfdl/logs/composite_%x_%j.err #the %x uses job names so logs won't overwrite each other
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=batch 

# Activate conda environment
source /app/conda/miniforge/etc/profile.d/conda.sh
conda activate /work/Shanice.Bailey/MHW-gfdl/envs/py311/

# cd to where the python scripts live
cd /work5/stb/MHW-gfdl/
python -u ${SCRIPT}
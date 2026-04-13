#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --output=logs/extract_latents_%j.out       # Output file (with job ID)
#SBATCH --error=logs/extract_latents_%j.err        # Error file (with job ID)
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=15G
#SBATCH --time=00-03:00
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --mem=15GB                        # Request 15 G of CPU memory
#SBATCH --account=def-bedelb              # Replace with your Compute Canada account

set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

# Set variables
epoch_to_use=200

# Run script to extract latent spaces
python /home/andim/BrainGrowthPrediction/scripts/preprocess/extract_latents_all.py \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete.csv \
  --aekl_ckpt   /home/andim/scratch/bgp/ae_all_output/autoencoder-ep-"$epoch_to_use".pth
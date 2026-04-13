#!/bin/bash
#SBATCH --job-name=autoencoder_brlp_training        # Job name
#SBATCH --output=logs/ae_set_%j.out       # Output file (with job ID)
#SBATCH --error=logs/ae_set_%j.err        # Error file (with job ID)
#SBATCH --gpus=h100:1                 # Full H100-80GB GPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=35G
#SBATCH --time=00-35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

# Set variables
base_dir=/home/andim/scratch/bgp
out_dir=ae_set_output
# Create an output and a cache directory
mkdir -p "$base_dir/$out_dir" 

# Run the training script
python /home/andim/BrainGrowthPrediction/scripts/train/train_autoencoder_set.py \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_with_split.csv \
  --n_epochs 600 \
  --early_stopping_patience 200 \
  --max_batch_size 3 \
  --batch_size 27 \
  --output_dir "$base_dir/$out_dir"

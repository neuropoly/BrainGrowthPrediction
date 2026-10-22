#!/bin/bash
#SBATCH --job-name=cvae_training        # Job name
#SBATCH --output=logs/cvae_%j.out       # Output file (with job ID)
#SBATCH --error=logs/cvae_%j.err        # Error file (with job ID)
#SBATCH --gpus=h100:1                 # Full H100-80GB GPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=35G
#SBATCH --time=00-55:00
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
out_dir=CVAE
timestamp=$(date +"%Y%m%d_%H%M%S")

full_out_dir=${base_dir}/${out_dir}/${timestamp}

mkdir -p "$full_out_dir"

echo "Output dir: $full_out_dir" 

# Run the training script
python /home/andim/BrainGrowthPrediction/scripts/train/train_CVAE.py \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_pairs.csv \
  --output_dir "$full_out_dir" \
  --project longitudinal_cvae \
  --run_name cvae_age0_dt_only_latent100 \
  --latent_dim 100 \
  --n_epochs 1000 \
  --batch_size 4 \
  --num_workers 8 \
  --lr 1e-5 \
  --resolution 1.3 1.3 1.3 \
  --target_shape 120 144 120 \
  --patience 100 \
  --forecast_base_image /home/andim/projects/def-bedelb/andim/brlp-data/sub-05_ses-02_brain.nii.gz \
  --forecast_base_age_m 17.477


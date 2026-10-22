#!/bin/bash
#SBATCH --job-name=cvae_eval        # Job name
#SBATCH --output=logs/cvae_%j.out       # Output file (with job ID)
#SBATCH --error=logs/cvae_%j.err        # Error file (with job ID)
#SBATCH --gpus=h100:1                 # Full H100-80GB GPU
#SBATCH --cpus-per-task=4  #prev=12
#SBATCH --mem=35G
#SBATCH --time=00-02:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load ants/2.6.2

source /home/andim/venvs/bgp/bin/activate
# export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

# Set variables
base_dir=/home/andim/scratch/bgp/CVAE
out_dir=eval_BCP_sub_model_20260421_131239
# Create an output 
mkdir -p "$base_dir/$out_dir" 

# Run the evaluation script
python /home/andim/BrainGrowthPrediction/scripts/evaluate/evaluate_CVAE.py \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_pairs.csv \
  --checkpoint /home/andim/scratch/bgp/CVAE/20260421_131239/best_cvae.pt \
  --output_dir "$base_dir/$out_dir" \
  --latent_dim 100 \
  --resolution 1.3 1.3 1.3 \
  --target_shape 120 144 120 \
  --subject_id sub-868724 \
  --direction forward \
  --deterministic \
  --make_gif \
  --compute_jacobian \
  --resume_after_triplets

# out_dir=eval_new
# --checkpoint /home/andim/scratch/bgp/CVAE/best_cvae.pt \
  # --subject_id sub-10008 \

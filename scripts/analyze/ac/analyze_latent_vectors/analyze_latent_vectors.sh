#!/bin/bash
#SBATCH --job-name=latent_pca_folds
#SBATCH --array=1-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00-00:30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/latent_pca_%A_%a.out
#SBATCH --error=logs/latent_pca_%A_%a.err
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

FOLD="$SLURM_ARRAY_TASK_ID"

CSV=/home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_with_cv_splits_latents_by_fold.csv
OUT_DIR=$HOME/scratch/bgp/latent_viz

python /home/andim/BrainGrowthPrediction/scripts/analyze/analyze_latent_vectors.py \
  --dataset_csv "$CSV" \
  --fold "$FOLD" \
  --ae_splits "train,val" \
  --out_dir "$OUT_DIR" \
  --diagnosis CC \
  --plot_sex \
  --plot_age 
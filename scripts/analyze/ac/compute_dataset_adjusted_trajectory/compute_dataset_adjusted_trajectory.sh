#!/bin/bash
#SBATCH --job-name=latent_report
#SBATCH --array=1-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/latent_report_%A_%a.out
#SBATCH --error=logs/latent_report_%A_%a.err

set -euo pipefail

mkdir -p logs

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

FOLD="${SLURM_ARRAY_TASK_ID}"

CSV="$HOME/scratch/bgp/latent_viz/fold_${FOLD}/latents_fold_${FOLD}.csv"
OUT_DIR="$HOME/scratch/bgp/latent_viz/fold_${FOLD}/trajectory_report"

if [[ ! -f "$CSV" ]]; then
  echo "[error] Missing latent CSV: $CSV"
  exit 1
fi

mkdir -p "$OUT_DIR"

python /home/andim/BrainGrowthPrediction/scripts/analyze/compute_dataset_adjusted_trajectory.py \
  --latent_csv "$CSV" \
  --out_dir "$OUT_DIR" \
  --age_step 1
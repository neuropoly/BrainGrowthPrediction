#!/bin/bash
#SBATCH --job-name=latent_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/latent_report_%j.out
#SBATCH --error=logs/latent_report_%j.err

set -euo pipefail

mkdir -p logs

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

OUT_DIR="$HOME/scratch/bgp/latent_viz/ALL_FOLDS_PLOTS"

mkdir -p "$OUT_DIR"

python /home/andim/BrainGrowthPrediction/scripts/analyze/plot_latent_reports.py \
  --root $HOME/scratch/bgp/latent_viz \
  --out_dir "$OUT_DIR"
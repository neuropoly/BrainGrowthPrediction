#!/bin/bash
#SBATCH --job-name=inspect_ext_recon
#SBATCH --array=1-5
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00-01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/inspect_ext_recon_%A_%a.out
#SBATCH --error=logs/inspect_ext_recon_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

FOLD="$SLURM_ARRAY_TASK_ID"

CSV=/home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_with_cv_splits.csv
CKPT_ROOT=/home/andim/scratch/bgp/ae_folds_output_NEW
OUT_ROOT=/home/andim/scratch/bgp/external_recon_inspection

python /home/andim/BrainGrowthPrediction/scripts/evaluate/inspect_external_recon_per_fold.py \
  --dataset_csv "$CSV" \
  --fold "$FOLD" \
  --ckpt_root "$CKPT_ROOT" \
  --n_scans 10 \
  --output_dir "$OUT_ROOT/fold_${FOLD}" 
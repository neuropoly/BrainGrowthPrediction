#!/bin/bash
#SBATCH --job-name=ae_folds    # Job name
#SBATCH --array=1-5                             # Folds 1,2,3,4,5                         
#SBATCH --gpus=h100:1                       
#SBATCH --cpus-per-task=12                       # 8 CPU cores
#SBATCH --mem=25G                               # 25 GB RAM per task
#SBATCH --time=00-45:00                       # DD-HH:MM:SS
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_folds_%A_%a.out         # %A = master jobID, %a = array index
#SBATCH --error=logs/ae_folds_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

# current fold (1..5)
FOLD=$SLURM_ARRAY_TASK_ID

# Base dirs
BASE=$HOME/scratch/bgp
OUT_ROOT=ae_folds_output_NEW
OUT_DIR=${BASE}/${OUT_ROOT}/fold_${FOLD}
mkdir -p "$OUT_DIR"

# Run the training script
python /home/andim/BrainGrowthPrediction/scripts/train/train_autoencoder_folds.py \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_with_cv_splits.csv \
  --n_epochs 600 \
  --max_batch_size 3 \
  --batch_size 27 \
  --fold "$FOLD" \
  --output_dir "$OUT_DIR"

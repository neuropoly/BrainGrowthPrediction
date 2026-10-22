#!/bin/bash
#SBATCH --job-name=ae_arms
#SBATCH --array=0-3                       # 0=baseline 1=aug 2=adv 3=aug_adv  (use 0-7 to add the pretrained variants)
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --time=3-00:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_arms_%A_%a.out
#SBATCH --error=logs/ae_arms_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
# W&B key: run `wandb login` once on the login node (stored in ~/.netrc) instead of putting it here.
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

ARMS=(baseline aug adv aug_adv)
IDX=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((IDX % 4))]}
SEED=0
 
# indices 4-7 = same arms, initialised from a pretrained autoencoder (set the path)
PRETRAINED_CKPT="/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth"
EXTRA=()
TAG="scratch"
if [ "$IDX" -ge 4 ]; then
  EXTRA=(--aekl_ckpt "$PRETRAINED_CKPT")
  TAG="pretrained"
fi
 
DATA=/home/andim/projects/def-bedelb/andim/brlp-data
BASE=$HOME/scratch/bgp
SPLITS_CSV=${DATA}/splits/datasets_splits.csv        # written once by make_splits.py (see below)
OUT_DIR=${BASE}/ae_midl/${ARM}_${TAG}_s${SEED}
CACHE_DIR=${BASE}/ae_cache                            # shared: only deterministic transforms are cached
mkdir -p "$OUT_DIR" "$CACHE_DIR" logs
 
python /home/andim/BrainGrowthPrediction/scripts/train/train_autoencoder.py \
  --dataset_csv "$SPLITS_CSV" \
  --arm "$ARM" \
  --seed "$SEED" \
  --n_epochs 600 \
  --patience 100 \
  --max_batch_size 3 \
  --batch_size 27 \
  --num_workers 10 \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUT_DIR" \
  --resume \
  "${EXTRA[@]}"
 
# ---------------------------------------------------------------------------
# Run ONCE before submitting (login node is fine, takes seconds):
#   python make_splits.py ${DATA}/datasets_all.csv --out-dir ${DATA}/splits
#
# If a job hits the time limit, resubmit only that index; it continues from state-last.pt
# in the same W&B run:   sbatch --array=2 run_ae_arms.sh
# A run that early-stopped writes OUT_DIR/DONE and exits immediately if resubmitted.
# ---------------------------------------------------------------------------
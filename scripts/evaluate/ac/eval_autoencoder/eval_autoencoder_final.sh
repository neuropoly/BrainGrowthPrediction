#!/bin/bash
#SBATCH --job-name=ae_eval_oldsplit
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=0-18:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_eval_oldsplit_%j.out
#SBATCH --error=logs/ae_eval_oldsplit_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

SPLITS=/home/andim/projects/def-bedelb/andim/brlp-data/splits_old
B=$HOME/scratch/bgp/ae_midl
PRE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
OUT=$HOME/scratch/bgp/eval_all8_oldsplit
CSV=$SPLITS/datasets_splits_eval.csv
SCRIPTS=/home/andim/BrainGrowthPrediction/scripts/evaluate

# 1) encode every scan once through all 9 models, then probes/statistics/plots
#    (resumes automatically from $OUT/encode_progress.pkl if the job is requeued)
python $SCRIPTS/eval_autoencoder.py \
  --dataset_csv "$CSV" \
  --out_dir "$OUT" \
  --cache_dir $HOME/scratch/bgp/ae_cache \
  --num_workers 6 \
  --runs baseline_scratch=$B/baseline_scratch_s0 aug_scratch=$B/aug_scratch_s0 \
         adv_scratch=$B/adv_scratch_s0 aug_adv_scratch=$B/aug_adv_scratch_s0 \
         baseline_pre=$B/baseline_pretrained_s0 aug_pre=$B/aug_pretrained_s0 \
         adv_pre=$B/adv_pretrained_s0 aug_adv_pre=$B/aug_adv_pretrained_s0 \
         frozen_pretrained=$PRE \
  --reference baseline_scratch \
  --probe_labels dataset scanner_key \
  --plot_color_by dataset manufacturer scanner_key \
  --save_recon_roles test external_tbi external_hc \
  --wandb --wandb_name eval_all8_oldsplit
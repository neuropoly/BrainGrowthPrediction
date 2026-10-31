#!/bin/bash
#SBATCH --job-name=ae_eval
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=0-15:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_eval_%j.out
#SBATCH --error=logs/ae_eval_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

B=$HOME/scratch/bgp/ae_midl
CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits/datasets_splits.csv
PRE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
OUT=$HOME/scratch/bgp/eval_all8

python /home/andim/BrainGrowthPrediction/scripts/evaluate/eval_autoencoder.py \
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
  --save_recon_roles test external_tbi \
  --wandb --wandb_name eval_all8

#!/bin/bash
#SBATCH --job-name=ae_eval_v2
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=0-18:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_eval_v2_%j.out
#SBATCH --error=logs/ae_eval_v2_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# v2 evaluation of all 8 arms (+ frozen pretrained) on the new splits:
#   * external groups: external_tbi (Daufin, Kaouenn, Koala) AND external_hc (MASiVar),
#     each with its own metrics (ssim_external_hc, age_mae_external_hc, ...)
#   * probes: dataset (site_*) and scanner_key (scanner_*), + age-only reference
#   * mTBI vs control-like (CC + OI + control) on the external TBI cohorts' latents
#   * latent plots by dataset, manufacturer and scanner_key
#   * reconstructions saved for test + both external groups (for SIAM Dice)
# Start after training:   sbatch --dependency=afterok:<training array job id> eval_all8_v2.sh
# Re-run only the CPU part (probes/plots/tables) later:   STAGE=analyze sbatch eval_all8_v2.sh
# -----------------------------------------------------------------------------
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

# ======================= EDIT THESE ==========================================
EVAL=/home/andim/BrainGrowthPrediction/scripts/evaluate/eval_autoencoder.py   # the NEW eval script
B=$HOME/scratch/bgp/ae_midl_v2                                                # v2 checkpoints
CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/datasets_splits.csv # the NEW splits
PRE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
OUT=$HOME/scratch/bgp/eval_all8_v2            # new folder: v1 results in eval_all8 stay untouched
# Eval items are only {image_uid, image_path}: the old cache is reused safely (no labels inside)
CACHE=$HOME/scratch/bgp/ae_cache_v2
STAGE=${STAGE:-all}                           # all | encode | analyze
CKPT=best-loss                                # fixed in advance (validation only)
N_BOOT=1000
GROUP_N_PERM=500                              # permutations for the mTBI probe (0 = skip the test)
WANDB_PROJECT=ae-midl-v2                      # same project as the v2 training runs
# =============================================================================

RUNS=(
  baseline_scratch=$B/baseline_scratch_s0   aug_scratch=$B/aug_scratch_s0
  adv_scratch=$B/adv_scratch_s0             aug_adv_scratch=$B/aug_adv_scratch_s0
  baseline_pre=$B/baseline_pretrained_s0    aug_pre=$B/aug_pretrained_s0
  adv_pre=$B/adv_pretrained_s0              aug_adv_pre=$B/aug_adv_pretrained_s0
  frozen_pretrained=$PRE
)

# ---- fail fast: every checkpoint must exist (and training should be finished) ----
missing=0
for r in "${RUNS[@]}"; do
  name=${r%%=*}; path=${r#*=}
  ck=$path; [[ $path == *.pth ]] || ck=$path/autoencoder-$CKPT.pth
  if [ ! -f "$ck" ]; then echo "[missing] $name: $ck"; missing=1; continue; fi
  if [[ $path != *.pth ]] && [ ! -f "$path/DONE" ]; then
    last=$(python -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['epoch'])" \
           "$path/state-last.pt" 2>/dev/null || echo "?")
    echo "[warn] $name has no DONE file (not early-stopped); last finished epoch: $last"
  fi
done
[ $missing -eq 0 ] || { echo "checkpoints missing -> stop"; exit 1; }
mkdir -p "$OUT" logs

python "$EVAL" \
  --dataset_csv "$CSV" \
  --out_dir "$OUT" \
  --cache_dir "$CACHE" \
  --num_workers 6 \
  --stage "$STAGE" \
  --ckpt "$CKPT" \
  --runs "${RUNS[@]}" \
  --reference baseline_scratch \
  --save_recon_roles test external_tbi external_hc \
  --metric_roles val test external_tbi external_hc \
  --probe_labels dataset scanner_key \
  --probe_min_count 8 \
  --plot_color_by dataset manufacturer scanner_key \
  --group_col group_raw \
  --group_case mTBI patient \
  --group_control CC OI control \
  --group_n_pcs 20 \
  --group_n_perm "$GROUP_N_PERM" \
  --n_boot "$N_BOOT" \
  --wandb --wandb_project "$WANDB_PROJECT" --wandb_name eval_all8_v2

echo "[done] $OUT: summary_table.csv | probe_reference.csv | group_probe.csv | latent_plots*.png | recon_manifest.csv"
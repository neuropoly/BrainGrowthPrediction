#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./sync_and_rename.sh --has-ses   (longitudinal datasets: require ses-)
#   ./sync_and_rename.sh --no-ses    (cross-sectional datasets: forbid ses-)
#   ./sync_and_rename.sh --auto      (default: allow either)

SES_MODE="auto"
case "${1:-}" in
  --has-ses) SES_MODE="required"; shift ;;
  --no-ses)  SES_MODE="forbid";   shift ;;
  --auto|"") SES_MODE="auto" ;;
  *) echo "Unknown arg: $1 (use --has-ses | --no-ses | --auto)"; exit 2 ;;
esac

###############################################################################
# User settings
###############################################################################
REMOTE="andim@fir.alliancecan.ca"
REMOTE_DIR="/home/andim/projects/def-bedelb/andim/brlp-data/rigid"  # destination root on CC

LOCAL_PARENT="/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs"

DATASETS=(
  "hc-new-england"
  # "hc-new-england"
)

LOCAL_SUBPATH="derivatives/brainprep"

REG_TYPE="rigid"  # or affine
PAT_BRAIN="*_space-ANTS80Years3Tbrainbiascorrected_desc-${REG_TYPE}-intnorm_T1w.nii.gz"
PAT_SEG="*_space-ANTS80Years3Tbrainbiascorrected_desc-${REG_TYPE}-synthseg_dseg.nii.gz"

# Upload behavior:
# 1 => upload under REMOTE_DIR/<dataset>/ to avoid name collisions
# 0 => upload all into REMOTE_DIR (risk of collisions if same sub IDs exist across datasets)
KEEP_DATASET_DIR=0

###############################################################################
# Main
###############################################################################
for ds in "${DATASETS[@]}"; do
  src="${LOCAL_PARENT}/${ds}/${LOCAL_SUBPATH}"
  [[ -d "$src" ]] || { echo "Skip (missing): $src"; continue; }

  if [[ "$KEEP_DATASET_DIR" -eq 1 ]]; then
    rdir="${REMOTE_DIR}/${ds}"
  else
    rdir="${REMOTE_DIR}"
  fi

  echo "=== $ds ==="
  echo "Source: $src"
  echo "Dest  : $REMOTE:$rdir"
  echo "SES_MODE=$SES_MODE  REG_TYPE=$REG_TYPE"

  cd "$src"

  # 1) Build one NUL-separated file list (brains + segs)
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' EXIT
  find . -type f \( -name "$PAT_BRAIN" -o -name "$PAT_SEG" \) -print0 > "$tmp"

  # 2) One rsync (single authentication)
  rsync -avP --files-from="$tmp" --from0 --no-relative ./ "$REMOTE:$rdir/"

  # 3) One remote rename pass (single authentication)
  ssh "$REMOTE" "bash -lc '
    set -euo pipefail
    shopt -s nullglob

    cd \"$rdir\" || { echo \"[remote] cannot cd to $rdir\" >&2; exit 1; }

    SES_MODE=\"$SES_MODE\"
    REG_TYPE=\"$REG_TYPE\"

    rename_one() {
      local f=\"\$1\" kind=\"\$2\"
      local base=\"\" dest=\"\"

      # Prefer stripping the long suffix at _space- if present; otherwise strip extension
      if [[ \"\$f\" == *\"_space-\"* ]]; then
        base=\"\${f%%_space-*}\"
      else
        base=\"\${f%.nii.gz}\"
        base=\"\${base%.nii}\"
      fi

      [[ \"\$base\" == sub-* ]] || return 0

      # Drop any run entity entirely
      base=\"\$(echo \"\$base\" | sed -E \"s/_run-[0-9]+//g\")\"

      # Enforce ses presence/absence if requested
      if [[ \"\$SES_MODE\" == required && \"\$base\" != *\"_ses-\"* ]]; then
        echo \"[rename] skip (missing ses): \$f\" >&2
        return 0
      fi
      if [[ \"\$SES_MODE\" == forbid && \"\$base\" == *\"_ses-\"* ]]; then
        echo \"[rename] skip (unexpected ses): \$f\" >&2
        return 0
      fi

      dest=\"\${base}_\${kind}.nii.gz\"
      mv -n -- \"\$f\" \"\$dest\"
    }

    # brains
    for f in sub-*_desc-\${REG_TYPE}-intnorm_T1w.nii.gz; do
      rename_one \"\$f\" brain
    done

    # segm
    for f in sub-*_desc-\${REG_TYPE}-synthseg_dseg.nii.gz; do
      rename_one \"\$f\" segm
    done
  '"

  rm -f "$tmp"
  trap - EXIT
done

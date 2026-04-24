#!/usr/bin/env bash
# Finetune LeWM per (task, split) from a pretrained object ckpt.
#
# Usage:
#   bash run_finetune.sh
#
# Dataset layout (produced by sdlwm/scripts/preprocess_data.sh):
#   $FINETUNE_ROOT/<task>/<N>_episodes/<ep>/{frames,actions_raw}.npy
#
# Pretrained ckpt must be a full pickled model from train.py's
# ModelObjectCallBack, e.g. `<run_dir>/lewm_epoch_<E>_object.ckpt`.
#
# Envs:
#   FINETUNE_FROM   required. Path to pretrained *_object.ckpt.
#   FINETUNE_ROOT   required. Root containing <task>/<N>_episodes/.
#   TASKS           space-sep list, default: the 4 reps from preprocess_data.sh.
#   SPLITS          space-sep list of N values, default: 1 3 10 50.
#   DRY_RUN=1       print commands without running.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export FINETUNE_FROM=~/.stable_worldmodel/slow/lewm_epoch_5_object.ckpt
export FINETUNE_ROOT=/cephfs/zhaorui/data/robotwin/dataset/robotwin_easy_finetune/

: "${FINETUNE_FROM:?set FINETUNE_FROM to a pretrained lewm_epoch_N_object.ckpt}"
: "${FINETUNE_ROOT:?set FINETUNE_ROOT to the per-task dataset root (e.g. robotwin_easy_finetune)}"

if [ ! -f "$FINETUNE_FROM" ]; then
    echo "FINETUNE_FROM not found: $FINETUNE_FROM"; exit 1
fi
if [ ! -d "$FINETUNE_ROOT" ]; then
    echo "FINETUNE_ROOT not found: $FINETUNE_ROOT"; exit 1
fi

DEFAULT_TASKS=(
    click_bell
    beat_block_hammer
    handover_block
    blocks_ranking_size
)
read -r -a TASKS <<< "${TASKS:-${DEFAULT_TASKS[@]}}"
read -r -a SPLITS <<< "${SPLITS:-1 3 10 50}"

# Per-split epoch budget: tiny splits need more passes because each epoch
# only has a handful of batches. Override via EPOCHS_<N> env var, or the
# fallback EPOCHS_DEFAULT for any split not listed.
EPOCHS_1="${EPOCHS_1:-90}"
EPOCHS_3="${EPOCHS_3:-60}"
EPOCHS_10="${EPOCHS_10:-30}"
EPOCHS_50="${EPOCHS_50:-30}"
EPOCHS_DEFAULT="${EPOCHS_DEFAULT:-30}"

epochs_for_split() {
    local n="$1"
    local var="EPOCHS_${n}"
    if [ -n "${!var:-}" ]; then
        echo "${!var}"
    else
        echo "$EPOCHS_DEFAULT"
    fi
}

export FINETUNE_FROM FINETUNE_ROOT

echo "[finetune] pretrained=$FINETUNE_FROM"
echo "[finetune] dataset_root=$FINETUNE_ROOT"
echo "[finetune] tasks=${TASKS[*]}"
echo "[finetune] splits=${SPLITS[*]}"

for TASK in "${TASKS[@]}"; do
    for N in "${SPLITS[@]}"; do
        SPLIT_DIR="$FINETUNE_ROOT/$TASK/${N}_episodes"
        if [ ! -d "$SPLIT_DIR" ]; then
            echo "  [skip] $TASK/${N}_episodes: $SPLIT_DIR not found"
            continue
        fi

        export FT_TASK="$TASK" FT_SPLIT="$N"
        SUBDIR="ft_${TASK}_${N}ep"
        EPOCHS="$(epochs_for_split "$N")"
        echo
        echo "================================================================"
        echo "[finetune] task=$TASK split=$N epochs=$EPOCHS  ->  subdir=$SUBDIR"
        echo "================================================================"

        CMD=(
            python train.py
            --config-name=lewm_finetune
            subdir="$SUBDIR"
            trainer.max_epochs="$EPOCHS"
        )
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY_RUN: ${CMD[*]}"
        else
            "${CMD[@]}"
        fi
    done
done

echo
echo "[finetune] done."

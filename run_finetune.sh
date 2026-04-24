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
        echo
        echo "================================================================"
        echo "[finetune] task=$TASK split=$N  ->  subdir=$SUBDIR"
        echo "================================================================"

        CMD=(
            python train.py
            --config-name=lewm_finetune
            subdir="$SUBDIR"
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

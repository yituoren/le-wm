#!/usr/bin/env bash
# Batch-compute action_stats.pt for every (task, split) the finetune
# produced. Mirrors the dir layout from run_finetune.sh:
#
#   $CKPT_ROOT/ft_<task>_<N>ep/action_stats.pt   <- output
#   $FINETUNE_ROOT/<task>/<N>_episodes/          <- input data
#
# Usage:
#   FINETUNE_ROOT=/cephfs/.../robotwin_easy_finetune \
#   CKPT_ROOT=$(python -c "import stable_worldmodel as swm; print(swm.data.utils.get_cache_dir())") \
#   bash compute_action_stats.sh
#
# Envs:
#   FINETUNE_ROOT   required. Root containing <task>/<N>_episodes/.
#   CKPT_ROOT       required. Root containing ft_<task>_<N>ep/ subdirs.
#                   (== swm cache dir, unless you moved the ckpts.)
#   TASKS           space-sep list, default 4 reps from preprocess_data.sh.
#   SPLITS          space-sep list, default: 1 3 10 50.
#   SUBDIR_TEMPLATE override the ft_<task>_<N>ep name. Use {task} and {N}
#                   as placeholders. Default: ft_{task}_{N}ep.
#   STD_EPS         std clamp (default 1e-6, matches policy).
#   FORCE=1         overwrite existing action_stats.pt instead of skipping.
#   DRY_RUN=1       print commands without running.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export FINETUNE_ROOT=/cephfs/zhaorui/data/robotwin/dataset/robotwin_easy_finetune/
export CKPT_ROOT=~/.stable_worldmodel

: "${FINETUNE_ROOT:?set FINETUNE_ROOT to the per-task dataset root}"
: "${CKPT_ROOT:?set CKPT_ROOT to the dir containing ft_<task>_<N>ep/ ckpt subdirs}"

if [ ! -d "$FINETUNE_ROOT" ]; then
    echo "FINETUNE_ROOT not found: $FINETUNE_ROOT"; exit 1
fi
if [ ! -d "$CKPT_ROOT" ]; then
    echo "CKPT_ROOT not found: $CKPT_ROOT"; exit 1
fi

DEFAULT_TASKS=(
    click_bell
    beat_block_hammer
    handover_block
    blocks_ranking_size
)
read -r -a TASKS <<< "${TASKS:-${DEFAULT_TASKS[@]}}"
read -r -a SPLITS <<< "${SPLITS:-1 3 10 50}"
if [ -z "${SUBDIR_TEMPLATE:-}" ]; then
    SUBDIR_TEMPLATE='ft_{task}_{N}ep'
fi
STD_EPS="${STD_EPS:-1e-6}"

resolve_subdir() {
    local task="$1" n="$2"
    printf '%s\n' "$SUBDIR_TEMPLATE" | sed -e "s/{task}/$task/g" -e "s/{N}/$n/g"
}

skipped=0
done_cnt=0
missing_data=0
missing_ckpt=0

for TASK in "${TASKS[@]}"; do
    for N in "${SPLITS[@]}"; do
        DATA_DIR="$FINETUNE_ROOT/$TASK/${N}_episodes"
        SUBDIR="$(resolve_subdir "$TASK" "$N")"
        CKPT_DIR="$CKPT_ROOT/$SUBDIR"
        OUT="$CKPT_DIR/action_stats.pt"

        if [ ! -d "$DATA_DIR" ]; then
            echo "  [skip] $TASK/${N}_episodes: no data at $DATA_DIR"
            missing_data=$((missing_data + 1))
            continue
        fi
        if [ ! -d "$CKPT_DIR" ]; then
            echo "  [skip] $SUBDIR: no ckpt dir at $CKPT_DIR"
            missing_ckpt=$((missing_ckpt + 1))
            continue
        fi
        if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "  [skip] $SUBDIR: $OUT already exists (FORCE=1 to overwrite)"
            skipped=$((skipped + 1))
            continue
        fi

        CMD=(
            python compute_action_stats.py
            --data-root "$DATA_DIR"
            --out "$OUT"
            --std-eps "$STD_EPS"
        )
        echo
        echo "[stats] $TASK/${N}ep  ->  $OUT"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY_RUN: ${CMD[*]}"
        else
            "${CMD[@]}"
            done_cnt=$((done_cnt + 1))
        fi
    done
done

echo
echo "================================================================"
echo "[stats] done=$done_cnt  skipped(existed)=$skipped  "\
"missing_data=$missing_data  missing_ckpt=$missing_ckpt"

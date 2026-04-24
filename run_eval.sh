#!/usr/bin/env bash
# Sweep eval_client.py across all (split, task) finetune ckpts.
#
# Expects finetune ckpts under $CKPT_ROOT/ft_<task>_<N>ep/*.ckpt (run_finetune.sh
# layout). Per split N, builds a temp symlink tree <tmp>/<task> -> the matching
# ckpt dir so eval_client's `--ckpt-dir <root> --ckpt-glob <pat>` convention
# (<root>/<task>/<pat>) works unchanged.
#
# Server must already be running at $HOST:$PORT -- this script only drives the
# client side. Each split spawns one eval_client run (which internally iterates
# all tasks), logs stdout to $LOG_DIR/split_<N>.log, then the script parses the
# per-task summary lines at the end and prints a split x task matrix.
#
# Envs:
#   GOALS_ROOT      required. --goals-root for eval_client.
#   CKPT_ROOT       required. Dir containing ft_<task>_<N>ep/ subdirs.
#                   Default: swm cache dir (guessed if unset).
#   TASKS           space-sep list; default = 4 reps.
#   SPLITS          space-sep list; default: 1 3 10 50.
#   CKPT_GLOB       picks which ckpt inside each ft_<task>_<N>ep/. Default:
#                   'lewm_ft_epoch_*_object.ckpt' (latest mtime wins).
#   LOG_DIR         where per-split client logs land. Default: ./eval_logs.
#   HOST / PORT     server address; default 127.0.0.1 / 9765.
#   DEVICE          --device; default cuda.
#   SUBDIR_TEMPLATE override ckpt subdir name; default ft_{task}_{N}ep.
#   EXTRA_ARGS      appended verbatim to each eval_client call (e.g. MPC knobs).
#   DRY_RUN=1       print commands without running.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export GOALS_ROOT=~/lwm/goals
export CKPT_ROOT=~/lwm/checkpoints/lewm/slow

: "${GOALS_ROOT:?set GOALS_ROOT to the goal bank root (--goals-root)}"

[ -d "$CKPT_ROOT" ] || { echo "CKPT_ROOT not a dir: $CKPT_ROOT"; exit 1; }
[ -d "$GOALS_ROOT" ] || { echo "GOALS_ROOT not a dir: $GOALS_ROOT"; exit 1; }

DEFAULT_TASKS=(
    click_bell
    beat_block_hammer
    handover_block
    blocks_ranking_size
)
read -r -a TASKS <<< "${TASKS:-${DEFAULT_TASKS[@]}}"
read -r -a SPLITS <<< "${SPLITS:-1 3 10 50}"
CKPT_GLOB="${CKPT_GLOB:-lewm_ft_epoch_*_object.ckpt}"
if [ -z "${SUBDIR_TEMPLATE:-}" ]; then
    SUBDIR_TEMPLATE='ft_{task}_{N}ep'
fi
LOG_DIR="${LOG_DIR:-$HERE/eval_logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9765}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$LOG_DIR"

read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

resolve_subdir() {
    local task="$1" n="$2"
    printf '%s\n' "$SUBDIR_TEMPLATE" | sed -e "s/{task}/$task/g" -e "s/{N}/$n/g"
}

run_split() {
    local N="$1"
    local tmp; tmp="$(mktemp -d)"
    local log="$LOG_DIR/split_${N}.log"
    trap 'rm -rf "$tmp"' RETURN

    local any_ckpt=0
    local any_stats=1
    local present_tasks=()
    for TASK in "${TASKS[@]}"; do
        local sub; sub="$(resolve_subdir "$TASK" "$N")"
        local src="$CKPT_ROOT/$sub"
        if [ ! -d "$src" ]; then
            continue
        fi
        # eval_client needs at least one file matching CKPT_GLOB inside src.
        # Test via bash glob to fail fast instead of getting a SystemExit later.
        shopt -s nullglob
        local matches=("$src"/$CKPT_GLOB)
        shopt -u nullglob
        if [ "${#matches[@]}" -eq 0 ]; then
            echo "  [skip] $sub: no ckpt matching '$CKPT_GLOB'"
            continue
        fi
        if [ ! -f "$src/action_stats.pt" ]; then
            echo "  [warn] $sub: action_stats.pt missing -- LeWMPolicy will refuse to load"
            any_stats=0
        fi
        ln -sfn "$src" "$tmp/$TASK"
        present_tasks+=("$TASK")
        any_ckpt=1
    done
    if [ "$any_ckpt" -eq 0 ]; then
        echo "[eval] split=$N: no ckpts found, skipping"
        return
    fi
    if [ "$any_stats" -eq 0 ]; then
        echo "[eval] split=$N: some action_stats.pt missing; run compute_action_stats.sh first"
    fi

    local task_args=()
    for TASK in "${present_tasks[@]}"; do
        task_args+=(--task "$TASK")
    done

    local CMD=(
        python eval_client.py
        --host "$HOST" --port "$PORT"
        --goals-root "$GOALS_ROOT"
        --ckpt-dir "$tmp"
        --ckpt-glob "$CKPT_GLOB"
        --device "$DEVICE"
        "${task_args[@]}"
    )
    if [ "${#EXTRA_ARGS_ARR[@]}" -gt 0 ]; then
        CMD+=("${EXTRA_ARGS_ARR[@]}")
    fi

    echo
    echo "================================================================"
    echo "[eval] split=$N tasks=${present_tasks[*]}"
    echo "       log -> $log"
    echo "================================================================"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: ${CMD[*]}"
    else
        # tee so user sees live output while also saving for the summary pass.
        "${CMD[@]}" 2>&1 | tee "$log"
    fi
}

for N in "${SPLITS[@]}"; do
    run_split "$N"
done

# ------------------------------------------------------------ summary
# eval_client prints per-task lines of the form `  <task_padded>  S/R = XX.X%`
# inside its "per-task summary" block. Grep them out and assemble a matrix.
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

echo
echo "================================================================"
echo "[eval] split x task summary (success / run = rate)"
echo "================================================================"

printf "%-28s" "task"
for N in "${SPLITS[@]}"; do
    printf " %12s" "${N}ep"
done
echo

for TASK in "${TASKS[@]}"; do
    printf "%-28s" "$TASK"
    for N in "${SPLITS[@]}"; do
        LOG="$LOG_DIR/split_${N}.log"
        if [ ! -f "$LOG" ]; then
            printf " %12s" "-"
            continue
        fi
        # Line looks like `  click_bell                               3/10 =  30.0%`
        LINE="$(awk -v t="$TASK" '
            $0 ~ ("^  " t "[[:space:]]+[0-9]+/[0-9]+[[:space:]]*=") { print; exit }
        ' "$LOG")"
        if [ -z "$LINE" ]; then
            printf " %12s" "-"
        else
            # Extract the S/R and rate.
            SR="$(echo "$LINE" | awk '{for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\/[0-9]+$/) {print $i; exit}}')"
            RATE="$(echo "$LINE" | awk '{for (i=1;i<=NF;i++) if ($i ~ /%$/) {print $i; exit}}')"
            printf " %12s" "${SR:-?}(${RATE:-?})"
        fi
    done
    echo
done

echo
echo "[eval] done. per-split logs in $LOG_DIR/"

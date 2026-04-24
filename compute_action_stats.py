"""Compute action mean/std from a preprocessed RoboTwin dataset dir.

Reproduces the stats that ``train.py`` computes on the fly via
``get_column_normalizer`` (see ``utils.py``) but writes them to disk so
``LeWMPolicy`` can denormalize its MPC output at inference time.

Input layout matches ``robotwin_dataset.RoboTwinDataset``:

    <data_root>/<episode_name>/
        actions_raw.npy   (T_ds, fs, A) float32
        meta.json         {has_actions_raw, action_dim, ds_length, ...}

Concatenates all episodes' chunks to shape ``(N, fs*A)`` and computes
per-column mean / std (unbiased, matching ``torch.std``'s default used by
``get_column_normalizer``). NaN rows are skipped, same as training.

Usage:
    python compute_action_stats.py \\
        --data-root $FINETUNE_ROOT/click_bell/50_episodes \\
        --out <ckpt_dir>/action_stats.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_chunks(data_root: Path) -> tuple[np.ndarray, int, int]:
    """Return concatenated (N, fs*A) action chunks, plus (fs, A)."""
    chunks: list[np.ndarray] = []
    fs_seen: int | None = None
    a_seen: int | None = None

    eps = sorted(p for p in data_root.iterdir()
                 if p.is_dir() and (p / "meta.json").exists())
    if not eps:
        raise FileNotFoundError(
            f"no episodes under {data_root} (need <ep>/meta.json + actions_raw.npy)"
        )

    for ep in eps:
        try:
            meta = json.loads((ep / "meta.json").read_text())
        except json.JSONDecodeError:
            continue
        if not meta.get("has_actions_raw", False):
            continue
        raw_path = ep / "actions_raw.npy"
        if not raw_path.exists():
            continue

        arr = np.load(raw_path, mmap_mode="r")  # (T_ds, fs, A)
        if arr.ndim != 3:
            raise ValueError(f"{raw_path}: expected 3D, got shape {arr.shape}")
        T_ds, fs, A = arr.shape
        if fs_seen is None:
            fs_seen, a_seen = fs, A
        elif (fs, A) != (fs_seen, a_seen):
            raise ValueError(
                f"inconsistent (fs, A) across episodes: {raw_path} has "
                f"({fs}, {A}) vs earlier ({fs_seen}, {a_seen})"
            )
        chunks.append(np.ascontiguousarray(arr).reshape(T_ds, -1))

    if not chunks or fs_seen is None or a_seen is None:
        raise RuntimeError(
            f"no usable episodes under {data_root} "
            f"(missing actions_raw.npy or has_actions_raw=false)"
        )

    return np.concatenate(chunks, axis=0), fs_seen, a_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="Preprocessed dataset dir (e.g. "
                        "$FINETUNE_ROOT/<task>/<N>_episodes).")
    p.add_argument("--out", required=True,
                   help="Output .pt path (convention: <ckpt_dir>/action_stats.pt).")
    p.add_argument("--std-eps", type=float, default=1e-6,
                   help="Clamp std_min to this to avoid div-by-zero on "
                        "constant joints. Matches the policy-side clamp.")
    args = p.parse_args()

    data_root = Path(args.data_root).expanduser()
    out_path = Path(args.out).expanduser()

    print(f"[stats] scanning {data_root}")
    flat, fs, A = _load_chunks(data_root)
    print(f"[stats] loaded {flat.shape[0]} rows  fs={fs}  A={A}  chunk_dim={fs*A}")

    data = torch.from_numpy(flat)
    mask = ~torch.isnan(data).any(dim=1)
    n_bad = int((~mask).sum().item())
    if n_bad:
        print(f"[stats] dropping {n_bad} rows with NaN")
    data = data[mask]
    if data.numel() == 0:
        raise RuntimeError("all rows contained NaN — cannot compute stats")

    mean = data.mean(dim=0, keepdim=True).float().clone()   # (1, fs*A)
    std = data.std(dim=0, keepdim=True).float().clone()     # (1, fs*A), unbiased
    std_clamped = std.clamp_min(float(args.std_eps))
    n_clamped = int((std < float(args.std_eps)).sum().item())
    if n_clamped:
        print(f"[stats] {n_clamped}/{std.numel()} std entries below {args.std_eps} "
              f"-> clamped (constant-valued dims)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": mean,                 # (1, fs*A)
        "std": std_clamped,           # (1, fs*A), eps-clamped
        "std_raw": std,               # (1, fs*A), pre-clamp (for reference)
        "frameskip": int(fs),
        "action_dim": int(A),
        "num_rows": int(data.shape[0]),
        "source": str(data_root),
    }, out_path)
    print(f"[stats] wrote {out_path}")
    print(f"        mean.min={mean.min().item():+.4f} "
          f"mean.max={mean.max().item():+.4f}")
    print(f"        std.min={std_clamped.min().item():.4f} "
          f"std.max={std_clamped.max().item():.4f}")


if __name__ == "__main__":
    main()

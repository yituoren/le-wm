"""RoboTwin dataset adapter for LeWM training (drop-in for swm.data.HDF5Dataset).

Reads the per-episode preprocessed layout produced by
``sdlwm/data/preprocess.py``:

    <data_root>/<episode_name>/
        frames.npy        (T_ds, 3, H, W) uint8           -- already resized
        actions_raw.npy   (T_ds, frame_skip, A) float32   -- LeWM-faithful raw
        meta.json

Returns dict items shaped to match the reference oracle/train.py contract:

    pixels:  (num_steps, 3, H, W) uint8 — the float cast, /255, and
             ImageNet normalization all happen GPU-side in ``lejepa_forward``
             (keeps CPU/IPC memory at 1 byte/pixel instead of 4).
    action:  (num_steps, frame_skip * A) float32, raw concat across the
             frame_skip window (matches ``effective_act_dim`` in train.py)

Implements the swm.data.HDF5Dataset surface that train.py touches:
``transform`` (settable), ``get_col_data``, ``get_dim``. Pixel
normalization is performed on-device in the training loop (not here), so
the robotwin.yaml turns off the Hydra-side image preprocessor to avoid
double-processing. Action normalization still flows through
``get_column_normalizer`` -> stats are computed over the concat'd vector
so the resulting ``mean/std`` of shape (1, frame_skip*A) broadcasts over
per-item ``(num_steps, frame_skip*A)``.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RoboTwinDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        num_steps: int,
        frameskip: int,
        keys_to_load: Iterable[str] = ("pixels", "action"),
        keys_to_cache: Iterable[str] = ("action",),
        transform=None,
        mmap: bool = True,
        require_raw_actions: bool = True,
    ):
        self.data_root = Path(data_root)
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.keys_to_load = list(keys_to_load)
        self.keys_to_cache = set(keys_to_cache)
        self.transform = transform
        self.mmap = bool(mmap)
        self.require_raw_actions = bool(require_raw_actions)

        if self.num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {self.num_steps}")

        self._episodes: list[tuple[Path, int]] = []     # (ep_dir, T_ds)
        self._cumcounts: list[int] = []
        self._action_dim: int | None = None

        # Tiny in-RAM cache for the columns named in keys_to_cache. Pixels are
        # never cached (mmap is fast enough; full uint8 frames would explode).
        self._cache_actions_raw: dict[Path, np.ndarray] = {}

        self._build_index()

        if not self._cumcounts or self._cumcounts[-1] == 0:
            raise RuntimeError(
                f"RoboTwinDataset: no usable windows under {self.data_root}. "
                f"Each episode needs ds_length >= {self.num_steps}"
                f"{', has_actions_raw=true' if self.require_raw_actions else ''}."
            )

    # ----- index ------------------------------------------------------

    def _build_index(self):
        total = 0
        for ep in sorted(
            p for p in self.data_root.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        ):
            try:
                meta = json.loads((ep / "meta.json").read_text())
            except json.JSONDecodeError:
                continue

            if "action" in self.keys_to_load:
                if not meta.get("has_actions_raw", False) or not (ep / "actions_raw.npy").exists():
                    continue

            T_ds = int(meta.get("ds_length", 0))
            if T_ds < self.num_steps:
                continue

            adim = meta.get("action_dim")
            if adim and self._action_dim is None:
                self._action_dim = int(adim)

            self._episodes.append((ep, T_ds))
            total += T_ds - self.num_steps + 1
            self._cumcounts.append(total)

    def __len__(self) -> int:
        return self._cumcounts[-1] if self._cumcounts else 0

    # ----- swm.data.HDF5Dataset surface ------------------------------

    def get_dim(self, col: str) -> int:
        """Return per-step dim for a non-pixel column (matches the reference)."""
        if col == "action":
            if self._action_dim is None:
                raise RuntimeError("action_dim unknown — index returned no episodes with actions.")
            return self._action_dim
        raise KeyError(f"unknown column: {col}")

    def get_col_data(self, col: str) -> np.ndarray:
        """Return the concat'd column stream used to compute global mean/std.

        For ``action``, returns shape ``(N_concat, frame_skip * A)`` so the
        downstream normalizer (mean shape ``(1, frame_skip*A)``) broadcasts
        cleanly over per-item ``(num_steps, frame_skip*A)``.
        """
        if col != "action":
            raise KeyError(f"get_col_data only supports 'action', got {col!r}")

        chunks = []
        for ep_dir, _T_ds in self._episodes:
            arr = self._load_actions_raw(ep_dir)  # (T_ds, fs, A)
            chunks.append(arr.reshape(arr.shape[0], -1))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    # ----- IO helpers -------------------------------------------------

    def _load_frames(self, ep_dir: Path) -> np.ndarray:
        return np.load(ep_dir / "frames.npy", mmap_mode="r" if self.mmap else None)

    def _load_actions_raw(self, ep_dir: Path) -> np.ndarray:
        if "action" in self.keys_to_cache and ep_dir in self._cache_actions_raw:
            return self._cache_actions_raw[ep_dir]
        arr = np.load(ep_dir / "actions_raw.npy", mmap_mode="r" if self.mmap else None)
        if "action" in self.keys_to_cache:
            self._cache_actions_raw[ep_dir] = np.asarray(arr)
        return arr

    # ----- sampling ---------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        ep_idx = bisect.bisect_right(self._cumcounts, idx)
        ep_dir, _T_ds = self._episodes[ep_idx]
        prev = self._cumcounts[ep_idx - 1] if ep_idx > 0 else 0
        t0 = idx - prev
        t1 = t0 + self.num_steps

        item: dict = {}

        if "pixels" in self.keys_to_load:
            frames = self._load_frames(ep_dir)[t0:t1]                                 # (T, 3, H, W) uint8
            # ascontiguousarray detaches from the read-only mmap so
            # torch.from_numpy doesn't warn. Float cast + /255 + ImageNet
            # norm are deferred to GPU in lejepa_forward.
            item["pixels"] = torch.from_numpy(np.ascontiguousarray(frames))           # (T, 3, H, W) uint8

        if "action" in self.keys_to_load:
            raw = self._load_actions_raw(ep_dir)[t0:t1]                                # (T, fs, A)
            raw = np.ascontiguousarray(raw).astype(np.float32).reshape(raw.shape[0], -1)
            item["action"] = torch.from_numpy(raw)

        if self.transform is not None:
            item = self.transform(item)
        return item

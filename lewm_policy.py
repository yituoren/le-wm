"""Inference-time MPC wrapper around the trained JEPA world model.

Loads a full-model pickle dumped by ``ModelObjectCallBack`` (see
``utils.py``) -- the JEPA instance is saved via ``torch.save(model, path)``
so the checkpoint is self-contained. The RoboTwin deploy shim at
``RoboTwin/policy/LeWM/`` imports this class and drives it with the
standard ``reset_obs / update_obs / get_action`` contract. The same class
is intentionally independent of the SD-LWM ``sdlwm.policies`` package --
sharing a BasePolicy would couple two models that only happen to share
the image-goal framing.

Planning is standard random-shooting MPC: sample K action sequences of
length ``planning_horizon``, roll each out through the JEPA predictor,
score by squared distance from the terminal emb to z_goal, and execute
the first action of the best candidate. ``JEPA.get_cost`` already does
the rollout + criterion; we just feed it a properly shaped info_dict.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def _preprocess_rgb_imagenet(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """HWC uint8 -> (3, S, S) float, ImageNet-normalised. Matches the
    training transform from ``utils.get_img_preprocessor``."""
    from PIL import Image

    img = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return torch.from_numpy((arr - mean) / std)


class LeWMPolicy:
    """Planning-based baseline driven by the reference JEPA world model."""

    def __init__(
        self,
        ckpt_path: str | Path,
        goal_image_path: str | Path | None = None,
        goal_bank_root: str | Path | None = None,
        task_name: str | None = None,
        camera_key: str = "head_camera",
        device: str | torch.device = "cuda",
        planning_horizon: int = 5,
        planning_iters: int = 100,
        history_size: int = 3,
        img_size: int = 224,
        action_dim: int = 14,
        frame_skip: int = 4,
        action_scale: float = 1.0,
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.camera_key = camera_key
        self.goal_bank_root = Path(goal_bank_root) if goal_bank_root else None
        self._task_name_hint = task_name
        self._seeds_cache: dict[Path, set[int]] = {}

        self.planning_horizon = int(planning_horizon)
        self.planning_iters = int(planning_iters)
        self.history_size = int(history_size)
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.frame_skip = int(frame_skip)
        self.chunk_dim = self.action_dim * self.frame_skip
        self.action_scale = float(action_scale)

        self.model = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.model.eval()
        self.model.requires_grad_(False)

        self.z_goal: torch.Tensor | None = None  # encoded goal (1, D)
        self.goal_pixels: torch.Tensor | None = None  # preprocessed (3, S, S)
        if goal_image_path is not None:
            self.load_goal_image(goal_image_path)

        self.frame_history: deque = deque(maxlen=self.history_size)
        self.action_history: deque = deque(maxlen=self.history_size)

    # --------------------------------------------------------- goal bank
    @torch.no_grad()
    def _set_goal_from_array(self, rgb: np.ndarray) -> None:
        pixels = _preprocess_rgb_imagenet(rgb, self.img_size).to(self.device)
        self.goal_pixels = pixels
        info = {"pixels": pixels.unsqueeze(0).unsqueeze(0)}  # (1, 1, 3, S, S)
        enc = self.model.encode(info)
        self.z_goal = enc["emb"][:, -1]  # (1, D)

    def load_goal_image(self, path: str | Path) -> None:
        from PIL import Image

        rgb = np.asarray(Image.open(str(path)).convert("RGB"))
        self._set_goal_from_array(rgb)

    def load_goal_for_seed(self, seed: int) -> bool:
        if self.goal_bank_root is None:
            return False
        task = self._task_name_hint or ""
        candidates = []
        if task:
            candidates.append(self.goal_bank_root / task / f"seed_{seed}.png")
        candidates.append(self.goal_bank_root / f"seed_{seed}.png")
        for p in candidates:
            if p.exists():
                self.load_goal_image(p)
                return True
        print(
            f"[LeWMPolicy] no goal frame for seed={seed} under "
            f"{self.goal_bank_root} (task={task!r}); z_goal unchanged."
        )
        return False

    def available_seeds(self, task_name: str | None = None) -> set[int]:
        if self.goal_bank_root is None:
            return set()
        task = task_name or self._task_name_hint or ""
        path = (self.goal_bank_root / task / "seeds.json") if task else (
            self.goal_bank_root / "seeds.json"
        )
        if path in self._seeds_cache:
            return self._seeds_cache[path]
        if not path.exists():
            self._seeds_cache[path] = set()
            return set()
        data = json.loads(path.read_text())
        seeds = set(int(s) for s in data.get("success_seeds", []))
        self._seeds_cache[path] = seeds
        return seeds

    # --------------------------------------------------------- history
    def _current_frame_tensor(self, obs: Mapping) -> torch.Tensor:
        rgb = obs["observation"][self.camera_key]["rgb"]
        return _preprocess_rgb_imagenet(np.asarray(rgb), self.img_size).to(self.device)

    def _pad_history(self, buf: deque, pad_value: torch.Tensor) -> list[torch.Tensor]:
        """Left-pad ``buf`` with ``pad_value`` to length history_size."""
        items = list(buf)
        if len(items) < self.history_size:
            items = [pad_value] * (self.history_size - len(items)) + items
        return items

    # ------------------------------------------------------- deploy API
    def reset_obs(self) -> None:
        self.frame_history.clear()
        self.action_history.clear()

    def update_obs(self, obs: Mapping) -> None:
        """Sample-level hook -- not used here (MPC re-plans each step)."""
        return

    @torch.no_grad()
    def get_action(self, obs: Mapping) -> list[np.ndarray]:
        if self.z_goal is None or self.goal_pixels is None:
            raise RuntimeError(
                "LeWMPolicy.get_action called before a goal was loaded. "
                "Use load_goal_image / load_goal_for_seed first."
            )

        current = self._current_frame_tensor(obs)
        self.frame_history.append(current)

        H = self.history_size
        T = H + self.planning_horizon
        K = self.planning_iters

        frames = self._pad_history(self.frame_history, current)
        pixels = torch.stack(frames, dim=0)  # (H, 3, S, S)
        pixels = pixels.unsqueeze(0).unsqueeze(0).expand(1, K, H, -1, -1, -1)

        goal = self.goal_pixels.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        goal = goal.expand(1, K, H, -1, -1, -1)

        if len(self.action_history) == 0:
            past_act = torch.zeros(1, K, H, self.chunk_dim, device=self.device)
        else:
            pad = torch.zeros(self.chunk_dim, device=self.device)
            hist = self._pad_history(self.action_history, pad)
            past_act = torch.stack(hist, dim=0).unsqueeze(0).unsqueeze(0)
            past_act = past_act.expand(1, K, H, -1)

        cand = torch.randn(
            1, K, self.planning_horizon, self.chunk_dim, device=self.device
        ) * self.action_scale

        action_sequence = torch.cat([past_act, cand], dim=2)  # (1, K, T, chunk)

        info_dict = {
            "pixels": pixels,
            "goal": goal,
            "action": torch.zeros(1, K, T, self.chunk_dim, device=self.device),
        }
        cost = self.model.get_cost(info_dict, action_sequence)  # (1, K)
        best = cost[0].argmin().item()
        best_chunk = cand[0, best, 0].detach()  # (chunk_dim,)

        self.action_history.append(best_chunk.clone())
        per_step = best_chunk.view(self.frame_skip, self.action_dim)
        return [per_step[i].cpu().numpy().astype(np.float32)
                for i in range(self.frame_skip)]

"""Inference-time MPC wrapper around the trained JEPA world model.

Loads a full-model pickle dumped by ``ModelObjectCallBack`` (see
``utils.py``) -- the JEPA instance is saved via ``torch.save(model, path)``
so the checkpoint is self-contained. The eval client at
``reference/oracle/eval_client.py`` runs in the LeWM training conda env
(``stable_worldmodel`` and friends already installed) and drives this
class with the standard ``reset_obs / update_obs / get_action`` contract.

Planning is delegated to the original LeWM solver, imported from the
upstream package: ``stable_worldmodel.solver.{CEMSolver, GradientSolver}``
(matching ``config/eval/solver/{cem,adam}.yaml``). The solver only sees
``planning_horizon`` future actions; ``_LeWMCostAdapter`` prepends the
``history_size``-long action history before delegating to
``JEPA.get_cost``, so the model still receives the past+future shape it
was trained on.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.solver import CEMSolver, GradientSolver


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


class _LeWMCostAdapter:
    """Bridges swm CEMSolver / GradientSolver and JEPA.get_cost.

    The solver feeds ``action_candidates`` of shape ``(B, S, T_horizon, A)``
    (future-only). JEPA was trained on a fixed-length window where the
    first ``history_size`` frames pair with their actually-observed past
    actions -- splitting them into ``act_0`` (past) and ``act_future``
    happens inside ``JEPA.rollout``. We hold ``past_act`` of shape
    ``(B, history_size, A)`` here, expand to match the sample batch and
    concatenate along the time axis before delegating.
    """

    def __init__(self, model, history_size: int):
        self.model = model
        self.history_size = int(history_size)
        self.past_act: torch.Tensor | None = None  # (B, history_size, A)

    def parameters(self):
        return self.model.parameters()

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        if self.past_act is None:
            raise RuntimeError("_LeWMCostAdapter.past_act must be set before get_cost")
        B, S, T_h, A = action_candidates.shape
        past = (
            self.past_act.to(action_candidates.device, action_candidates.dtype)
            .unsqueeze(1)
            .expand(B, S, -1, -1)
        )
        full = torch.cat([past, action_candidates], dim=2)  # (B, S, H + T_h, A)
        return self.model.get_cost(info_dict, full)


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
        solver: str = "cem",
        planning_horizon: int = 5,
        history_size: int = 3,
        img_size: int = 224,
        action_dim: int = 14,
        frame_skip: int = 4,
        # CEM knobs (mirror conf/eval/solver/cem.yaml in the swm repo).
        cem_num_samples: int = 300,
        cem_topk: int = 30,
        cem_var_scale: float = 1.0,
        cem_n_steps: int = 30,
        # GD knobs (mirror conf/eval/solver/adam.yaml).
        gd_n_steps: int = 30,
        gd_num_samples: int = 100,
        gd_var_scale: float = 1.0,
        gd_lr: float = 0.1,
        gd_action_noise: float = 0.0,
        gd_optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        seed: int = 1234,
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.camera_key = camera_key
        self.goal_bank_root = Path(goal_bank_root) if goal_bank_root else None
        self._task_name_hint = task_name
        self._seeds_cache: dict[Path, set[int]] = {}

        self.planning_horizon = int(planning_horizon)
        self.history_size = int(history_size)
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.frame_skip = int(frame_skip)
        self.chunk_dim = self.action_dim * self.frame_skip

        self.model = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.model.eval()
        self.model.requires_grad_(False)

        # Training applied (x - mean) / std as a dataset transform (see
        # train.py's get_column_normalizer), so the model's inputs AND the
        # candidates the solver picks from both live in normalized space.
        # Stats are NOT stored in the ckpt (closure-captured), so
        # compute_action_stats.py writes them next to the ckpt.
        stats_path = self._resolve_stats_path(ckpt_path)
        if stats_path is None:
            raise FileNotFoundError(
                f"action_stats.pt not found next to {Path(ckpt_path).resolve()}. "
                f"Run compute_action_stats.py on the finetune dataset first."
            )
        stats = torch.load(str(stats_path), map_location=self.device, weights_only=False)
        if int(stats["frameskip"]) != self.frame_skip:
            raise ValueError(
                f"action_stats frameskip={int(stats['frameskip'])} "
                f"!= policy frame_skip={self.frame_skip}"
            )
        if int(stats["action_dim"]) != self.action_dim:
            raise ValueError(
                f"action_stats action_dim={int(stats['action_dim'])} "
                f"!= policy action_dim={self.action_dim}"
            )
        self.act_mean = stats["mean"].to(self.device).view(-1)  # (fs*A,)
        self.act_std = stats["std"].to(self.device).view(-1)    # (fs*A,)
        print(f"[LeWMPolicy] loaded action stats from {stats_path}")

        # --- solver -------------------------------------------------------
        # The upstream solvers split horizon/action_dim out of __init__ and
        # into ``configure(action_space=, n_envs=, config=)`` (called by
        # ``swm.policy.WorldModelPolicy`` from a gym env's action_space). We
        # don't have a gym env at deploy time, so we hand-build a
        # ``Box(shape=(1, action_dim_per_step))`` -- ``configure`` reads
        # ``action_space.shape[1:]`` for the per-step dim and multiplies by
        # ``config.action_block`` to recover ``chunk_dim``.
        self._cost_adapter = _LeWMCostAdapter(self.model, self.history_size)
        action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, self.action_dim), dtype=np.float32,
        )
        plan_config = swm.PlanConfig(
            horizon=self.planning_horizon,
            receding_horizon=1,           # we re-plan every step; not used here
            history_len=self.history_size,
            action_block=self.frame_skip,
        )

        solver = solver.lower()
        if solver == "cem":
            self._solver = CEMSolver(
                model=self._cost_adapter,
                batch_size=1,
                num_samples=int(cem_num_samples),
                var_scale=float(cem_var_scale),
                n_steps=int(cem_n_steps),
                topk=int(cem_topk),
                device=self.device,
                seed=int(seed),
            )
        elif solver in ("gd", "adam", "adamw"):
            self._solver = GradientSolver(
                model=self._cost_adapter,
                n_steps=int(gd_n_steps),
                batch_size=1,
                var_scale=float(gd_var_scale),
                num_samples=int(gd_num_samples),
                action_noise=float(gd_action_noise),
                device=self.device,
                seed=int(seed),
                optimizer_cls=gd_optimizer_cls,
                optimizer_kwargs={"lr": float(gd_lr)},
            )
        else:
            raise ValueError(f"unknown solver={solver!r}; expected 'cem' or 'gd'")
        self._solver.configure(action_space=action_space, n_envs=1, config=plan_config)
        self.solver = solver

        self.z_goal: torch.Tensor | None = None  # encoded goal (1, D)
        self.goal_pixels: torch.Tensor | None = None  # preprocessed (3, S, S)
        if goal_image_path is not None:
            self.load_goal_image(goal_image_path)

        self.frame_history: deque = deque(maxlen=self.history_size)
        self.action_history: deque = deque(maxlen=self.history_size)

    # ------------------------------------------------------- stats I/O
    @staticmethod
    def _resolve_stats_path(ckpt_path: str | Path) -> Path | None:
        ckpt = Path(ckpt_path).expanduser().resolve()
        cand = ckpt.parent / "action_stats.pt"
        return cand if cand.exists() else None

    # --------------------------------------------------------- goal bank
    @torch.no_grad()
    def _set_goal_from_array(self, rgb: np.ndarray) -> None:
        pixels = _preprocess_rgb_imagenet(rgb, self.img_size).to(self.device)
        self.goal_pixels = pixels
        info = {"pixels": pixels.unsqueeze(0).unsqueeze(0)}
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
        items = list(buf)
        if len(items) < self.history_size:
            items = [pad_value] * (self.history_size - len(items)) + items
        return items

    # ------------------------------------------------------- deploy API
    def reset_obs(self) -> None:
        self.frame_history.clear()
        self.action_history.clear()

    def update_obs(self, obs: Mapping) -> None:
        return

    def get_action(self, obs: Mapping) -> list[np.ndarray]:
        if self.z_goal is None or self.goal_pixels is None:
            raise RuntimeError(
                "LeWMPolicy.get_action called before a goal was loaded. "
                "Use load_goal_image / load_goal_for_seed first."
            )

        current = self._current_frame_tensor(obs)
        self.frame_history.append(current)

        H = self.history_size

        frames = self._pad_history(self.frame_history, current)
        pixels_hist = torch.stack(frames, dim=0)                    # (H, 3, S, S)

        # info_dict shape contract: leading dim is total_envs (=1 online),
        # the solver inserts the num_samples dim and expands tensors.
        pixels = pixels_hist.unsqueeze(0)                           # (1, H, 3, S, S)
        goal = self.goal_pixels.unsqueeze(0).unsqueeze(0).expand(1, H, -1, -1, -1)

        if len(self.action_history) == 0:
            past_act = torch.zeros(1, H, self.chunk_dim, device=self.device)
        else:
            pad = torch.zeros(self.chunk_dim, device=self.device)
            hist = self._pad_history(self.action_history, pad)
            past_act = torch.stack(hist, dim=0).unsqueeze(0)        # (1, H, chunk)

        # ``action`` is a placeholder: ``JEPA.get_cost`` does ``goal.pop("action")``
        # before encoding the goal pixels, so the key must exist. Its values are
        # never read -- ``rollout`` overwrites ``info["action"]`` with the past
        # split of ``action_candidates``.
        T_total = H + self.planning_horizon
        info_dict = {
            "pixels": pixels,                                       # (1, H, 3, S, S)
            "goal": goal,                                           # (1, H, 3, S, S)
            "action": torch.zeros(1, T_total, self.chunk_dim, device=self.device),
        }

        self._cost_adapter.past_act = past_act
        try:
            outputs = self._solver(info_dict)
        finally:
            self._cost_adapter.past_act = None

        # CEM returns mean (B, T_horizon, A); GD returns the best sample.
        best_chunk_seq = outputs["actions"].to(self.device)         # (1, T_horizon, chunk)
        best_chunk = best_chunk_seq[0, 0].detach()                  # (chunk_dim,)

        # History stays in normalized space -- the model's action_encoder
        # expects that distribution. Env receives denormalized qpos.
        self.action_history.append(best_chunk.clone())
        best_chunk_raw = best_chunk * self.act_std + self.act_mean
        per_step = best_chunk_raw.view(self.frame_skip, self.action_dim)
        return [per_step[i].cpu().numpy().astype(np.float32)
                for i in range(self.frame_skip)]

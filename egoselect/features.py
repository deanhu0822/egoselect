"""Per-episode visual, motion, and quality features.

RGB is sampled sparsely (default 8 uniform frames) and encoded with a frozen
pretrained DINOv2. Full-episode motion is read from on-disk ee_pose arrays
without decoding every JPEG. SQL labels are not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simplejpeg
import torch
from PIL import Image

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode

from egoselect.dataset import DEFAULT_CACHE_DIR, local_episode_path

N_RGB_FRAMES = 8
DINO_MODEL_ID = "facebook/dinov2-small"
STATIONARY_SPEED_MPS = 0.02
DIRECTION_CHANGE_RAD = float(np.deg2rad(30.0))
INVALID_POSE_ABS = 1e6

MOTION_COLUMNS = (
    "path_length",
    "displacement",
    "mean_speed",
    "speed_std",
    "max_speed",
    "mean_abs_accel",
    "direction_change_ratio",
    "stationary_ratio",
    "duration_s",
    "n_frames",
)

QUALITY_COLUMNS = (
    "quality_score",
    "quality_valid_frame_ratio",
    "quality_trajectory_completeness",
    "quality_finite_pose_ratio",
    "quality_nonstationary",
    "quality_temporal_validity",
)


def sample_frame_indices(n_frames: int, n_samples: int = N_RGB_FRAMES) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(0, dtype=int)
    n = min(int(n_samples), int(n_frames))
    idx = np.linspace(0, n_frames - 1, n)
    return np.unique(np.rint(idx).astype(int))


def _as_jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else bytes(value)
    if isinstance(value, np.void):
        value = bytes(value.item())
    if not isinstance(value, (bytes, bytearray)):
        value = bytes(value)
    return bytes(value)


def decode_front_frame(episode: ZarrEpisode, index: int) -> np.ndarray | None:
    try:
        raw = episode.read({"images.front_1": (index, None)})["images.front_1"]
        jpeg = _as_jpeg_bytes(raw)
        return simplejpeg.decode_jpeg(jpeg, colorspace="RGB")
    except Exception:
        return None


def read_ee_pose(episode: ZarrEpisode, side: str) -> np.ndarray | None:
    key = f"{side}.obs_ee_pose"
    if key not in episode._collect_keys():
        return None
    n = int(episode.metadata.get("total_frames") or 0)
    arr = np.asarray(episode.read({key: (0, n)})[key], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _xyz_trajectory(pose: np.ndarray | None) -> np.ndarray:
    if pose is None or pose.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xyz = np.asarray(pose[:, :3], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    finite &= np.max(np.abs(xyz), axis=1) < INVALID_POSE_ABS
    return xyz[finite]


def _arm_motion(xyz: np.ndarray, dt: float) -> dict[str, float]:
    empty = {
        "path_length": 0.0,
        "displacement": 0.0,
        "speeds": np.zeros(0, dtype=np.float64),
        "n_valid": int(len(xyz)),
    }
    if len(xyz) < 2:
        empty["displacement"] = 0.0
        return empty
    deltas = np.diff(xyz, axis=0)
    step = np.linalg.norm(deltas, axis=1)
    speeds = step / max(dt, 1e-9)
    return {
        "path_length": float(step.sum()),
        "displacement": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "speeds": speeds,
        "deltas": deltas,
        "n_valid": int(len(xyz)),
    }


def motion_features(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    n_frames: int,
    fps: float | None,
) -> dict[str, float]:
    dt = 1.0 / fps if fps and fps > 0 else 1.0
    duration_s = float(n_frames / fps) if fps and fps > 0 else float(n_frames)
    left_xyz = _xyz_trajectory(left)
    right_xyz = _xyz_trajectory(right)
    left_m = _arm_motion(left_xyz, dt)
    right_m = _arm_motion(right_xyz, dt)
    speeds = np.concatenate([left_m["speeds"], right_m["speeds"]])
    if speeds.size == 0:
        mean_speed = std_speed = max_speed = accel = 0.0
        dir_ratio = 0.0
        stationary = 1.0
    else:
        mean_speed = float(speeds.mean())
        std_speed = float(speeds.std())
        max_speed = float(speeds.max())
        accel = float(np.mean(np.abs(np.diff(speeds)) / max(dt, 1e-9))) if speeds.size > 1 else 0.0
        dir_flags = []
        for arm in (left_m, right_m):
            deltas = arm.get("deltas")
            arm_speeds = arm["speeds"]
            if deltas is None or len(deltas) < 2:
                continue
            a, b = deltas[:-1], deltas[1:]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            ok = (na > 1e-8) & (nb > 1e-8) & (arm_speeds[1:] > STATIONARY_SPEED_MPS)
            cos = np.ones(len(a))
            cos[ok] = np.clip((a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok]), -1.0, 1.0)
            ang = np.arccos(cos)
            dir_flags.append((ang > DIRECTION_CHANGE_RAD) & ok)
        dir_ratio = float(np.concatenate(dir_flags).mean()) if dir_flags else 0.0
        # Idle if both arms are slow on overlapping steps; fall back to pooled speeds.
        stationary = float((speeds < STATIONARY_SPEED_MPS).mean())
    return {
        "path_length": float(left_m["path_length"] + right_m["path_length"]),
        "displacement": float(
            np.mean([left_m["displacement"], right_m["displacement"]])
        ),
        "mean_speed": mean_speed,
        "speed_std": std_speed,
        "max_speed": max_speed,
        "mean_abs_accel": accel,
        "direction_change_ratio": dir_ratio,
        "stationary_ratio": stationary,
        "duration_s": duration_s,
        "n_frames": float(n_frames),
        "n_valid_left": float(left_m["n_valid"]),
        "n_valid_right": float(right_m["n_valid"]),
        "fps": float(fps) if fps and fps > 0 else 0.0,
    }


def quality_features(
    *,
    n_frames: int,
    n_sampled: int,
    n_decoded: int,
    motion: dict[str, float],
) -> dict[str, float]:
    valid_frame = float(n_decoded / n_sampled) if n_sampled else 0.0
    expected = max(float(n_frames), 1.0)
    completeness = float(
        min(motion["n_valid_left"], motion["n_valid_right"]) / expected
    )
    finite_ratio = float(
        (motion["n_valid_left"] + motion["n_valid_right"]) / (2.0 * expected)
    )
    nonstationary = float(1.0 - np.clip(motion["stationary_ratio"], 0.0, 1.0))
    temporal = 1.0 if (n_frames > 1 and motion["fps"] > 0) else 0.0
    score = (
        0.25 * valid_frame
        + 0.25 * np.clip(completeness, 0.0, 1.0)
        + 0.20 * np.clip(finite_ratio, 0.0, 1.0)
        + 0.15 * nonstationary
        + 0.15 * temporal
    )
    return {
        "quality_score": float(np.clip(score, 0.0, 1.0)),
        "quality_valid_frame_ratio": valid_frame,
        "quality_trajectory_completeness": float(np.clip(completeness, 0.0, 1.0)),
        "quality_finite_pose_ratio": float(np.clip(finite_ratio, 0.0, 1.0)),
        "quality_nonstationary": nonstationary,
        "quality_temporal_validity": temporal,
    }


@dataclass
class DinoEncoder:
    model_id: str = DINO_MODEL_ID
    device: str | None = None

    def __post_init__(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed_frames(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        if not frames_rgb:
            return np.zeros((0, self.dim), dtype=np.float32)
        images = [Image.fromarray(f) for f in frames_rgb]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0]
        return cls.detach().cpu().numpy().astype(np.float32)


def mean_pool_normalize(frame_emb: np.ndarray, dim: int) -> np.ndarray:
    if frame_emb.size == 0:
        return np.zeros(dim, dtype=np.float32)
    vec = frame_emb.mean(axis=0).astype(np.float32)
    nrm = float(np.linalg.norm(vec))
    if nrm > 0:
        vec = vec / nrm
    return vec


def extract_episode_features(
    episode_hash: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    encoder: DinoEncoder,
    n_rgb: int = N_RGB_FRAMES,
    visual_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    path = local_episode_path(episode_hash, cache_dir)
    episode = ZarrEpisode(path)
    n_frames = int(episode.metadata.get("total_frames") or 0)
    fps_raw = episode.metadata.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else None
    except (TypeError, ValueError):
        fps = None
    if fps is not None and fps <= 0:
        fps = None

    indices = sample_frame_indices(n_frames, n_rgb)
    frames: list[np.ndarray] = []
    frame_report: list[dict[str, Any]] = []
    for idx in indices:
        rgb = decode_front_frame(episode, int(idx))
        ok = rgb is not None
        if ok:
            frames.append(rgb)
        frame_report.append(
            {
                "index": int(idx),
                "ok": ok,
                "shape": list(rgb.shape) if ok else None,
                "mean": float(rgb.mean()) if ok else None,
            }
        )

    if visual_cache is not None and episode_hash in visual_cache:
        vis = np.asarray(visual_cache[episode_hash], dtype=np.float32)
    else:
        frame_emb = encoder.embed_frames(frames)
        vis = mean_pool_normalize(frame_emb, encoder.dim)

    left = read_ee_pose(episode, "left")
    right = read_ee_pose(episode, "right")
    motion = motion_features(left, right, n_frames=n_frames, fps=fps)
    quality = quality_features(
        n_frames=n_frames,
        n_sampled=len(indices),
        n_decoded=len(frames),
        motion=motion,
    )
    row: dict[str, Any] = {
        "episode_hash": episode_hash,
        "visual_model": encoder.model_id,
        "visual_dim": int(vis.shape[0]),
        "n_rgb_sampled": int(len(indices)),
        "n_rgb_decoded": int(len(frames)),
        "sampled_indices": [int(i) for i in indices],
        "vis_emb": vis.astype(np.float32),
        **{k: float(motion[k]) for k in MOTION_COLUMNS},
        **quality,
        "fps": float(motion["fps"]),
    }
    row["_frame_report"] = frame_report
    return row


def load_visual_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: dict[str, np.ndarray] = {}
    for _, rec in df.iterrows():
        out[str(rec["episode_hash"])] = np.asarray(rec["vis_emb"], dtype=np.float32)
    return out


def save_visual_cache(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pd.DataFrame(
        [
            {
                "episode_hash": r["episode_hash"],
                "visual_model": r["visual_model"],
                "visual_dim": r["visual_dim"],
                "n_rgb_sampled": r["n_rgb_sampled"],
                "n_rgb_decoded": r["n_rgb_decoded"],
                "vis_emb": np.asarray(r["vis_emb"], dtype=np.float32),
            }
            for r in rows
        ]
    )
    payload.to_parquet(path, index=False)
    return path

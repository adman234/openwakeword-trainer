"""The openWakeWord classifier and its training loop.

The network matches openWakeWord's "dnn" model (flatten, linear, layer norm, ReLU blocks, sigmoid) so the
exported files work anywhere openWakeWord models do. Training follows openWakeWord's auto_train: hard
example mining, a negative class weight that ramps up over each phase, three phases with decreasing
learning rate, then averaging the best checkpoints.
"""
import copy
import math
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import FEATURE_DIM, FRAME_SECONDS


class FCNBlock(nn.Module):
    def __init__(self, layer_dim):
        super().__init__()
        self.fcn_layer = nn.Linear(layer_dim, layer_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(layer_dim)

    def forward(self, x):
        return self.relu(self.layer_norm(self.fcn_layer(x)))


class WakeNet(nn.Module):
    def __init__(self, n_frames: int = 16, layer_dim: int = 32, n_blocks: int = 1):
        super().__init__()
        self.n_frames = n_frames
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(n_frames * FEATURE_DIM, layer_dim)
        self.relu1 = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(layer_dim)
        self.blocks = nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
        self.last_layer = nn.Linear(layer_dim, 1)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        return self.last_act(self.last_layer(x))


def lr_at(step: int, warmup: int, hold: int, total: int, target: float) -> float:
    if step < warmup:
        return target * step / max(1, warmup)
    if step < warmup + hold:
        return target
    return 0.5 * target * (1 + math.cos(math.pi * (step - warmup - hold) / max(1, total - warmup - hold)))


def count_activations(scores: np.ndarray, threshold: float, refractory: int = 12) -> int:
    """Count separate activations, treating hits within ~1 s of each other as one."""
    idx = np.flatnonzero(scores >= threshold)
    if idx.size == 0:
        return 0
    return 1 + int(np.sum(np.diff(idx) > refractory))


class FeaturePool:
    def __init__(self, arr, device):
        arr = np.asarray(arr)
        self.n = len(arr)
        self.t = torch.from_numpy(arr.astype(np.float16)).to(device) if self.n else None
        self.device = device

    def sample(self, n: int) -> torch.Tensor | None:
        if n <= 0 or not self.n:
            return None
        idx = torch.randint(0, self.n, (n,), device=self.device)
        return self.t[idx].float()


class AcavStream:
    """Generic negative windows from the 2,000 hour ACAV100M feature file.

    Blocks are read sequentially from a moving cursor on a background thread (fast even on spinning
    disks), swapped into a rolling pool, and every batch samples that pool at random.
    """

    def __init__(self, path, n_frames: int, device, block_rows: int = 1024, pool_blocks: int = 48, seed: int = 0):
        self.data = np.load(path, mmap_mode="r")
        self.n_frames = n_frames
        self.device = device
        self.block_rows = block_rows
        self.rng = np.random.default_rng(seed)
        self.cursor = int(self.rng.integers(0, max(1, len(self.data) - block_rows)))
        self.q = queue.Queue(maxsize=8)
        self.stopped = False
        blocks = [self._read_block() for _ in range(pool_blocks)]
        self.block_windows = len(blocks[0])
        self.pool_blocks = pool_blocks
        self.pool = torch.from_numpy(np.concatenate(blocks)).to(device)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _read_block(self) -> np.ndarray:
        if self.cursor + self.block_rows > len(self.data):
            self.cursor = 0
        blk = np.array(self.data[self.cursor:self.cursor + self.block_rows], dtype=np.float16)
        self.cursor += self.block_rows
        if self.n_frames != blk.shape[1]:
            # Rows are consecutive 16-frame windows, so longer models get re-chunked continuous frames
            flat = blk.reshape(-1, FEATURE_DIM)
            m = len(flat) // self.n_frames
            blk = flat[: m * self.n_frames].reshape(m, self.n_frames, FEATURE_DIM)
        return blk

    def _reader(self):
        while not self.stopped:
            blk = self._read_block()
            while not self.stopped:
                try:
                    self.q.put(blk, timeout=1)
                    break
                except queue.Full:
                    continue

    def sample(self, n: int) -> torch.Tensor:
        try:
            blk = self.q.get_nowait()
            slot = int(self.rng.integers(0, self.pool_blocks))
            self.pool[slot * self.block_windows:(slot + 1) * self.block_windows] = torch.from_numpy(blk).to(self.device)
        except queue.Empty:
            pass
        idx = torch.randint(0, self.pool.shape[0], (n,), device=self.device)
        return self.pool[idx].float()

    def close(self):
        self.stopped = True


@dataclass
class TrainingData:
    n_frames: int
    pos_tts: np.ndarray
    pos_real: np.ndarray
    neg_tts: np.ndarray
    neg_real: np.ndarray
    neg_background: np.ndarray
    val_pos_tts: np.ndarray
    val_pos_real: np.ndarray
    val_neg: np.ndarray
    fp_val: np.ndarray  # (T, 96) continuous features of ~11 hours of speech, music and noise
    acav_path: str


class Validator:
    def __init__(self, data: TrainingData, device):
        self.device = device
        self.n_frames = data.n_frames
        self.pos_tts = FeaturePool(data.val_pos_tts, device)
        self.pos_real = FeaturePool(data.val_pos_real, device)
        self.neg = FeaturePool(data.val_neg, device)
        self.fp = torch.from_numpy(np.asarray(data.fp_val, dtype=np.float32)).to(device)
        self.hours = len(data.fp_val) * FRAME_SECONDS / 3600

    @torch.no_grad()
    def _scores(self, model, pool: FeaturePool, chunk: int = 8192) -> np.ndarray:
        if not pool.n:
            return np.zeros(0, dtype=np.float32)
        return torch.cat([model(pool.t[i:i + chunk].float()).squeeze(1) for i in range(0, pool.n, chunk)]).cpu().numpy()

    @torch.no_grad()
    def _fp_scores(self, model, chunk: int = 16384) -> np.ndarray:
        windows = self.fp.unfold(0, self.n_frames, 1).transpose(1, 2)  # (T - F + 1, F, 96) view
        return torch.cat([model(windows[i:i + chunk]).squeeze(1) for i in range(0, len(windows), chunk)]).cpu().numpy()

    def raw_scores(self, model) -> dict:
        was_training = model.training
        model.eval()
        out = {"pos_tts": self._scores(model, self.pos_tts), "pos_real": self._scores(model, self.pos_real),
               "neg": self._scores(model, self.neg), "fp": self._fp_scores(model)}
        model.train(was_training)
        return out

    def metrics(self, scores: dict, threshold: float = 0.5) -> dict:
        recall_tts = float(np.mean(scores["pos_tts"] >= threshold)) if len(scores["pos_tts"]) else 0.0
        recall_real = float(np.mean(scores["pos_real"] >= threshold)) if len(scores["pos_real"]) else None
        return {
            "threshold": threshold,
            "recall_tts": recall_tts,
            "recall_real": recall_real,
            "recall": recall_tts if recall_real is None else (recall_tts + recall_real) / 2,
            "similar_phrase_accept": float(np.mean(scores["neg"] >= threshold)) if len(scores["neg"]) else 0.0,
            "fp_per_hour": count_activations(scores["fp"], threshold) / max(self.hours, 1e-6),
        }

    def evaluate(self, model, threshold: float = 0.5) -> dict:
        return self.metrics(self.raw_scores(model), threshold)

    def threshold_table(self, model, target_fp: float) -> tuple[list[dict], float]:
        scores = self.raw_scores(model)
        rows = [self.metrics(scores, round(float(t), 2)) for t in np.arange(0.1, 0.951, 0.05)]
        ok = [r for r in rows if r["fp_per_hour"] <= target_fp and r["threshold"] >= 0.3]
        recommended = ok[0]["threshold"] if ok else min(rows, key=lambda r: (r["fp_per_hour"], -r["threshold"]))["threshold"]
        return rows, recommended


def _selection_key(m: dict, target_fp: float) -> tuple:
    feasible = m["fp_per_hour"] <= target_fp
    return (feasible, m["recall"] if feasible else -m["fp_per_hour"], m["recall"] if not feasible else -m["fp_per_hour"])


def train(data: TrainingData, settings: dict, device, log, progress, should_stop=lambda: False):
    torch.manual_seed(1234)
    n_frames = data.n_frames
    model = WakeNet(n_frames, int(settings["layer_size"]), int(settings["n_blocks"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pools = {name: FeaturePool(getattr(data, name), device)
             for name in ("pos_tts", "pos_real", "neg_tts", "neg_real", "neg_background")}
    log(f"Training pools: " + ", ".join(f"{k}={v.n}" for k, v in pools.items()))
    log("Opening ACAV100M negative features...")
    acav = AcavStream(data.acav_path, n_frames, device)
    validator = Validator(data, device)
    log(f"False positive validation set: {validator.hours:.1f} hours")

    n_pos, n_neg, n_acav = 64, 64, 1024
    n_pos_real = int(round(n_pos * float(settings["real_weight"]))) if pools["pos_real"].n else 0
    n_neg_real = 16 if pools["neg_real"].n else 0
    n_neg_bg = 8 if pools["neg_background"].n else 0
    counts = {"pos_tts": n_pos - n_pos_real, "pos_real": n_pos_real,
              "neg_tts": n_neg - n_neg_real - n_neg_bg, "neg_real": n_neg_real, "neg_background": n_neg_bg}
    y = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_neg + n_acav, device=device)])

    steps = int(settings["steps"])
    phases = [(steps, 1e-4), (max(100, steps // 10), 1e-5), (max(100, steps // 10), 1e-6)]
    total_steps = sum(p[0] for p in phases)
    max_weight = float(settings["max_negative_weight"])
    target_fp = float(settings["target_fp_per_hour"])

    history, checkpoints = [], []
    best_fp = math.inf
    done = 0
    started = time.time()
    loss_ema = None
    for phase, (n_steps, lr) in enumerate(phases):
        if phase > 0 and best_fp > target_fp:
            max_weight *= 2
            log(f"False positives above target ({best_fp:.2f}/h > {target_fp}/h), doubling negative weight to {max_weight:.0f}")
        log(f"Phase {phase + 1}/3: {n_steps} steps, learning rate {lr:g}")
        first_eval = int(n_steps * 0.75) if phase == 0 else 0
        eval_at = set(np.linspace(first_eval, n_steps - 1, 16 if phase == 0 else 8).astype(int).tolist())
        warmup, hold = n_steps // 5, n_steps // 3
        # openWakeWord ramps the negative weight across 50k steps. Ramping as fast on a short run
        # overwhelms the positives before they are learned (recall collapses to 0), so short runs
        # keep the same slope and simply stop at a lower weight.
        ramp_steps = max(n_steps, 50000 if phase == 0 else 5000)

        for step in range(n_steps):
            if should_stop():
                acav.close()
                raise KeyboardInterrupt("cancelled")
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, warmup, hold, n_steps, lr)
            neg_weight = 1 + (max_weight - 1) * min(1.0, step / ramp_steps)

            parts = [pools[name].sample(n) for name, n in counts.items()]
            x = torch.cat([p for p in parts if p is not None] + [acav.sample(n_acav)])
            pred = model(x).squeeze(1)
            keep = ((y == 0) & (pred >= 0.001)) | ((y == 1) & (pred < 0.999))
            if keep.any():
                yk = y[keep]
                weight = torch.where(yk == 1, torch.ones_like(yk), torch.full_like(yk, neg_weight))
                loss = nn.functional.binary_cross_entropy(pred[keep], yk, weight=weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 50 == 0:
                    loss_ema = loss.item() if loss_ema is None else 0.8 * loss_ema + 0.2 * loss.item()

            if step in eval_at:
                m = validator.evaluate(model)
                m.update(step=done + step, phase=phase + 1)
                history.append(m)
                best_fp = min(best_fp, m["fp_per_hour"])
                checkpoints.append((m, copy.deepcopy(model.state_dict())))
                real = f", your voice {m['recall_real']:.0%}" if m["recall_real"] is not None else ""
                log(f"  step {done + step}: recall {m['recall_tts']:.1%}{real}, "
                    f"similar phrases accepted {m['similar_phrase_accept']:.1%}, false activations {m['fp_per_hour']:.2f}/hour")

            global_step = done + step
            if global_step % 200 == 0:
                elapsed = time.time() - started
                eta = elapsed / max(1, global_step) * (total_steps - global_step)
                loss_txt = f", loss {loss_ema:.4f}" if loss_ema is not None else ""
                progress(global_step / total_steps, f"Step {global_step:,} of {total_steps:,}{loss_txt}, about {eta / 60:.0f} min left")
        done += n_steps
    acav.close()

    ranked = sorted(checkpoints, key=lambda c: _selection_key(c[0], target_fp), reverse=True)
    top = ranked[:5]
    averaged = {k: sum(sd[k].float() for _, sd in top) / len(top) for k in top[0][1]}
    model.load_state_dict(averaged)
    m_avg = validator.evaluate(model)
    best_m, best_sd = ranked[0]
    if _selection_key(m_avg, target_fp) >= _selection_key(best_m, target_fp):
        chosen, how = m_avg, f"average of the best {len(top)} checkpoints"
    else:
        model.load_state_dict(best_sd)
        chosen, how = best_m, f"best single checkpoint (step {best_m['step']})"
    log(f"Selected {how}: recall {chosen['recall_tts']:.1%}, false activations {chosen['fp_per_hour']:.2f}/hour")

    table, recommended = validator.threshold_table(model, target_fp)
    return model.eval(), {"selected": how, "metrics_at_0_5": chosen, "thresholds": table,
                          "recommended_threshold": recommended, "history": history}


@torch.no_grad()
def clip_max_score(model, features: np.ndarray) -> float:
    """Highest score over every window of a clip's features, like streaming detection would see."""
    f = model.n_frames
    if len(features) < f:
        features = np.concatenate([np.zeros((f - len(features), FEATURE_DIM), dtype=np.float32), features])
    windows = np.stack([features[i:i + f] for i in range(len(features) - f + 1)])
    device = next(model.parameters()).device
    return float(model(torch.from_numpy(windows).float().to(device)).max().item())

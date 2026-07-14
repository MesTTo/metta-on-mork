from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = ROOT / "oracle"
SCRATCH_DIR = ROOT / "scratch"

SEED = 20260715
INPUT_DIM = 2
HIDDEN_DIM = 2
OUTPUT_DIM = 2
SETTLE_STEPS = 16
TRAIN_STEPS = 50
ERROR_LR = np.float32(0.1)
WEIGHT_LR = np.float32(0.001)

X_SINGLE = np.asarray([[0.0, 1.0]], dtype=np.float32)
Y_SINGLE = np.asarray([[0.0, 1.0]], dtype=np.float32)

X_TRAIN = np.asarray(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
Y_TRAIN = np.asarray(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Weights:
    wxh: np.ndarray
    why: np.ndarray

    def as_npz(self, prefix: str) -> Dict[str, np.ndarray]:
        return {
            f"{prefix}_wxh": self.wxh.astype(np.float32),
            f"{prefix}_why": self.why.astype(np.float32),
        }


def as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def tanh_prime(state: np.ndarray) -> np.ndarray:
    phi = np.tanh(state, dtype=np.float32)
    return as_float32(1.0 - phi * phi)


JPC_NATIVE_TRACE_KEYS = (
    "pce_h_before",
    "pce_y_before",
    "pre_h",
    "pcs_h",
    "phi_h",
    "pre_y",
    "pcs_y",
    "residual",
    "energy",
    "pcb_h",
    "pcg_h",
    "pcg_y",
    "pce_h_after",
    "pce_y_after",
)

PAPER_TRACE_KEYS = tuple(key for key in JPC_NATIVE_TRACE_KEYS if key not in {"pce_y_before", "pre_y", "pcg_y", "pce_y_after"})


def make_trace(keys: Iterable[str]) -> Dict[str, list[np.ndarray]]:
    return {key: [] for key in keys}


def append_trace(trace: Dict[str, list[np.ndarray]], **values: np.ndarray) -> None:
    for key, value in values.items():
        trace[key].append(np.asarray(value, dtype=np.float32).copy())


def stack_trace(trace: Dict[str, list[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.stack(value).astype(np.float32) for key, value in trace.items()}


def hidden_forward(weights: Weights, x: np.ndarray, e_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_h = as_float32(x @ weights.wxh.T)
    pcs_h = as_float32(pre_h + e_h)
    phi_h = as_float32(np.tanh(pcs_h))
    return pre_h, pcs_h, phi_h


def supervised_context(weights: Weights, x: np.ndarray, e_h: np.ndarray) -> Tuple[np.float32, np.ndarray, np.ndarray, np.ndarray]:
    pre_h, pcs_h, phi_h = hidden_forward(weights, x, e_h)
    return np.float32(x.shape[0]), pre_h, pcs_h, phi_h


def zero_error_state(batch_size: int, include_output: bool) -> Tuple[np.ndarray, ...]:
    state = [np.zeros((batch_size, HIDDEN_DIM), dtype=np.float32)]
    if include_output:
        state.append(np.zeros((batch_size, OUTPUT_DIM), dtype=np.float32))
    return tuple(state)


def mean_energy(e_h: np.ndarray, residual: np.ndarray, batch_size: np.float32) -> np.ndarray:
    return as_float32(
        0.5 * np.sum(e_h * e_h, dtype=np.float32) / batch_size
        + 0.5 * np.sum(residual * residual, dtype=np.float32) / batch_size
    )


def hidden_feedback(weights: Weights, residual: np.ndarray, pcs_h: np.ndarray, e_h: np.ndarray, batch_size: np.float32) -> Tuple[np.ndarray, np.ndarray]:
    pcb_h = as_float32((residual @ weights.why) / batch_size)
    pcg_h = as_float32((e_h / batch_size) - tanh_prime(pcs_h) * pcb_h)
    return pcb_h, pcg_h


def base_step_payload(
    pre_h: np.ndarray,
    pcs_h: np.ndarray,
    phi_h: np.ndarray,
    pcs_y: np.ndarray,
    residual: np.ndarray,
    energy: np.ndarray,
    pcb_h: np.ndarray,
    pcg_h: np.ndarray,
    next_e_h: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "pre_h": pre_h,
        "pcs_h": pcs_h,
        "phi_h": phi_h,
        "pcs_y": pcs_y,
        "residual": residual,
        "energy": np.asarray(energy, dtype=np.float32),
        "pcb_h": pcb_h,
        "pcg_h": pcg_h,
        "pce_h_after": next_e_h,
    }


def settle_trace(
    keys: Iterable[str],
    initial_state: Tuple[np.ndarray, ...],
    steps: int,
    before: Callable[..., Dict[str, np.ndarray]],
    run_step: Callable[..., Dict[str, np.ndarray]],
    advance: Callable[[Tuple[np.ndarray, ...], Dict[str, np.ndarray]], Tuple[np.ndarray, ...]],
) -> Dict[str, np.ndarray]:
    state = initial_state
    trace = make_trace(keys)
    for _ in range(steps):
        append_trace(trace, **before(*state))
        step = run_step(*state)
        append_trace(trace, **step)
        state = advance(state, step)
    return stack_trace(trace)


def jpc_native_step(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    e_h: np.ndarray,
    e_y: np.ndarray,
    error_lr: float = float(ERROR_LR),
) -> Dict[str, np.ndarray]:
    batch_size, pre_h, pcs_h, phi_h = supervised_context(weights, x, e_h)
    pre_y = as_float32(phi_h @ weights.why.T)
    pcs_y = as_float32(pre_y - e_y)
    residual = as_float32(y - pcs_y)
    energy = mean_energy(e_h, residual, batch_size)
    pcg_y = as_float32(residual / batch_size)
    pcb_h, pcg_h = hidden_feedback(weights, residual, pcs_h, e_h, batch_size)
    next_e_h = as_float32(e_h - np.float32(error_lr) * pcg_h)
    next_e_y = as_float32(e_y - np.float32(error_lr) * pcg_y)

    payload = base_step_payload(pre_h, pcs_h, phi_h, pcs_y, residual, energy, pcb_h, pcg_h, next_e_h)
    payload.update({"pre_y": pre_y, "pcg_y": pcg_y, "pce_y_after": next_e_y})
    return payload


def error_settle(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    steps: int = SETTLE_STEPS,
    error_lr: float = float(ERROR_LR),
) -> Dict[str, np.ndarray]:
    if mode == "jpc-native":
        keys = JPC_NATIVE_TRACE_KEYS
        initial_state = zero_error_state(x.shape[0], include_output=True)
        before = lambda e_h, e_y: {"pce_h_before": e_h, "pce_y_before": e_y}
        run_step = lambda e_h, e_y: jpc_native_step(weights, x, y, e_h, e_y, error_lr)
        advance = lambda _state, step: (step["pce_h_after"], step["pce_y_after"])
    elif mode == "paper-error":
        keys = PAPER_TRACE_KEYS
        initial_state = zero_error_state(x.shape[0], include_output=False)
        before = lambda e_h: {"pce_h_before": e_h}
        run_step = lambda e_h: paper_error_step(weights, x, y, e_h, error_lr)
        advance = lambda _state, step: (step["pce_h_after"],)
    else:
        raise ValueError(f"unknown error settle mode: {mode}")
    return settle_trace(keys, initial_state, steps, before, run_step, advance)


def jpc_native_settle(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    steps: int = SETTLE_STEPS,
    error_lr: float = float(ERROR_LR),
) -> Dict[str, np.ndarray]:
    return error_settle(weights, x, y, "jpc-native", steps, error_lr)


def paper_error_step(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    e_h: np.ndarray,
    error_lr: float = float(ERROR_LR),
) -> Dict[str, np.ndarray]:
    batch_size, pre_h, pcs_h, phi_h = supervised_context(weights, x, e_h)
    pcs_y = as_float32(phi_h @ weights.why.T)
    residual = as_float32(y - pcs_y)
    energy = mean_energy(e_h, residual, batch_size)
    pcb_h, pcg_h = hidden_feedback(weights, residual, pcs_h, e_h, batch_size)
    next_e_h = as_float32(e_h - np.float32(error_lr) * pcg_h)
    return base_step_payload(pre_h, pcs_h, phi_h, pcs_y, residual, energy, pcb_h, pcg_h, next_e_h)


def paper_error_settle(
    weights: Weights,
    samples: np.ndarray,
    targets: np.ndarray,
    steps: int = SETTLE_STEPS,
    error_lr: float = float(ERROR_LR),
) -> Dict[str, np.ndarray]:
    return error_settle(weights, samples, targets, "paper-error", steps, error_lr)


def max_abs_rel(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    abs_dev = np.abs(left - right)
    denom = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    rel_dev = abs_dev / denom
    return float(np.max(abs_dev)), float(np.max(rel_dev))


def flatten_relation(parts: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts])


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)

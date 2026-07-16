from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from oracle.common import (
    ERROR_LR,
    LOCAL_TRAIN_CRITERION,
    LOCAL_WEIGHT_LR,
    ROOT,
    SCRATCH_DIR,
    SETTLE_STEPS,
    X_SINGLE,
    X_TRAIN,
    Y_SINGLE,
    Y_TRAIN,
    Weights,
    batch_settle_loss,
    flatten_relation,
    jpc_native_step,
    local_train_to_criterion,
    max_abs_rel,
)


STRATIFIED_BIN = ROOT / "scratch" / "bin" / "mork-einsum-leapfrog-bulk-factorized-stratified"
DEFAULT_MORK_BIN = STRATIFIED_BIN if STRATIFIED_BIN.exists() else Path("/home/user/Dev/mork-integration/MORK/target/release/mork")
MORK_BIN = Path(os.environ.get("MORK_BIN", DEFAULT_MORK_BIN))
RULES_PATH = ROOT / "rules" / "xor_tick.mm2"
CELL_RE = re.compile(r"^\((pcs|pce|pcb|pcg) ([^ ]+) ([0-9]+) ([^)]+)\)$")
WEIGHT_RE = re.compile(r"^\((wxh|why) ([0-9]+) ([0-9]+) ([^)]+)\)$")
PCW_RE = re.compile(r"^\(pcw (xh|hy) ([0-9]+) ([0-9]+) ([^)]+)\)$")
HISTORY_RE = re.compile(r"^\((pcsh|pceh) ([0-9]+) ([^ ]+) ([0-9]+) ([^)]+)\)$")
TRAIN_HISTORY_RE = re.compile(r"^\((pctrsh|pctreh) ([0-9]+) ([0-9]+) ([^ ]+) ([0-9]+) ([^)]+)\)$")
LOADED_RE = re.compile(r"loaded ([0-9]+) expressions")
PHASE_RE = re.compile(r"^; %%PHASE ([A-Z0-9_]+)$")
PHASE_END = "; %%END"
BASE_PHASE_ORDER = (
    "P00_CLEAR_PCS_H",
    "P00_CLEAR_PCS_Y",
    "P00_CLEAR_PCPHI_H",
    "P00_CLEAR_PREH",
    "P00_CLEAR_SH",
    "P00_CLEAR_PHIH",
    "P00_CLEAR_PREY",
    "P00_CLEAR_SY",
    "P00_CLEAR_GY",
    "P00_CLEAR_PCG_Y",
    "P00_CLEAR_BACKH",
    "P00_CLEAR_PCB_H",
    "P00_CLEAR_BH",
    "P00_CLEAR_PHI2H",
    "P00_CLEAR_PRIMEH",
    "P00_CLEAR_BPH",
    "P00_CLEAR_PCG_H",
    "P00_CLEAR_GH",
    "P10_PREH",
    "P10_STATE",
    "P10_PREY",
    "P10_OUTPUT",
    "P20_GY",
    "P20_BACK",
    "P20_PCB",
    "P30_PHI2",
    "P30_PRIME",
    "P30_BP",
    "P30_GH",
    "P35_HIST_EH",
    "P35_HIST_EY",
    "P40_EH",
    "P40_EY",
)
TRAIN_HISTORY_PHASES = (
    "P35_TRAIN_SH_H",
    "P35_TRAIN_SH_Y",
    "P35_TRAIN_EH",
    "P35_TRAIN_EY",
)
WEIGHT_PHASES = (
    "P60_CLEAR_DWXH",
    "P60_CLEAR_DWHY",
    "P60_CLEAR_SDWXH",
    "P60_CLEAR_SDWHY",
    "P60_DWXH",
    "P60_DWHY",
    "P60_SCALE_DWXH",
    "P60_SCALE_DWHY",
    "P60_FOLD_WXH",
    "P60_FOLD_WHY",
    "P60_CLEAR_PCW_XH",
    "P60_SYNC_PCW_XH",
    "P60_CLEAR_PCW_HY",
    "P60_SYNC_PCW_HY",
)
POST_WEIGHT_TICK_PHASES = (
    "P69_ADVANCE_TICK",
)
RELOAD_PHASES = (
    "P70_ADVANCE_UPDATE",
    "P70_CLEAR_PCIN_X",
    "P70_CLEAR_PCS_X",
    "P70_CLEAR_PCPHI_X",
    "P70_CLEAR_PHIX",
    "P70_CLEAR_PCIN_Y",
    "P70_CLEAR_YT",
    "P70_CLEAR_PCE_H",
    "P70_CLEAR_EH",
    "P70_CLEAR_PCE_Y",
    "P70_CLEAR_EY",
    "P71_LOAD_X",
    "P71_LOAD_Y",
    "P71_RESET_EH",
    "P71_RESET_EY",
)
TRAIN_TICK_PHASE_ORDER = BASE_PHASE_ORDER[:-2] + TRAIN_HISTORY_PHASES + BASE_PHASE_ORDER[-2:]
TRAIN_PHASE_ORDER = TRAIN_TICK_PHASE_ORDER + WEIGHT_PHASES + POST_WEIGHT_TICK_PHASES + RELOAD_PHASES
PHASE_ORDER = BASE_PHASE_ORDER


def f32(value: float) -> str:
    return np.format_float_positional(np.float32(value), unique=False, precision=9, trim="-")


def emit_cell_program(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    e_h: np.ndarray,
    e_y: np.ndarray,
    steps: int = SETTLE_STEPS,
) -> str:
    lines = [
        "; Generated fixed-chain XOR cells.",
        "(one 1.0)",
        f"(pchp error-lr {f32(float(ERROR_LR))})",
        f"(pchp weight-lr {f32(float(LOCAL_WEIGHT_LR))})",
        f"(pchp settle-steps {steps})",
        "(pctopo xh x h)",
        "(pctopo hy h y)",
        "(pctick 0)",
    ]
    for tick in range(max(0, steps - 1)):
        lines.append(f"(pcnext {tick} {tick + 1})")
    if steps > 0:
        lines.append(f"(pclast {steps - 1})")
    for i in range(weights.wxh.shape[0]):
        for j in range(weights.wxh.shape[1]):
            value = f32(float(weights.wxh[i, j]))
            lines.append(f"(pcw xh {i} {j} {value})")
            lines.append(f"(wxh {i} {j} {value})")
    for i in range(weights.why.shape[0]):
        for j in range(weights.why.shape[1]):
            value = f32(float(weights.why[i, j]))
            lines.append(f"(pcw hy {i} {j} {value})")
            lines.append(f"(why {i} {j} {value})")
    for i, value in enumerate(x.reshape(-1)):
        text = f32(float(value))
        lines.append(f"(pcin x {i} {text})")
        lines.append(f"(pcs x {i} {text})")
        lines.append(f"(pcphi x {i} {text})")
        lines.append(f"(phix {i} {text})")
    for i, value in enumerate(y.reshape(-1)):
        text = f32(float(value))
        lines.append(f"(pcin y {i} {text})")
        lines.append(f"(yt {i} {text})")
    for i, value in enumerate(e_h.reshape(-1)):
        text = f32(float(value))
        lines.append(f"(pce h {i} {text})")
        lines.append(f"(eh {i} {text})")
    for i, value in enumerate(e_y.reshape(-1)):
        text = f32(float(value))
        lines.append(f"(pce y {i} {text})")
        lines.append(f"(ey {i} {text})")
    return "\n".join(lines) + "\n"


def emit_training_cell_program(weights: Weights, mode: str, updates: int, steps: int = SETTLE_STEPS) -> str:
    if mode not in {"m1", "m2"}:
        raise ValueError(f"unknown training mode: {mode}")
    if updates < 1:
        raise ValueError("training run needs at least one update")
    lines = emit_cell_program(
        weights,
        X_TRAIN[0:1],
        Y_TRAIN[0:1],
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        steps=steps,
    ).splitlines()
    lines.extend(
        [
            f"(pcmode {mode})",
            "(pcupdate 0)",
            f"(pctrain-updates {updates})",
        ]
    )
    for update in range(updates):
        lines.append(f"(pcupdate-sample {update} {update % X_TRAIN.shape[0]})")
        if update + 1 < updates:
            lines.append(f"(pcnextupdate {update} {update + 1})")
    for sample, row in enumerate(X_TRAIN):
        for index, value in enumerate(row.reshape(-1)):
            lines.append(f"(pcx {sample} {index} {f32(float(value))})")
    for sample, row in enumerate(Y_TRAIN):
        for index, value in enumerate(row.reshape(-1)):
            lines.append(f"(pcy {sample} {index} {f32(float(value))})")
    for index in range(2):
        lines.append(f"(pczero h {index} 0)")
        lines.append(f"(pczero y {index} 0)")
    return "\n".join(lines) + "\n"


def count_top_level_expressions(program: str) -> int:
    depth = 0
    count = 0
    in_comment = False
    for char in program:
        if in_comment:
            if char == "\n":
                in_comment = False
            continue
        if char == ";":
            in_comment = True
            continue
        if char == "(":
            if depth == 0:
                count += 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced program: negative parenthesis depth")
    if depth != 0:
        raise ValueError(f"unbalanced program: final parenthesis depth {depth}")
    return count


class MorkRunner:
    def __init__(self, binary: Path = MORK_BIN) -> None:
        self.binary = binary
        self.invocations = 0
        self.logs: List[str] = []
        self.loaded_counts: List[int] = []
        self.wall_times: List[float] = []

    def run(self, program: Path, dump: Path, steps: int = 200) -> str:
        if not self.binary.exists():
            raise FileNotFoundError(f"MORK binary missing: {self.binary}")
        self.invocations += 1
        expected_loaded = count_top_level_expressions(program.read_text())
        start = time.monotonic()
        result = subprocess.run(
            [str(self.binary), "run", "--steps", str(steps), str(program), str(dump)],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        self.wall_times.append(time.monotonic() - start)
        output = result.stdout + result.stderr
        self.logs.append(output)
        if result.returncode != 0:
            raise RuntimeError(
                f"MORK failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        loaded_match = LOADED_RE.search(output)
        if loaded_match is None:
            raise AssertionError(f"MORK output did not report loaded expressions:\n{output}")
        loaded = int(loaded_match.group(1))
        if loaded != expected_loaded:
            raise AssertionError(f"MORK loaded {loaded} expressions, expected {expected_loaded} from {program}")
        self.loaded_counts.append(loaded)
        return output


def load_phase_rules() -> Dict[str, str]:
    phases: Dict[str, list[str]] = {}
    current: str | None = None
    for line in RULES_PATH.read_text().splitlines():
        match = PHASE_RE.match(line)
        if match is not None:
            current = match.group(1)
            phases[current] = []
            continue
        if line == PHASE_END:
            current = None
            continue
        if current is not None:
            phases[current].append(line)
    required = tuple(dict.fromkeys(BASE_PHASE_ORDER + TRAIN_HISTORY_PHASES + WEIGHT_PHASES + POST_WEIGHT_TICK_PHASES + RELOAD_PHASES))
    missing = [phase for phase in required if phase not in phases]
    if missing:
        raise KeyError(f"missing rule phase(s): {missing}")
    return {phase: "\n".join(lines).strip() + "\n" for phase, lines in phases.items()}


def phase_symbol(phase: str) -> str:
    return phase.lower().replace("_", "-")


def replace_head(form: str, head: str) -> str:
    lines = form.strip().splitlines()
    if not lines:
        raise ValueError("empty rule form")
    lines[0] = head
    return "\n".join(lines)


def replace_exec_head(form: str, head: str) -> str:
    lines = form.strip().splitlines()
    if not lines or not lines[0].startswith("(exec "):
        raise ValueError(f"phase block is not a single exec form:\n{form}")
    return replace_head(form, head)


def phase_fact(phase: str, body: str) -> str:
    return replace_exec_head(body, f"((pcphase {phase_symbol(phase)})")


def phase_exec(phase: str, body: str) -> str:
    return replace_exec_head(body, f"(exec (pcphase {phase_symbol(phase)})")


def barrier_fact(phase: str, next_phase: str | None, phase_order: Sequence[str] = BASE_PHASE_ORDER) -> str:
    symbol = phase_symbol(phase)
    if next_phase is None:
        first = phase_symbol(phase_order[0])
        return f"""((pcbarrier {symbol})
  (, (pctick $tick)
     (pcnext $tick $next)
     ((pcphase {first}) $p $t)
     ((pcbarrier {first}) $bp $bt))
  (O (- (pctick $tick))
     (+ (pctick $next))
     (+ (pcadvanced $tick $next))
     (+ (exec (pcphase {first}) $p $t))
     (+ (exec (quiesce {first}) $bp $bt))))"""
    next_symbol = phase_symbol(next_phase)
    return f"""((pcbarrier {symbol})
  (, ((pcphase {next_symbol}) $p $t)
     ((pcbarrier {next_symbol}) $bp $bt))
  (, (exec (pcphase {next_symbol}) $p $t)
     (exec (quiesce {next_symbol}) $bp $bt)))"""


def barrier_exec(phase: str, fact: str) -> str:
    return replace_head(fact, f"(exec (quiesce {phase_symbol(phase)})")


def build_single_invocation_program(facts: str, phases: Dict[str, str], phase_order: Sequence[str] = BASE_PHASE_ORDER) -> str:
    phase_facts = [phase_fact(phase, phases[phase]) for phase in phase_order]
    barrier_facts = []
    for index, phase in enumerate(phase_order):
        next_phase = phase_order[index + 1] if index + 1 < len(phase_order) else None
        barrier_facts.append(barrier_fact(phase, next_phase, phase_order))

    first_phase = phase_order[0]
    program_parts = [
        facts,
        "\n".join(phase_facts),
        "\n".join(barrier_facts),
        phase_exec(first_phase, phases[first_phase]),
        barrier_exec(first_phase, barrier_facts[0]),
    ]
    return "\n\n".join(part.strip() for part in program_parts) + "\n"


def normal_barrier_fact(phase: str, next_phase: str) -> str:
    return barrier_fact(phase, next_phase, TRAIN_PHASE_ORDER)


def p40_training_barrier_fact() -> str:
    symbol = phase_symbol("P40_EY")
    first = phase_symbol(TRAIN_TICK_PHASE_ORDER[0])
    weight = phase_symbol(WEIGHT_PHASES[0])
    return f"""((pcbarrier {symbol})
  (, (pcmode m1)
     (pctick $tick)
     (pcnext $tick $next)
     ((pcphase {first}) $p $t)
     ((pcbarrier {first}) $bp $bt))
  (O (- (pctick $tick))
     (+ (pctick $next))
     (+ (pcadvanced $tick $next))
     (+ (exec (pcphase {first}) $p $t))
     (+ (exec (quiesce {first}) $bp $bt))))

((pcbarrier {symbol})
  (, (pcmode m1)
     (pctick $tick)
     (pclast $tick)
     ((pcphase {weight}) $p $t)
     ((pcbarrier {weight}) $bp $bt))
  (O (+ (pcweight-final $tick))
     (+ (exec (pcphase {weight}) $p $t))
     (+ (exec (quiesce {weight}) $bp $bt))))

((pcbarrier {symbol})
  (, (pcmode m2)
     (pctick $tick)
     (pcnext $tick $next)
     ((pcphase {weight}) $p $t)
     ((pcbarrier {weight}) $bp $bt))
  (O (+ (pcweight-cont $tick))
     (+ (exec (pcphase {weight}) $p $t))
     (+ (exec (quiesce {weight}) $bp $bt))))

((pcbarrier {symbol})
  (, (pcmode m2)
     (pctick $tick)
     (pclast $tick)
     ((pcphase {weight}) $p $t)
     ((pcbarrier {weight}) $bp $bt))
  (O (+ (pcweight-final $tick))
     (+ (exec (pcphase {weight}) $p $t))
     (+ (exec (quiesce {weight}) $bp $bt))))"""


def weight_tail_training_barrier_fact() -> str:
    symbol = phase_symbol(WEIGHT_PHASES[-1])
    advance_tick = phase_symbol(POST_WEIGHT_TICK_PHASES[0])
    reload_first = phase_symbol(RELOAD_PHASES[0])
    return f"""((pcbarrier {symbol})
  (, (pcmode m2)
     (pcweight-cont $tick)
     (pctick $tick)
     (pcnext $tick $next)
     ((pcphase {advance_tick}) $p $t)
     ((pcbarrier {advance_tick}) $bp $bt))
  (O (- (pcweight-cont $tick))
     (+ (exec (pcphase {advance_tick}) $p $t))
     (+ (exec (quiesce {advance_tick}) $bp $bt))))

((pcbarrier {symbol})
  (, (pctick $tick)
     (pcweight-final $tick)
     (pclast $tick)
     (pcupdate $u)
     (pcnextupdate $u $next)
     ((pcphase {reload_first}) $p $t)
     ((pcbarrier {reload_first}) $bp $bt))
  (O (- (pcweight-final $tick))
     (+ (exec (pcphase {reload_first}) $p $t))
     (+ (exec (quiesce {reload_first}) $bp $bt))))"""


def reload_tail_training_barrier_fact() -> str:
    symbol = phase_symbol(RELOAD_PHASES[-1])
    first = phase_symbol(TRAIN_TICK_PHASE_ORDER[0])
    return f"""((pcbarrier {symbol})
  (, ((pcphase {first}) $p $t)
     ((pcbarrier {first}) $bp $bt))
  (, (exec (pcphase {first}) $p $t)
     (exec (quiesce {first}) $bp $bt)))"""


def advance_tick_tail_training_barrier_fact() -> str:
    symbol = phase_symbol(POST_WEIGHT_TICK_PHASES[0])
    first = phase_symbol(TRAIN_TICK_PHASE_ORDER[0])
    return f"""((pcbarrier {symbol})
  (, ((pcphase {first}) $p $t)
     ((pcbarrier {first}) $bp $bt))
  (, (exec (pcphase {first}) $p $t)
     (exec (quiesce {first}) $bp $bt)))"""


def build_training_program(facts: str, phases: Dict[str, str]) -> str:
    phase_facts = [phase_fact(phase, phases[phase]) for phase in TRAIN_PHASE_ORDER]
    barrier_facts: List[str] = []
    for index, phase in enumerate(TRAIN_TICK_PHASE_ORDER):
        if phase == "P40_EY":
            barrier_facts.append(p40_training_barrier_fact())
        else:
            barrier_facts.append(normal_barrier_fact(phase, TRAIN_TICK_PHASE_ORDER[index + 1]))
    for index, phase in enumerate(WEIGHT_PHASES):
        if index + 1 < len(WEIGHT_PHASES):
            barrier_facts.append(normal_barrier_fact(phase, WEIGHT_PHASES[index + 1]))
        else:
            barrier_facts.append(weight_tail_training_barrier_fact())
    barrier_facts.append(advance_tick_tail_training_barrier_fact())
    for index, phase in enumerate(RELOAD_PHASES):
        if index + 1 < len(RELOAD_PHASES):
            barrier_facts.append(normal_barrier_fact(phase, RELOAD_PHASES[index + 1]))
        else:
            barrier_facts.append(reload_tail_training_barrier_fact())

    first_phase = TRAIN_TICK_PHASE_ORDER[0]
    program_parts = [
        facts,
        "\n".join(phase_facts),
        "\n".join(barrier_facts),
        phase_exec(first_phase, phases[first_phase]),
        barrier_exec(first_phase, barrier_facts[0]),
    ]
    return "\n\n".join(part.strip() for part in program_parts) + "\n"


def parse_cells(dump: Path) -> Dict[Tuple[str, str, int], float]:
    cells: Dict[Tuple[str, str, int], float] = {}
    for line in dump.read_text().splitlines():
        match = CELL_RE.match(line.strip())
        if match is None:
            continue
        rel, node, index, value = match.groups()
        cells[(rel, node, int(index))] = float(np.float32(value))
    return cells


def parse_history(dump: Path) -> Dict[Tuple[str, int, str, int], float]:
    history: Dict[Tuple[str, int, str, int], float] = {}
    for line in dump.read_text().splitlines():
        match = HISTORY_RE.match(line.strip())
        if match is None:
            continue
        rel, tick, node, index, value = match.groups()
        history[(rel, int(tick), node, int(index))] = float(np.float32(value))
    return history


def parse_train_history(dump: Path) -> Dict[Tuple[str, int, int, str, int], float]:
    history: Dict[Tuple[str, int, int, str, int], float] = {}
    for line in dump.read_text().splitlines():
        match = TRAIN_HISTORY_RE.match(line.strip())
        if match is None:
            continue
        rel, update, tick, node, index, value = match.groups()
        history[(rel, int(update), int(tick), node, int(index))] = float(np.float32(value))
    return history


def parse_weights(dump: Path) -> Weights:
    wxh = np.full((2, 2), np.nan, dtype=np.float32)
    why = np.full((2, 2), np.nan, dtype=np.float32)
    for line in dump.read_text().splitlines():
        match = WEIGHT_RE.match(line.strip())
        if match is None:
            continue
        rel, row, col, value = match.groups()
        target = wxh if rel == "wxh" else why
        target[int(row), int(col)] = np.float32(value)
    if np.isnan(wxh).any() or np.isnan(why).any():
        raise KeyError("dump did not contain a complete wxh/why weight set")
    return Weights(wxh, why)


def parse_pcw_weights(dump: Path) -> Weights:
    wxh = np.full((2, 2), np.nan, dtype=np.float32)
    why = np.full((2, 2), np.nan, dtype=np.float32)
    for line in dump.read_text().splitlines():
        match = PCW_RE.match(line.strip())
        if match is None:
            continue
        edge, row, col, value = match.groups()
        target = wxh if edge == "xh" else why
        target[int(row), int(col)] = np.float32(value)
    if np.isnan(wxh).any() or np.isnan(why).any():
        raise KeyError("dump did not contain a complete pcw weight set")
    return Weights(wxh, why)


def get_vec(cells: Dict[Tuple[str, str, int], float], rel: str, node: str, size: int = 2) -> np.ndarray:
    values = []
    for index in range(size):
        key = (rel, node, index)
        if key not in cells:
            raise KeyError(f"missing cell {rel} {node} {index}")
        values.append(cells[key])
    return np.asarray(values, dtype=np.float32).reshape(1, size)


def get_history_vec(
    history: Dict[Tuple[str, int, str, int], float],
    rel: str,
    tick: int,
    node: str,
    size: int = 2,
) -> np.ndarray:
    values = []
    for index in range(size):
        key = (rel, tick, node, index)
        if key not in history:
            raise KeyError(f"missing history cell {rel} {tick} {node} {index}")
        values.append(history[key])
    return np.asarray(values, dtype=np.float32).reshape(1, size)


def get_train_history_vec(
    history: Dict[Tuple[str, int, int, str, int], float],
    rel: str,
    update: int,
    tick: int,
    node: str,
    size: int = 2,
) -> np.ndarray:
    values = []
    for index in range(size):
        key = (rel, update, tick, node, index)
        if key not in history:
            raise KeyError(f"missing training history cell {rel} {update} {tick} {node} {index}")
        values.append(history[key])
    return np.asarray(values, dtype=np.float32).reshape(1, size)


def history_counts(history: Dict[Tuple[str, int, str, int], float]) -> Dict[str, int]:
    return {
        "pcsh": sum(1 for rel, *_ in history if rel == "pcsh"),
        "pceh": sum(1 for rel, *_ in history if rel == "pceh"),
    }


def history_energy_rows(history: Dict[Tuple[str, int, str, int], float], y: np.ndarray, steps: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for tick in range(steps):
        pcs_y = get_history_vec(history, "pcsh", tick, "y")
        e_h = get_history_vec(history, "pceh", tick, "h")
        residual = np.asarray(y, dtype=np.float32) - pcs_y
        energy = np.float32(
            0.5 * np.sum(e_h * e_h, dtype=np.float32)
            + 0.5 * np.sum(residual * residual, dtype=np.float32)
        )
        rows.append({"tick": tick, "energy": float(energy)})
    return rows


def train_history_counts(history: Dict[Tuple[str, int, int, str, int], float]) -> Dict[str, int]:
    return {
        "pctrsh": sum(1 for rel, *_ in history if rel == "pctrsh"),
        "pctreh": sum(1 for rel, *_ in history if rel == "pctreh"),
    }


def train_sample_energy_rows(
    history: Dict[Tuple[str, int, int, str, int], float],
    updates: int,
    steps: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    tick = steps - 1
    for update in range(updates):
        sample = update % X_TRAIN.shape[0]
        pcs_y = get_train_history_vec(history, "pctrsh", update, tick, "y")
        e_h = get_train_history_vec(history, "pctreh", update, tick, "h")
        residual = np.asarray(Y_TRAIN[sample : sample + 1], dtype=np.float32) - pcs_y
        energy = np.float32(
            0.5 * np.sum(e_h * e_h, dtype=np.float32)
            + 0.5 * np.sum(residual * residual, dtype=np.float32)
        )
        rows.append({"update": update + 1, "sample": sample, "sample_energy": float(energy)})
    return rows


def final_cell_payload(cells: Dict[Tuple[str, str, int], float]) -> Dict[str, np.ndarray]:
    return {
        "pcs_h": get_vec(cells, "pcs", "h"),
        "pcs_y": get_vec(cells, "pcs", "y"),
        "pcb_h": get_vec(cells, "pcb", "h"),
        "pcg_h": get_vec(cells, "pcg", "h"),
        "pcg_y": get_vec(cells, "pcg", "y"),
        "pce_h_after": get_vec(cells, "pce", "h"),
        "pce_y_after": get_vec(cells, "pce", "y"),
    }


def run_tick(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    e_h: np.ndarray,
    e_y: np.ndarray,
    step_index: int,
    runner: MorkRunner,
) -> Dict[str, np.ndarray]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    facts = emit_cell_program(weights, x, y, e_h, e_y, steps=1)
    phases = load_phase_rules()
    dump = SCRATCH_DIR / f"xor_tick_{step_index:02d}.dump"
    for phase_index, phase in enumerate(PHASE_ORDER, start=1):
        program = SCRATCH_DIR / f"xor_tick_{step_index:02d}_{phase_index:02d}_{phase}.mm2"
        dump = SCRATCH_DIR / f"xor_tick_{step_index:02d}_{phase_index:02d}_{phase}.dump"
        program.write_text(facts + "\n" + phases[phase])
        runner.run(program, dump, steps=20)
        facts = dump.read_text()
    final_dump = SCRATCH_DIR / f"xor_tick_{step_index:02d}.dump"
    final_dump.write_text(facts)
    dump = final_dump
    cells = parse_cells(dump)
    return final_cell_payload(cells)


def relation_table(expected: Dict[str, np.ndarray], got: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    relations = {
        "pcs": (expected["pcs_h"], expected["pcs_y"], got["pcs_h"], got["pcs_y"]),
        "pcb": (expected["pcb_h"], got["pcb_h"]),
        "pcg": (expected["pcg_h"], expected["pcg_y"], got["pcg_h"], got["pcg_y"]),
        "pce": (expected["pce_h_after"], expected["pce_y_after"], got["pce_h_after"], got["pce_y_after"]),
    }
    table: Dict[str, Dict[str, float]] = {}
    for rel, parts in relations.items():
        half = len(parts) // 2
        exp = flatten_relation(parts[:half])
        actual = flatten_relation(parts[half:])
        abs_dev, rel_dev = max_abs_rel(exp, actual)
        table[rel] = {"max_abs": abs_dev, "max_rel": rel_dev}
    return table


def compare_or_raise(table: Dict[str, Dict[str, float]], tolerance: float = 5e-5) -> None:
    bad = {
        rel: values
        for rel, values in table.items()
        if values["max_abs"] > tolerance and values["max_rel"] > tolerance
    }
    if bad:
        raise AssertionError(f"MORK cell gate failed: {json.dumps(bad, sort_keys=True)}")


def oracle_steps(weights: Weights, steps: int) -> List[Dict[str, np.ndarray]]:
    e_h = np.zeros((1, 2), dtype=np.float32)
    e_y = np.zeros((1, 2), dtype=np.float32)
    rows = []
    for _step_index in range(steps):
        expected = jpc_native_step(weights, X_SINGLE, Y_SINGLE, e_h, e_y)
        expected["pce_h_before"] = e_h.copy()
        expected["pce_y_before"] = e_y.copy()
        rows.append(expected)
        e_h = expected["pce_h_after"]
        e_y = expected["pce_y_after"]
    return rows


def run_legacy_settle(weights: Weights, steps: int = SETTLE_STEPS):
    e_h = np.zeros((1, 2), dtype=np.float32)
    e_y = np.zeros((1, 2), dtype=np.float32)
    energy_rows = []
    first_table = None
    final_cells = None
    runner = MorkRunner()
    for step_index in range(steps):
        expected = jpc_native_step(weights, X_SINGLE, Y_SINGLE, e_h, e_y)
        got = run_tick(weights, X_SINGLE, Y_SINGLE, e_h, e_y, step_index + 1, runner)
        table = relation_table(expected, got)
        if first_table is None:
            first_table = table
        compare_or_raise(table)
        energy_rows.append(
            {
                "step": step_index + 1,
                "oracle": float(expected["energy"]),
                "mork": float(expected["energy"]),
                "max_cell_abs": max(values["max_abs"] for values in table.values()),
                "max_cell_rel": max(values["max_rel"] for values in table.values()),
            }
        )
        e_h = got["pce_h_after"]
        e_y = got["pce_y_after"]
        final_cells = got
    if first_table is None or final_cells is None:
        raise AssertionError("legacy settle did not execute any steps")
    return {
        "first_iteration_gate": first_table,
        "energy_trajectory": energy_rows,
        "final_cells": final_cells,
        "mork_invocations": runner.invocations,
        "loaded_count_summary": loaded_count_summary(runner.loaded_counts),
        "wall_seconds_total": float(sum(runner.wall_times)),
    }


def history_step_table(
    expected_rows: Sequence[Dict[str, np.ndarray]],
    history: Dict[Tuple[str, int, str, int], float],
    energy_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    rows = []
    for tick, expected in enumerate(expected_rows):
        history_parts = (
            get_history_vec(history, "pcsh", tick, "h"),
            get_history_vec(history, "pcsh", tick, "y"),
            get_history_vec(history, "pceh", tick, "h"),
            get_history_vec(history, "pceh", tick, "y"),
        )
        expected_parts = (
            expected["pcs_h"],
            expected["pcs_y"],
            expected["pce_h_before"],
            expected["pce_y_before"],
        )
        hist_abs, hist_rel = max_abs_rel(flatten_relation(expected_parts), flatten_relation(history_parts))
        energy_abs, energy_rel = max_abs_rel(
            np.asarray([expected["energy"]], dtype=np.float32),
            np.asarray([energy_rows[tick]["energy"]], dtype=np.float32),
        )
        rows.append(
            {
                "step": tick + 1,
                "tick": tick,
                "oracle_energy": float(expected["energy"]),
                "mork_history_energy": energy_rows[tick]["energy"],
                "energy_abs": energy_abs,
                "energy_rel": energy_rel,
                "history_cell_abs": hist_abs,
                "history_cell_rel": hist_rel,
            }
        )
    return rows


def run_single_invocation_settle(weights: Weights, steps: int = SETTLE_STEPS):
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    facts = emit_cell_program(
        weights,
        X_SINGLE,
        Y_SINGLE,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        steps=steps,
    )
    phases = load_phase_rules()
    program = SCRATCH_DIR / "xor_settle_single.mm2"
    dump = SCRATCH_DIR / "xor_settle_single.dump"
    program.write_text(build_single_invocation_program(facts, phases))
    runner = MorkRunner()
    runner.run(program, dump, steps=max(1_000, steps * len(PHASE_ORDER) * 4))
    if runner.invocations != 1:
        raise AssertionError(f"single path invoked MORK {runner.invocations} times")

    cells = parse_cells(dump)
    history = parse_history(dump)
    expected_rows = oracle_steps(weights, steps)
    final_table = relation_table(expected_rows[-1], final_cell_payload(cells))
    compare_or_raise(final_table)

    counts = history_counts(history)
    expected_history_count = steps * 4
    if counts != {"pcsh": expected_history_count, "pceh": expected_history_count}:
        raise AssertionError(f"bad history counts: {counts}, expected {expected_history_count} per relation")

    energy_rows = history_energy_rows(history, Y_SINGLE, steps)
    step_table = history_step_table(expected_rows, history, energy_rows)
    bad_rows = [
        row
        for row in step_table
        if (row["energy_abs"] > 5e-5 and row["energy_rel"] > 5e-5)
        or (row["history_cell_abs"] > 5e-5 and row["history_cell_rel"] > 5e-5)
    ]
    if bad_rows:
        raise AssertionError(f"single path history gate failed: {json.dumps(bad_rows, sort_keys=True)}")

    return {
        "final_iteration_gate": final_table,
        "energy_trajectory": step_table,
        "history_counts": counts,
        "final_cells": final_cell_payload(cells),
        "mork_invocations": runner.invocations,
        "loaded_counts": runner.loaded_counts,
        "wall_seconds": runner.wall_times,
        "program": str(program),
        "dump": str(dump),
    }


def flatten_final_cells(payload: Dict[str, np.ndarray]) -> np.ndarray:
    return flatten_relation(
        (
            payload["pcs_h"],
            payload["pcs_y"],
            payload["pcb_h"],
            payload["pcg_h"],
            payload["pcg_y"],
            payload["pce_h_after"],
            payload["pce_y_after"],
        )
    )


def flatten_weights(weights: Weights) -> np.ndarray:
    return flatten_relation((weights.wxh, weights.why))


def weights_table(expected: Weights, got: Weights) -> Dict[str, Dict[str, float]]:
    wxh_abs, wxh_rel = max_abs_rel(expected.wxh, got.wxh)
    why_abs, why_rel = max_abs_rel(expected.why, got.why)
    return {
        "wxh": {"max_abs": wxh_abs, "max_rel": wxh_rel},
        "why": {"max_abs": why_abs, "max_rel": why_rel},
    }


def loaded_count_summary(counts: Sequence[int]) -> Dict[str, int]:
    if not counts:
        raise AssertionError("no loaded-expression counts recorded")
    return {"count": len(counts), "min": min(counts), "max": max(counts)}


def run_single_invocation_training(weights: Weights, mode: str, updates: int, steps: int = SETTLE_STEPS) -> Dict[str, Any]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    facts = emit_training_cell_program(weights, mode, updates, steps=steps)
    phases = load_phase_rules()
    program = SCRATCH_DIR / f"xor_train_{mode}_{updates:04d}.mm2"
    dump = SCRATCH_DIR / f"xor_train_{mode}_{updates:04d}.dump"
    program.write_text(build_training_program(facts, phases))
    runner = MorkRunner()
    max_steps = max(
        10_000,
        updates
        * steps
        * (len(TRAIN_TICK_PHASE_ORDER) + len(WEIGHT_PHASES) + len(POST_WEIGHT_TICK_PHASES) + len(RELOAD_PHASES) + 8)
        * 4,
    )
    runner.run(program, dump, steps=max_steps)
    if runner.invocations != 1:
        raise AssertionError(f"{mode} training invoked MORK {runner.invocations} times")

    history = parse_train_history(dump)
    counts = train_history_counts(history)
    expected_history_count = updates * steps * 4
    if counts != {"pctrsh": expected_history_count, "pctreh": expected_history_count}:
        raise AssertionError(f"bad {mode} train history counts: {counts}, expected {expected_history_count} per relation")

    final_weights = parse_weights(dump)
    pcw_weights = parse_pcw_weights(dump)
    pcw_abs, pcw_rel = max_abs_rel(flatten_weights(final_weights), flatten_weights(pcw_weights))
    if pcw_abs > 5e-7 and pcw_rel > 5e-7:
        raise AssertionError(f"{mode} pcw mirror diverged from helper weights: abs={pcw_abs} rel={pcw_rel}")
    batch_energy, batch_pred = batch_settle_loss(final_weights)
    return {
        "mode": mode,
        "updates": updates,
        "sample_energy_trajectory": train_sample_energy_rows(history, updates, steps),
        "history_counts": counts,
        "final_weights": final_weights,
        "pcw_weights": pcw_weights,
        "pcw_vs_helper": {"max_abs": pcw_abs, "max_rel": pcw_rel},
        "final_batch_energy": float(batch_energy),
        "final_batch_pred": batch_pred,
        "mork_invocations": runner.invocations,
        "loaded_counts": runner.loaded_counts,
        "wall_seconds": runner.wall_times,
        "program": str(program),
        "dump": str(dump),
    }


def load_weights(path: Path) -> Weights:
    data = np.load(path)
    return Weights(data["initial_wxh"].astype(np.float32), data["initial_why"].astype(np.float32))


def training_payload(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in run.items()
        if key not in {"final_weights", "pcw_weights", "final_batch_pred"}
    } | {"final_batch_pred": np.asarray(run["final_batch_pred"], dtype=np.float32).tolist()}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle" / "xor_jpc_reference.npz")
    parser.add_argument("--steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--m1-updates", type=int)
    parser.add_argument("--m2-updates", type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)
    weights = load_weights(args.oracle)
    single = run_single_invocation_settle(weights, args.steps)
    report = {
        "mork_bin": str(MORK_BIN),
        "rules": str(RULES_PATH),
        "single_invocation": {key: value for key, value in single.items() if key != "final_cells"},
        "timing_note": "provisional-under-load",
    }
    if args.compare_legacy:
        legacy = run_legacy_settle(weights, args.steps)
        legacy_abs, legacy_rel = max_abs_rel(
            flatten_final_cells(single["final_cells"]),
            flatten_final_cells(legacy["final_cells"]),
        )
        report["legacy_reference"] = {
            "first_iteration_gate": legacy["first_iteration_gate"],
            "energy_trajectory": legacy["energy_trajectory"],
            "mork_invocations": legacy["mork_invocations"],
            "loaded_count_summary": legacy["loaded_count_summary"],
            "wall_seconds_total": legacy["wall_seconds_total"],
        }
        report["legacy_vs_single_final_cells"] = {
            "max_abs": legacy_abs,
            "max_rel": legacy_rel,
        }
    if args.train:
        m1_ref = local_train_to_criterion(weights, "m1")
        m2_ref = local_train_to_criterion(weights, "m2")
        m1_updates = args.m1_updates or int(m1_ref["criterion_update"])
        m2_updates = args.m2_updates or int(m2_ref["criterion_update"])
        m1_one = run_single_invocation_training(weights, "m1", 1, args.steps)
        m1 = run_single_invocation_training(weights, "m1", m1_updates, args.steps)
        m2 = run_single_invocation_training(weights, "m2", m2_updates, args.steps)
        report["training"] = {
            "criterion": float(LOCAL_TRAIN_CRITERION),
            "weight_lr": float(LOCAL_WEIGHT_LR),
            "m1_reference_criterion_update": int(m1_ref["criterion_update"]),
            "m2_reference_criterion_update": int(m2_ref["criterion_update"]),
            "m1_one_step": training_payload(m1_one),
            "m1": training_payload(m1),
            "m2": training_payload(m2),
        }
    report_path = SCRATCH_DIR / "mork_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

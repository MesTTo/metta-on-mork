from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from oracle.common import (
    ERROR_LR,
    ROOT,
    SCRATCH_DIR,
    SETTLE_STEPS,
    X_SINGLE,
    Y_SINGLE,
    Weights,
    flatten_relation,
    jpc_native_step,
    max_abs_rel,
)


MORK_BIN = Path("/home/user/Dev/mork-integration/MORK/target/release/mork")
RULES_PATH = ROOT / "rules" / "xor_tick.mm2"
CELL_RE = re.compile(r"^\((pcs|pce|pcb|pcg) ([^ ]+) ([0-9]+) ([^)]+)\)$")
PHASE_RE = re.compile(r"^; %%PHASE ([A-Z0-9_]+)$")
PHASE_END = "; %%END"
PHASE_ORDER = (
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
    "P40_EH",
    "P40_EY",
)


def f32(value: float) -> str:
    return np.format_float_positional(np.float32(value), unique=False, precision=9, trim="-")


def emit_cell_program(weights: Weights, x: np.ndarray, y: np.ndarray, e_h: np.ndarray, e_y: np.ndarray) -> str:
    lines = [
        "; Generated fixed-chain XOR cells.",
        "(one 1.0)",
        f"(pchp error-lr {f32(float(ERROR_LR))})",
        "(pchp settle-steps 16)",
        "(pctopo xh x h)",
        "(pctopo hy h y)",
        "(pctick 0)",
    ]
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


def run_mork(program: Path, dump: Path, steps: int = 200) -> str:
    if not MORK_BIN.exists():
        raise FileNotFoundError(f"MORK binary missing: {MORK_BIN}")
    result = subprocess.run(
        [str(MORK_BIN), "run", "--steps", str(steps), str(program), str(dump)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MORK failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout + result.stderr


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
    missing = [phase for phase in PHASE_ORDER if phase not in phases]
    if missing:
        raise KeyError(f"missing rule phase(s): {missing}")
    return {phase: "\n".join(lines).strip() + "\n" for phase, lines in phases.items()}


def parse_cells(dump: Path) -> Dict[Tuple[str, str, int], float]:
    cells: Dict[Tuple[str, str, int], float] = {}
    for line in dump.read_text().splitlines():
        match = CELL_RE.match(line.strip())
        if match is None:
            continue
        rel, node, index, value = match.groups()
        cells[(rel, node, int(index))] = float(np.float32(value))
    return cells


def get_vec(cells: Dict[Tuple[str, str, int], float], rel: str, node: str, size: int = 2) -> np.ndarray:
    values = []
    for index in range(size):
        key = (rel, node, index)
        if key not in cells:
            raise KeyError(f"missing cell {rel} {node} {index}")
        values.append(cells[key])
    return np.asarray(values, dtype=np.float32).reshape(1, size)


def run_tick(weights: Weights, x: np.ndarray, y: np.ndarray, e_h: np.ndarray, e_y: np.ndarray, step_index: int) -> Dict[str, np.ndarray]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    facts = emit_cell_program(weights, x, y, e_h, e_y)
    phases = load_phase_rules()
    dump = SCRATCH_DIR / f"xor_tick_{step_index:02d}.dump"
    for phase_index, phase in enumerate(PHASE_ORDER, start=1):
        program = SCRATCH_DIR / f"xor_tick_{step_index:02d}_{phase_index:02d}_{phase}.mm2"
        dump = SCRATCH_DIR / f"xor_tick_{step_index:02d}_{phase_index:02d}_{phase}.dump"
        program.write_text(facts + "\n" + phases[phase])
        run_mork(program, dump, steps=20)
        facts = dump.read_text()
    final_dump = SCRATCH_DIR / f"xor_tick_{step_index:02d}.dump"
    final_dump.write_text(facts)
    dump = final_dump
    cells = parse_cells(dump)
    return {
        "pcs_h": get_vec(cells, "pcs", "h"),
        "pcs_y": get_vec(cells, "pcs", "y"),
        "pcb_h": get_vec(cells, "pcb", "h"),
        "pcg_h": get_vec(cells, "pcg", "h"),
        "pcg_y": get_vec(cells, "pcg", "y"),
        "pce_h_after": get_vec(cells, "pce", "h"),
        "pce_y_after": get_vec(cells, "pce", "y"),
    }


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


def run_settle(weights: Weights, steps: int = SETTLE_STEPS):
    e_h = np.zeros((1, 2), dtype=np.float32)
    e_y = np.zeros((1, 2), dtype=np.float32)
    energy_rows = []
    first_table = None
    for step_index in range(steps):
        expected = jpc_native_step(weights, X_SINGLE, Y_SINGLE, e_h, e_y)
        got = run_tick(weights, X_SINGLE, Y_SINGLE, e_h, e_y, step_index + 1)
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
    return first_table, energy_rows


def load_weights(path: Path) -> Weights:
    data = np.load(path)
    return Weights(data["initial_wxh"].astype(np.float32), data["initial_why"].astype(np.float32))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle" / "xor_jpc_reference.npz")
    parser.add_argument("--steps", type=int, default=SETTLE_STEPS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    weights = load_weights(args.oracle)
    first_table, energy_rows = run_settle(weights, args.steps)
    report = {
        "mork_bin": str(MORK_BIN),
        "rules": str(RULES_PATH),
        "first_iteration_gate": first_table,
        "energy_trajectory": energy_rows,
    }
    report_path = SCRATCH_DIR / "mork_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


STRATIFIED_BIN = ROOT / "scratch" / "bin" / "mork-einsum-leapfrog-bulk-factorized-stratified"
DEFAULT_MORK_BIN = STRATIFIED_BIN if STRATIFIED_BIN.exists() else Path("/home/user/Dev/mork-integration/MORK/target/release/mork")
MORK_BIN = Path(os.environ.get("MORK_BIN", DEFAULT_MORK_BIN))
RULES_PATH = ROOT / "rules" / "xor_tick.mm2"
CELL_RE = re.compile(r"^\((pcs|pce|pcb|pcg) ([^ ]+) ([0-9]+) ([^)]+)\)$")
HISTORY_RE = re.compile(r"^\((pcsh|pceh) ([0-9]+) ([^ ]+) ([0-9]+) ([^)]+)\)$")
PHASE_RE = re.compile(r"^; %%PHASE ([A-Z0-9_]+)$")
PHASE_END = "; %%END"
PHASE_ORDER = (
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


class MorkRunner:
    def __init__(self, binary: Path = MORK_BIN) -> None:
        self.binary = binary
        self.invocations = 0
        self.logs: List[str] = []

    def run(self, program: Path, dump: Path, steps: int = 200) -> str:
        if not self.binary.exists():
            raise FileNotFoundError(f"MORK binary missing: {self.binary}")
        self.invocations += 1
        result = subprocess.run(
            [str(self.binary), "run", "--steps", str(steps), str(program), str(dump)],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        output = result.stdout + result.stderr
        self.logs.append(output)
        if result.returncode != 0:
            raise RuntimeError(
                f"MORK failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
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
    missing = [phase for phase in PHASE_ORDER if phase not in phases]
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


def barrier_fact(phase: str, next_phase: str | None) -> str:
    symbol = phase_symbol(phase)
    if next_phase is None:
        first = phase_symbol(PHASE_ORDER[0])
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


def build_single_invocation_program(facts: str, phases: Dict[str, str]) -> str:
    phase_facts = [phase_fact(phase, phases[phase]) for phase in PHASE_ORDER]
    barrier_facts = []
    for index, phase in enumerate(PHASE_ORDER):
        next_phase = PHASE_ORDER[index + 1] if index + 1 < len(PHASE_ORDER) else None
        barrier_facts.append(barrier_fact(phase, next_phase))

    first_phase = PHASE_ORDER[0]
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


def load_weights(path: Path) -> Weights:
    data = np.load(path)
    return Weights(data["initial_wxh"].astype(np.float32), data["initial_why"].astype(np.float32))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle" / "xor_jpc_reference.npz")
    parser.add_argument("--steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--compare-legacy", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    weights = load_weights(args.oracle)
    single = run_single_invocation_settle(weights, args.steps)
    report = {
        "mork_bin": str(MORK_BIN),
        "rules": str(RULES_PATH),
        "single_invocation": {key: value for key, value in single.items() if key != "final_cells"},
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
        }
        report["legacy_vs_single_final_cells"] = {
            "max_abs": legacy_abs,
            "max_rel": legacy_rel,
        }
    report_path = SCRATCH_DIR / "mork_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

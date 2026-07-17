from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
sys.path.insert(0, str(ROOT))

from driver import (  # noqa: E402
    final_cell_payload,
    max_abs_rel,
    parse_cells,
    parse_train_history,
    parse_weights,
    relation_table,
    weights_table,
)
from oracle.common import SETTLE_STEPS, Weights  # noqa: E402


def run(cmd: list[str]) -> str:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.stdout


def assert_table_within(table: dict[str, dict[str, float]], tolerance: float, label: str) -> None:
    bad = {
        key: value
        for key, value in table.items()
        if value["max_abs"] > tolerance and value["max_rel"] > tolerance
    }
    if bad:
        raise AssertionError(f"{label} exceeded tolerance {tolerance}: {json.dumps(bad, sort_keys=True)}")


def trajectory_diff(rows: list[dict[str, float]], expected: np.ndarray) -> dict[str, float]:
    actual = np.asarray([row["sample_energy"] for row in rows], dtype=np.float32)
    return dict(zip(("max_abs", "max_rel"), max_abs_rel(expected.astype(np.float32), actual), strict=True))


def assert_loaded_counts(run_report: dict, label: str) -> None:
    counts = run_report.get("loaded_counts")
    if not counts or not all(isinstance(count, int) and count > 0 for count in counts):
        raise AssertionError(f"{label} did not report positive loaded-expression counts: {counts}")
    if run_report.get("mork_invocations") != len(counts):
        raise AssertionError(f"{label} loaded-count length does not match invocation count")


def load_weights_pair(data: np.lib.npyio.NpzFile, prefix: str) -> Weights:
    return Weights(data[f"{prefix}_wxh"].astype(np.float32), data[f"{prefix}_why"].astype(np.float32))


def main() -> None:
    run([str(PYTHON), str(ROOT / "oracle" / "jpc_ref.py")])
    run([str(PYTHON), str(ROOT / "oracle" / "epc_ref.py")])
    run([str(PYTHON), str(ROOT / "oracle" / "ipc_ref.py")])
    run([str(PYTHON), str(ROOT / "driver.py"), "--compare-legacy", "--train"])
    report = json.loads((ROOT / "scratch" / "mork_gate_report.json").read_text())
    single = report["single_invocation"]
    legacy = report["legacy_reference"]
    training = report["training"]
    jpc_npz = np.load(ROOT / "oracle" / "xor_jpc_reference.npz")
    ipc_npz = np.load(ROOT / "oracle" / "xor_ipc_reference.npz")

    max_final_abs = max(value["max_abs"] for value in single["final_iteration_gate"].values())
    max_final_rel = max(value["max_rel"] for value in single["final_iteration_gate"].values())
    max_energy_abs = max(row["energy_abs"] for row in single["energy_trajectory"])
    max_energy_rel = max(row["energy_rel"] for row in single["energy_trajectory"])
    max_history_abs = max(row["history_cell_abs"] for row in single["energy_trajectory"])
    max_history_rel = max(row["history_cell_rel"] for row in single["energy_trajectory"])
    max_legacy_abs = report["legacy_vs_single_final_cells"]["max_abs"]
    assert single["mork_invocations"] == 1
    assert legacy["mork_invocations"] > single["mork_invocations"]
    assert single["history_counts"] == {"pceh": 64, "pcsh": 64}
    assert max_final_abs <= 5e-5 or max_final_rel <= 5e-5
    assert max_energy_abs <= 5e-5 or max_energy_rel <= 5e-5
    assert max_history_abs <= 5e-5 or max_history_rel <= 5e-5
    assert max_legacy_abs <= 5e-7

    m1_one = training["m1_one_step"]
    m1_one_dump = Path(m1_one["dump"])
    one_cells = final_cell_payload(parse_cells(m1_one_dump))
    expected_one = {key.removeprefix("local_m1_first_"): value for key, value in jpc_npz.items() if key.startswith("local_m1_first_")}
    q1_cells = relation_table(expected_one, one_cells)
    q1_weights = weights_table(load_weights_pair(jpc_npz, "local_m1_after_one"), parse_weights(m1_one_dump))
    assert_table_within(q1_cells, 5e-5, "q1 m1 one-step cells")
    assert_table_within(q1_weights, 5e-5, "q1 m1 one-step weights")
    q1_energy_abs, q1_energy_rel = max_abs_rel(
        np.asarray([jpc_npz["local_m1_first_energy"]], dtype=np.float32),
        np.asarray([m1_one["sample_energy_trajectory"][0]["sample_energy"]], dtype=np.float32),
    )
    assert q1_energy_abs <= 5e-5 or q1_energy_rel <= 5e-5

    m1 = training["m1"]
    m1_updates = int(jpc_npz["local_m1_train_criterion_update"])
    assert m1["updates"] == m1_updates
    assert m1["mork_invocations"] == 1
    assert_loaded_counts(m1, "q2 m1")
    assert m1["final_batch_energy"] <= float(jpc_npz["local_m1_train_criterion"]) + 5e-6
    q2_weights = weights_table(load_weights_pair(jpc_npz, "local_m1_train_final"), parse_weights(Path(m1["dump"])))
    assert_table_within(q2_weights, 5e-5, "q2 m1 final weights")
    q2_loss = trajectory_diff(m1["sample_energy_trajectory"], jpc_npz["local_m1_train_sample_energy"])
    assert q2_loss["max_abs"] <= 5e-5 or q2_loss["max_rel"] <= 5e-5

    m2 = training["m2"]
    m2_updates = int(ipc_npz["train_criterion_update"])
    assert m2["updates"] == m2_updates
    assert m2["mork_invocations"] == 1
    assert_loaded_counts(m2, "q3 m2")
    assert m2["final_batch_energy"] <= float(ipc_npz["train_criterion"]) + 5e-6
    q3_weights = weights_table(load_weights_pair(ipc_npz, "train_final"), parse_weights(Path(m2["dump"])))
    assert_table_within(q3_weights, 5e-5, "q3 m2 final weights")
    q3_loss = trajectory_diff(m2["sample_energy_trajectory"], ipc_npz["train_sample_energy"])
    assert q3_loss["max_abs"] <= 5e-5 or q3_loss["max_rel"] <= 5e-5
    assert m2_updates != m1_updates

    assert_loaded_counts(single, "q4 single settle")
    assert legacy["loaded_count_summary"]["count"] == legacy["mork_invocations"]
    assert legacy["loaded_count_summary"]["min"] > 0
    assert legacy["loaded_count_summary"]["max"] > 0

    jscpd = run(
        [
            "jscpd",
            "--reporters",
            "ai",
            "--ignore",
            ".venv/**,scratch/**,__pycache__/**,oracle/__pycache__/**,rules/*.handwritten",
            ".",
        ]
    )
    assert "0 clones" in jscpd

    print(
        json.dumps(
            {
                "q1_energy_abs": q1_energy_abs,
                "q1_energy_rel": q1_energy_rel,
                "q1_weight_max_abs": max(value["max_abs"] for value in q1_weights.values()),
                "q2_loss_max_abs": q2_loss["max_abs"],
                "q2_m1_updates": m1_updates,
                "q2_m1_wall_seconds": m1["wall_seconds"][0],
                "q3_loss_max_abs": q3_loss["max_abs"],
                "q3_m2_updates": m2_updates,
                "q3_m2_wall_seconds": m2["wall_seconds"][0],
                "legacy_mork_invocations": legacy["mork_invocations"],
                "max_energy_abs": max_energy_abs,
                "max_energy_rel": max_energy_rel,
                "max_final_abs": max_final_abs,
                "max_final_rel": max_final_rel,
                "max_history_abs": max_history_abs,
                "max_history_rel": max_history_rel,
                "max_legacy_final_abs": max_legacy_abs,
                "single_mork_invocations": single["mork_invocations"],
                "single_loaded_counts": single["loaded_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

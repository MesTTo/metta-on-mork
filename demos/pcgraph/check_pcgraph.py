from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"


def run(cmd: list[str]) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)


def main() -> None:
    run([str(PYTHON), str(ROOT / "oracle" / "jpc_ref.py")])
    run([str(PYTHON), str(ROOT / "oracle" / "epc_ref.py")])
    run([str(PYTHON), str(ROOT / "driver.py"), "--compare-legacy"])
    report = json.loads((ROOT / "scratch" / "mork_gate_report.json").read_text())
    single = report["single_invocation"]
    legacy = report["legacy_reference"]
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
    print(
        json.dumps(
            {
                "legacy_mork_invocations": legacy["mork_invocations"],
                "max_energy_abs": max_energy_abs,
                "max_energy_rel": max_energy_rel,
                "max_final_abs": max_final_abs,
                "max_final_rel": max_final_rel,
                "max_history_abs": max_history_abs,
                "max_history_rel": max_history_rel,
                "max_legacy_final_abs": max_legacy_abs,
                "single_mork_invocations": single["mork_invocations"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

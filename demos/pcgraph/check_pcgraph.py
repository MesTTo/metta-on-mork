from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)


def main() -> None:
    run([str(PYTHON), str(ROOT / "oracle" / "jpc_ref.py")])
    run([str(PYTHON), str(ROOT / "oracle" / "epc_ref.py")])
    run([str(PYTHON), str(ROOT / "driver.py")])
    report = json.loads((ROOT / "scratch" / "mork_gate_report.json").read_text())
    max_first_abs = max(value["max_abs"] for value in report["first_iteration_gate"].values())
    max_first_rel = max(value["max_rel"] for value in report["first_iteration_gate"].values())
    max_energy_cell_abs = max(row["max_cell_abs"] for row in report["energy_trajectory"])
    assert max_first_abs <= 5e-5 or max_first_rel <= 5e-5
    assert max_energy_cell_abs <= 5e-5
    print(
        json.dumps(
            {
                "max_first_iteration_abs": max_first_abs,
                "max_first_iteration_rel": max_first_rel,
                "max_settle_cell_abs": max_energy_cell_abs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

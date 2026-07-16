from __future__ import annotations

import json

import numpy as np

from common import (
    LOCAL_TRAIN_CRITERION,
    LOCAL_TRAIN_MAX_UPDATES,
    ORACLE_DIR,
    SETTLE_STEPS,
    load_jpc_reference,
    local_m2_update,
    local_train_to_criterion,
    save_npz,
)


def main() -> None:
    _jpc_npz, initial = load_jpc_reference()
    after_one, first_step, first_delta = local_m2_update(initial, 0)
    m2 = local_train_to_criterion(initial, "m2")

    out = ORACLE_DIR / "xor_ipc_reference.npz"
    save_npz(
        out,
        settle_steps=np.asarray(SETTLE_STEPS, dtype=np.int64),
        ipc_train_criterion=np.asarray(LOCAL_TRAIN_CRITERION, dtype=np.float32),
        ipc_train_max_updates=np.asarray(LOCAL_TRAIN_MAX_UPDATES, dtype=np.int64),
        **after_one.as_npz("after_one_update"),
        **{f"first_{key}": value for key, value in first_step.items()},
        **{f"first_{key}": value for key, value in first_delta.items()},
        **{f"train_{key}": value for key, value in m2.items()},
    )
    report_path = ORACLE_DIR / "ipc_report.json"
    report_path.write_text(
        json.dumps(
            {
                "reference": "store-native iPC",
                "rule": "same local outer-product fold as m1, applied after every error tick",
                "criterion_update": int(m2["criterion_update"]),
                "final_batch_energy": float(m2["batch_energy"][-1]),
                "output": str(out),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {out}")
    print(json.dumps({"ipc_criterion_update": int(m2["criterion_update"])}, sort_keys=True))


if __name__ == "__main__":
    main()

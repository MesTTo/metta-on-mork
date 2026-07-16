from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jpc
import numpy as np
import optax

from common import (
    ERROR_LR,
    HIDDEN_DIM,
    INPUT_DIM,
    ORACLE_DIR,
    OUTPUT_DIM,
    SEED,
    SETTLE_STEPS,
    TRAIN_STEPS,
    WEIGHT_LR,
    LOCAL_TRAIN_CRITERION,
    LOCAL_TRAIN_MAX_UPDATES,
    LOCAL_WEIGHT_LR,
    X_SINGLE,
    X_TRAIN,
    Y_SINGLE,
    Y_TRAIN,
    Weights,
    jpc_native_settle,
    local_m1_update,
    local_train_to_criterion,
    max_abs_rel,
    save_npz,
)


def make_model(seed: int = SEED):
    key = jax.random.PRNGKey(seed)
    return jpc.make_mlp(
        key,
        INPUT_DIM,
        HIDDEN_DIM,
        2,
        OUTPUT_DIM,
        "tanh",
        False,
        "sp",
    )


def extract_weights(model) -> Weights:
    return Weights(
        wxh=np.asarray(model[0].layers[1].weight, dtype=np.float32),
        why=np.asarray(model[1].layers[1].weight, dtype=np.float32),
    )


def jpc_batch(value: np.ndarray):
    return jnp.asarray(value, dtype=jnp.float32)


def epc_call_options(x: np.ndarray) -> dict:
    return {"input": jpc_batch(x), "loss_id": "mse", "param_type": "sp"}


def settle_with_jpc(model, x: np.ndarray, y: np.ndarray):
    errors = jpc.init_epc_errors([INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM], x.shape[0], "supervised")
    optim = optax.sgd(float(ERROR_LR))
    opt_state = optim.init(errors)
    energies = []
    errors_after = []
    grads = []
    for _ in range(SETTLE_STEPS):
        result = jpc.update_epc_errors(
            (model, None),
            errors,
            optim,
            opt_state,
            jpc_batch(y),
            **epc_call_options(x),
        )
        energies.append(np.asarray(result["energy"], dtype=np.float32))
        grads.append([np.asarray(grad, dtype=np.float32) for grad in result["grads"]])
        errors = result["errors"]
        opt_state = result["opt_state"]
        errors_after.append([np.asarray(error, dtype=np.float32) for error in errors])
    return energies, errors_after, grads, errors


def update_params_once(model, x: np.ndarray, y: np.ndarray):
    _, _, _, errors = settle_with_jpc(model, x, y)
    optim = optax.sgd(float(WEIGHT_LR))
    opt_state = optim.init((model, None))
    result = jpc.update_epc_params(
        (model, None),
        errors,
        optim,
        opt_state,
        jpc_batch(y),
        **epc_call_options(x),
    )
    return result["model"]


def train_steps(model, steps: int = TRAIN_STEPS):
    for _ in range(steps):
        model = update_params_once(model, X_TRAIN, Y_TRAIN)
    return model


def main() -> None:
    model = make_model()
    initial = extract_weights(model)
    manual = jpc_native_settle(initial, X_SINGLE, Y_SINGLE)
    jpc_energies, jpc_errors_after, jpc_grads, _ = settle_with_jpc(model, X_SINGLE, Y_SINGLE)

    jpc_energy = np.stack(jpc_energies).astype(np.float32)
    jpc_pce_h_after = np.stack([pair[0] for pair in jpc_errors_after]).astype(np.float32)
    jpc_pce_y_after = np.stack([pair[1] for pair in jpc_errors_after]).astype(np.float32)
    jpc_pcg_h = np.stack([pair[0] for pair in jpc_grads]).astype(np.float32)
    jpc_pcg_y = np.stack([pair[1] for pair in jpc_grads]).astype(np.float32)

    checks = {
        "energy": max_abs_rel(manual["energy"], jpc_energy),
        "pce_h_after": max_abs_rel(manual["pce_h_after"], jpc_pce_h_after),
        "pce_y_after": max_abs_rel(manual["pce_y_after"], jpc_pce_y_after),
        "pcg_h": max_abs_rel(manual["pcg_h"], jpc_pcg_h),
        "pcg_y": max_abs_rel(manual["pcg_y"], jpc_pcg_y),
    }
    max_abs = max(value[0] for value in checks.values())
    if max_abs > 5e-6:
        raise AssertionError(f"manual jpc-native trace does not match jpc: {checks}")

    one_update_model = update_params_once(model, X_TRAIN, Y_TRAIN)
    after_one = extract_weights(one_update_model)
    after_t50 = extract_weights(train_steps(model, TRAIN_STEPS))
    local_after_one, local_step, local_delta = local_m1_update(initial, 0)
    local_m1 = local_train_to_criterion(initial, "m1")

    out = ORACLE_DIR / "xor_jpc_reference.npz"
    save_npz(
        out,
        seed=np.asarray(SEED, dtype=np.int64),
        error_lr=np.asarray(ERROR_LR, dtype=np.float32),
        weight_lr=np.asarray(WEIGHT_LR, dtype=np.float32),
        local_weight_lr=np.asarray(LOCAL_WEIGHT_LR, dtype=np.float32),
        local_train_criterion=np.asarray(LOCAL_TRAIN_CRITERION, dtype=np.float32),
        local_train_max_updates=np.asarray(LOCAL_TRAIN_MAX_UPDATES, dtype=np.int64),
        settle_steps=np.asarray(SETTLE_STEPS, dtype=np.int64),
        train_steps=np.asarray(TRAIN_STEPS, dtype=np.int64),
        x_single=X_SINGLE,
        y_single=Y_SINGLE,
        x_train=X_TRAIN,
        y_train=Y_TRAIN,
        **initial.as_npz("initial"),
        **after_one.as_npz("after_one_update"),
        **after_t50.as_npz("after_t50"),
        **local_after_one.as_npz("local_m1_after_one"),
        **{f"local_m1_first_{key}": value for key, value in local_step.items()},
        **{f"local_m1_first_{key}": value for key, value in local_delta.items()},
        **{f"local_m1_train_{key}": value for key, value in local_m1.items()},
        **{f"settle_{key}": value for key, value in manual.items()},
    )
    report_path = ORACLE_DIR / "jpc_report.json"
    report_path.write_text(
        json.dumps(
            {
                "reference": "jpc native ePC",
                "jpc_source": str(Path("/home/user/Dev/jpc").resolve()),
                "local_learning_rule": (
                    "m1 uses jpc-native settling, then the store-native local outer product "
                    "dW = eps_post x phi(pre) once at settle end."
                ),
                "local_m1_criterion_update": int(local_m1["criterion_update"]),
                "local_m1_final_batch_energy": float(local_m1["batch_energy"][-1]),
                "topology": "2-2-2 one-hot XOR, matching jpc supervised example output conventions",
                "manual_vs_jpc": {
                    key: {"max_abs": value[0], "max_rel": value[1]}
                    for key, value in checks.items()
                },
                "output": str(out),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {out}")
    print(json.dumps({key: value[0] for key, value in checks.items()}, sort_keys=True))


if __name__ == "__main__":
    main()

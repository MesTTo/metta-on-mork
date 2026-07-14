from __future__ import annotations

import json

import numpy as np
import torch

from common import (
    ERROR_LR,
    HIDDEN_DIM,
    JPC_NATIVE_TRACE_KEYS,
    ORACLE_DIR,
    OUTPUT_DIM,
    SETTLE_STEPS,
    TRAIN_STEPS,
    WEIGHT_LR,
    X_SINGLE,
    X_TRAIN,
    Y_SINGLE,
    Y_TRAIN,
    Weights,
    append_trace,
    jpc_native_settle,
    make_trace,
    max_abs_rel,
    paper_error_settle,
    save_npz,
    stack_trace,
)


def torch_jpc_native_step(weights: Weights, x_np: np.ndarray, y_np: np.ndarray, e_h_np: np.ndarray, e_y_np: np.ndarray):
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    wxh = torch.tensor(weights.wxh, dtype=torch.float32)
    why = torch.tensor(weights.why, dtype=torch.float32)
    e_h = torch.tensor(e_h_np, dtype=torch.float32, requires_grad=True)
    e_y = torch.tensor(e_y_np, dtype=torch.float32, requires_grad=True)
    pre_h = x @ wxh.T
    pcs_h = pre_h + e_h
    phi_h = torch.tanh(pcs_h)
    pcs_y = phi_h @ why.T - e_y
    energy = 0.5 * torch.sum(e_h * e_h) / x.shape[0] + 0.5 * torch.sum((y - pcs_y) ** 2) / x.shape[0]
    energy.backward()
    with torch.no_grad():
        next_e_h = e_h - float(ERROR_LR) * e_h.grad
        next_e_y = e_y - float(ERROR_LR) * e_y.grad
        residual = y - pcs_y
        pcb_h = residual @ why / x.shape[0]
    return {
        "pre_h": pre_h.detach().numpy().astype(np.float32),
        "pcs_h": pcs_h.detach().numpy().astype(np.float32),
        "phi_h": phi_h.detach().numpy().astype(np.float32),
        "pre_y": (phi_h @ why.T).detach().numpy().astype(np.float32),
        "pcs_y": pcs_y.detach().numpy().astype(np.float32),
        "residual": residual.detach().numpy().astype(np.float32),
        "energy": np.asarray(energy.detach().numpy(), dtype=np.float32),
        "pcb_h": pcb_h.detach().numpy().astype(np.float32),
        "pcg_h": e_h.grad.detach().numpy().astype(np.float32),
        "pcg_y": e_y.grad.detach().numpy().astype(np.float32),
        "pce_h_after": next_e_h.detach().numpy().astype(np.float32),
        "pce_y_after": next_e_y.detach().numpy().astype(np.float32),
    }


def torch_jpc_native_settle(weights: Weights, x: np.ndarray, y: np.ndarray):
    e_h = np.zeros((x.shape[0], HIDDEN_DIM), dtype=np.float32)
    e_y = np.zeros((x.shape[0], OUTPUT_DIM), dtype=np.float32)
    traces = make_trace(JPC_NATIVE_TRACE_KEYS)
    for _ in range(SETTLE_STEPS):
        append_trace(traces, pce_h_before=e_h, pce_y_before=e_y)
        step = torch_jpc_native_step(weights, x, y, e_h, e_y)
        append_trace(traces, **step)
        e_h = step["pce_h_after"]
        e_y = step["pce_y_after"]
    return stack_trace(traces)


def torch_update_params_once(weights: Weights, x_np: np.ndarray, y_np: np.ndarray) -> Weights:
    settled = torch_jpc_native_settle(weights, x_np, y_np)
    e_h_np = settled["pce_h_after"][-1]
    e_y_np = settled["pce_y_after"][-1]
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    wxh = torch.tensor(weights.wxh, dtype=torch.float32, requires_grad=True)
    why = torch.tensor(weights.why, dtype=torch.float32, requires_grad=True)
    e_h = torch.tensor(e_h_np, dtype=torch.float32)
    e_y = torch.tensor(e_y_np, dtype=torch.float32)
    pcs_h = x @ wxh.T + e_h
    pcs_y = torch.tanh(pcs_h) @ why.T - e_y
    energy = 0.5 * torch.sum(e_h * e_h) / x.shape[0] + 0.5 * torch.sum((y - pcs_y) ** 2) / x.shape[0]
    energy.backward()
    with torch.no_grad():
        next_wxh = wxh - float(WEIGHT_LR) * wxh.grad
        next_why = why - float(WEIGHT_LR) * why.grad
    return Weights(next_wxh.numpy().astype(np.float32), next_why.numpy().astype(np.float32))


def torch_train_steps(weights: Weights) -> Weights:
    current = weights
    for _ in range(TRAIN_STEPS):
        current = torch_update_params_once(current, X_TRAIN, Y_TRAIN)
    return current


def main() -> None:
    jpc_npz = np.load(ORACLE_DIR / "xor_jpc_reference.npz")
    initial = Weights(
        wxh=jpc_npz["initial_wxh"].astype(np.float32),
        why=jpc_npz["initial_why"].astype(np.float32),
    )
    torch_native = torch_jpc_native_settle(initial, X_SINGLE, Y_SINGLE)
    numpy_native = jpc_native_settle(initial, X_SINGLE, Y_SINGLE)
    paper = paper_error_settle(initial, X_SINGLE, Y_SINGLE)

    native_checks = {
        key: max_abs_rel(numpy_native[key], torch_native[key])
        for key in ("energy", "pcs_h", "pcs_y", "pcb_h", "pcg_h", "pcg_y", "pce_h_after", "pce_y_after")
    }
    jpc_checks = {
        key: max_abs_rel(jpc_npz[f"settle_{key}"], torch_native[key])
        for key in ("energy", "pcs_h", "pcs_y", "pcb_h", "pcg_h", "pcg_y", "pce_h_after", "pce_y_after")
    }
    paper_checks = {
        "energy": max_abs_rel(jpc_npz["settle_energy"], paper["energy"]),
        "pcs_h": max_abs_rel(jpc_npz["settle_pcs_h"], paper["pcs_h"]),
        "pcs_y": max_abs_rel(jpc_npz["settle_pcs_y"], paper["pcs_y"]),
        "pce_h_after": max_abs_rel(jpc_npz["settle_pce_h_after"], paper["pce_h_after"]),
    }

    after_one = torch_update_params_once(initial, X_TRAIN, Y_TRAIN)
    after_t50 = torch_train_steps(initial)
    weight_checks = {
        "after_one_wxh": max_abs_rel(jpc_npz["after_one_update_wxh"], after_one.wxh),
        "after_one_why": max_abs_rel(jpc_npz["after_one_update_why"], after_one.why),
        "after_t50_wxh": max_abs_rel(jpc_npz["after_t50_wxh"], after_t50.wxh),
        "after_t50_why": max_abs_rel(jpc_npz["after_t50_why"], after_t50.why),
    }
    max_native_abs = max(value[0] for value in {**native_checks, **jpc_checks, **weight_checks}.values())
    if max_native_abs > 5e-5:
        raise AssertionError(
            "torch jpc-native reference does not match jpc within fp32 tolerance: "
            f"{json.dumps({k: v[0] for k, v in {**native_checks, **jpc_checks, **weight_checks}.items()}, sort_keys=True)}"
        )

    out = ORACLE_DIR / "xor_error_based_reference.npz"
    save_npz(
        out,
        **{f"torch_native_{key}": value for key, value in torch_native.items()},
        **{f"paper_{key}": value for key, value in paper.items()},
        **after_one.as_npz("torch_after_one_update"),
        **after_t50.as_npz("torch_after_t50"),
    )
    report_path = ORACLE_DIR / "oracle_comparison.json"
    report_path.write_text(
        json.dumps(
            {
                "jpc_vs_torch_native": {
                    key: {"max_abs": value[0], "max_rel": value[1]}
                    for key, value in {**native_checks, **jpc_checks, **weight_checks}.items()
                },
                "jpc_native_vs_error_based_pc_paper_path": {
                    key: {"max_abs": value[0], "max_rel": value[1]}
                    for key, value in paper_checks.items()
                },
                "interpretation": (
                    "jpc native supervised ePC initializes a hidden and an output error variable. "
                    "error_based_PC/PCE optimizes hidden errors only and keeps the output residual inside the loss."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {out}")
    print(json.dumps({"max_jpc_vs_torch_native_abs": max_native_abs}, sort_keys=True))
    print(json.dumps({"paper_path_energy_max_abs": paper_checks["energy"][0]}, sort_keys=True))


if __name__ == "__main__":
    main()

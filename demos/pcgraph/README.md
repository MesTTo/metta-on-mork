# PC Graph Demo

This directory covers Stage 1 chunks 1 through 3.

The verified network is a fixed `2-2-2` XOR chain with one-hot outputs, tanh on the hidden state, squared-error output loss, `error_lr = 0.1`, and `K = 16` inner error updates. The `2-2-2` shape follows jpc's supervised ePC convention, where output targets are vectors and `init_epc_errors([2, 2, 2], batch, "supervised")` creates hidden and output error variables.

`oracle/xor_jpc_reference.npz` is the primary interchange artifact. It stores the seeded initial weights, one-sample settle traces for all `K` inner steps, the energy trajectory, weights after one full-XOR weight update, and weights after `T = 50` full-XOR training updates.

`oracle/xor_error_based_reference.npz` is the Torch secondary artifact. It stores a Torch implementation of jpc-native ePC and a separate `error_based_PC` paper path. The paper path matches `PCE.E` and `PCE.y_pred` from `/home/user/Dev/error_based_PC`: only hidden errors are optimized, and the output residual stays inside the loss. That path diverges from jpc-native after the first step because jpc also updates an output error variable. The MORK gate follows jpc-native because chunk 1 makes jpc the primary reference.

`driver.py` emits cells using the design schemas:

- `(pcw <edge> i j v)`
- `(pcs <node> i v)`
- `(pce <node> i v)`
- `(pcin <node> i v)`
- `(pcb <node> i v)`
- `(pcg <node> i v)`
- `(pchp <name> v)`
- `(pcsh <tick> <node> i v)` for tick-indexed state history
- `(pceh <tick> <node> i v)` for tick-indexed error history

The generated run files also contain fixed-chain helper relations such as `(wxh i j v)` and `(phix i v)` so the current tensor-op sink can consume dense tensors without symbolic edge labels. The public cells remain the gate surface.

The MORK rules in `rules/xor_tick.mm2` use tensor-op sinks for the matrix-vector products and `pure` f32 calls for `tanh`, `tanh'`, scalar add, subtraction, and scaled error updates. The default driver path now emits one program and runs MORK once. It turns the checked-in phase blocks into phase facts, arms one `(exec (pcphase ...))`, and lets `(exec (quiesce ...))` barriers re-arm the next phase after the lower phase has stopped changing the store. The final barrier uses the in-store `pcnext` relation to advance `pctick`; when there is no successor for tick 15, the run ends with all 16 history rows in the dump.

The old per-phase/per-tick path is still available through `driver.py --compare-legacy`. The checker uses it as a reference and asserts that its final public cells match the single-invocation path.

The chunk-3 run was verified with a local copy of the MORK binary built from `/home/user/Dev/mork-integration/MORK` on branch `einsum-port` at `63c4d213130e4444ace590d88300e29b748d2a16`, with features `einsum,leapfrog,bulk_emit,factorized_aggregate,stratified_quiescence` and `RUSTFLAGS='-C target-cpu=native -Awarnings'`. Before the pcgraph gate, that feature combination produced the required process-calculus dump SHA:

```text
f88e8253c947b4f986d4c6a4acd40448408fdbf6effc3430174a67dccaca685e
```

The copied binary lives under `scratch/bin/`, which is ignored. A shared `target/release/mork` does not need to keep the stratified feature set after the copy.

Run the checked path with:

```bash
demos/pcgraph/.venv/bin/python demos/pcgraph/check_pcgraph.py
```

The check regenerates oracle artifacts, runs the single-invocation settle, runs the legacy reference path, and writes `scratch/mork_gate_report.json`. On the verified run, the single path used 1 MORK invocation, the legacy path used 528, `pcsh` and `pceh` each contained 64 cells, and the final-cell max absolute difference between the two paths was `0.0`. The report contains the full 16-row energy table computed from the in-store history.

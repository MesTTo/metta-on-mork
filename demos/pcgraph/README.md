# PC Graph Demo

This directory covers Stage 1 chunks 1 and 2 only.

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

The generated run files also contain fixed-chain helper relations such as `(wxh i j v)` and `(phix i v)` so the current tensor-op sink can consume dense tensors without symbolic edge labels. The public cells remain the gate surface.

The MORK rules in `rules/xor_tick.mm2` use tensor-op sinks for the matrix-vector products. They use `pure` f32 calls for `tanh`, `tanh'`, scalar add, subtraction, and scaled error updates because the current binary was built without `stratified_quiescence`, and because the ported sink tests expose only the matrix-style `add` form. The driver therefore runs the hand-written phase blocks in order, with each dumped store state becoming the input to the next phase.

Run the checked path with:

```bash
demos/pcgraph/.venv/bin/python demos/pcgraph/check_pcgraph.py
```

The check regenerates oracle artifacts, runs one MORK tick for each of the `K` settle iterations, verifies the iteration-1 cell gate relation by relation, and writes `scratch/mork_gate_report.json`.

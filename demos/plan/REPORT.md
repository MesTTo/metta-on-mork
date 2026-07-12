# 8-Puzzle MITM Planning Demonstrator

This directory builds a STRIPS-style sliding-tile planning demo on MORK. States are ground `(state c0 ... c8)` tuples and `_` is the blank. The generated MM2 uses static move and barrier rule facts. Layer depth changes through `active-*-budget`, `decFn`, `*-currentLayerFn`, and `*-nextLayerFn` facts, so round counters are not baked into transition rule bytes.

The claim tested here is layer composition on MORK: barrier staging plus joins can expose the expected meet-in-the-middle shape. This is not a claim that this MM2 encoding beats specialized 8-puzzle planners.

## Commands

```sh
python3 demos/plan/plan_oracle.py --write-instances demos/plan/instances.json
python3 demos/plan/build_plan_programs.py
python3 demos/plan/run_plan_measurements.py
```

The driver then executed these MORK commands:

```sh
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d08a-backward-d8.mm2 /home/user/Dev/metta-on-mork/demos/plan/d08a-backward-d8.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d08a-forward-d8.mm2 /home/user/Dev/metta-on-mork/demos/plan/d08a-forward-d8.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d08a-mitm-d8-f4-b4.mm2 /home/user/Dev/metta-on-mork/demos/plan/d08a-mitm-d8-f4-b4.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d08b-backward-d8.mm2 /home/user/Dev/metta-on-mork/demos/plan/d08b-backward-d8.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d08b-forward-d8.mm2 /home/user/Dev/metta-on-mork/demos/plan/d08b-forward-d8.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d08b-mitm-d8-f4-b4.mm2 /home/user/Dev/metta-on-mork/demos/plan/d08b-mitm-d8-f4-b4.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d12a-backward-d12.mm2 /home/user/Dev/metta-on-mork/demos/plan/d12a-backward-d12.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d12a-forward-d12.mm2 /home/user/Dev/metta-on-mork/demos/plan/d12a-forward-d12.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d12a-mitm-d12-f6-b6.mm2 /home/user/Dev/metta-on-mork/demos/plan/d12a-mitm-d12-f6-b6.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d12b-backward-d12.mm2 /home/user/Dev/metta-on-mork/demos/plan/d12b-backward-d12.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d12b-forward-d12.mm2 /home/user/Dev/metta-on-mork/demos/plan/d12b-forward-d12.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d12b-mitm-d12-f6-b6.mm2 /home/user/Dev/metta-on-mork/demos/plan/d12b-mitm-d12-f6-b6.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d16a-backward-d16.mm2 /home/user/Dev/metta-on-mork/demos/plan/d16a-backward-d16.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d16a-forward-d16.mm2 /home/user/Dev/metta-on-mork/demos/plan/d16a-forward-d16.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d16a-mitm-d16-f8-b8.mm2 /home/user/Dev/metta-on-mork/demos/plan/d16a-mitm-d16-f8-b8.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d20a-backward-d20.mm2 /home/user/Dev/metta-on-mork/demos/plan/d20a-backward-d20.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d20a-forward-d20.mm2 /home/user/Dev/metta-on-mork/demos/plan/d20a-forward-d20.dump
/home/user/Dev/mork-integration/MORK/target/release/mork run /home/user/Dev/metta-on-mork/demos/plan/d20a-mitm-d20-f10-b10.mm2 /home/user/Dev/metta-on-mork/demos/plan/d20a-mitm-d20-f10-b10.dump
```

The wall-time numbers below are measured by the Python driver around each command. The per-program logs are written beside the dumps.

## Instances

| instance | optimal | start state | scramble |
| --- | ---: | --- | --- |
| d08a | 8 | `(state 1 2 3 8 7 5 4 6 _)` | `ULDLURDR` |
| d08b | 8 | `(state 1 3 _ 7 2 5 8 4 6)` | `ULLDRUUR` |
| d12a | 12 | `(state 5 1 3 7 _ 6 2 4 8)` | `LLURULDRDLUR` |
| d12b | 12 | `(state 7 1 2 4 _ 3 5 8 6)` | `LULDRRUULLDR` |
| d16a | 16 | `(state _ 2 3 6 1 5 4 7 8)` | `LLURRULDLURRDLLU` |
| d20a | 20 | `(state 3 4 6 1 7 5 2 8 _)` | `LUULDRDLUURRDLLURDDR` |

## Results

| instance | oracle | forward depth | MITM split | MITM meet depth | forward ms | backward ms | MITM ms | forward states | MITM fwd states | MITM bwd states | meet facts |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d08a | 8 | 8 | 4+4 | 8 | 18.19 | 17.82 | 13.37 | 268 | 31 | 31 | 1 |
| d08b | 8 | 8 | 4+4 | 8 | 17.94 | 17.99 | 13.65 | 268 | 31 | 31 | 1 |
| d12a | 12 | 12 | 6+6 | 12 | 74.67 | 62.68 | 28.73 | 2389 | 129 | 90 | 1 |
| d12b | 12 | 12 | 6+6 | 12 | 71.85 | 72.14 | 27.01 | 2389 | 129 | 90 | 1 |
| d16a | 16 | 16 | 8+8 | 16 | 401.67 | 367.55 | 55.25 | 11764 | 268 | 268 | 2 |
| d20a | 20 | 20 | 10+10 | 20 | 1466.73 | 1429.48 | 118.51 | 54802 | 706 | 706 | 1 |

## Strata

### d08a

- Forward-only fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116}`
- Backward-only bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116}`
- MITM fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16}`
- MITM bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16}`

### d08b

- Forward-only fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116}`
- Backward-only bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116}`
- MITM fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16}`
- MITM bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16}`

### d12a

- Forward-only fwd-layer counts: `{0: 1, 1: 4, 2: 8, 3: 8, 4: 16, 5: 32, 6: 60, 7: 72, 8: 136, 9: 200, 10: 376, 11: 512, 12: 964}`
- Backward-only bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286, 11: 396, 12: 748}`
- MITM fwd-layer counts: `{0: 1, 1: 4, 2: 8, 3: 8, 4: 16, 5: 32, 6: 60}`
- MITM bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39}`

### d12b

- Forward-only fwd-layer counts: `{0: 1, 1: 4, 2: 8, 3: 8, 4: 16, 5: 32, 6: 60, 7: 72, 8: 136, 9: 200, 10: 376, 11: 512, 12: 964}`
- Backward-only bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286, 11: 396, 12: 748}`
- MITM fwd-layer counts: `{0: 1, 1: 4, 2: 8, 3: 8, 4: 16, 5: 32, 6: 60}`
- MITM bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39}`

### d16a

- Forward-only fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286, 11: 396, 12: 748, 13: 1024, 14: 1893, 15: 2512, 16: 4485}`
- Backward-only bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286, 11: 396, 12: 748, 13: 1024, 14: 1893, 15: 2512, 16: 4485}`
- MITM fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116}`
- MITM bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116}`

### d20a

- Forward-only fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286, 11: 396, 12: 748, 13: 1024, 14: 1893, 15: 2512, 16: 4485, 17: 5638, 18: 9529, 19: 10878, 20: 16993}`
- Backward-only bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286, 11: 396, 12: 748, 13: 1024, 14: 1893, 15: 2512, 16: 4485, 17: 5638, 18: 9529, 19: 10878, 20: 16993}`
- MITM fwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286}`
- MITM bwd-layer counts: `{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 20, 6: 39, 7: 62, 8: 116, 9: 152, 10: 286}`

## Branching Shape

The measured distinct layer counts are small because the 8-puzzle state graph has many duplicate paths and the guard table keeps first visits only. To show the usual search-tree intuition, the table below derives an effective `b` from the measured final forward stratum, so `b^d` equals that final layer count, then compares it with `2*b^(d/2)` at the same oracle depth.

| instance | measured b | d | b^d | 2*b^(d/2) | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| d08a | 1.8116 | 8 | 116.00 | 21.54 | 5.39 |
| d08b | 1.8116 | 8 | 116.00 | 21.54 | 5.39 |
| d12a | 1.7729 | 12 | 964.00 | 62.10 | 15.52 |
| d12b | 1.7729 | 12 | 964.00 | 62.10 | 15.52 |
| d16a | 1.6914 | 16 | 4485.00 | 133.94 | 33.49 |
| d20a | 1.6275 | 20 | 16993.00 | 260.71 | 65.18 |

## Verification

- d08a: oracle=8, forward=8, backward=8, MITM=8
- d08b: oracle=8, forward=8, backward=8, MITM=8
- d12a: oracle=12, forward=12, backward=12, MITM=12
- d12b: oracle=12, forward=12, backward=12, MITM=12
- d16a: oracle=16, forward=16, backward=16, MITM=16
- d20a: oracle=20, forward=20, backward=20, MITM=20

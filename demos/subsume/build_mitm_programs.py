#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAX_N = 26


@dataclass(frozen=True)
class Target:
    bound: int
    formula: str


TARGETS: dict[str, Target] = {
    "jarr": Target(
        13,
        "(→ (→ (→ 𝜑 𝜓) 𝜒) (→ 𝜓 𝜒))",
    ),
    "imim1": Target(
        15,
        "(→ (→ 𝜑 𝜓) (→ (→ 𝜓 𝜒) (→ 𝜑 𝜒)))",
    ),
    "loowoz": Target(
        19,
        "(→ (→ (→ 𝜑 𝜓) (→ 𝜑 𝜒)) (→ (→ 𝜓 𝜑) (→ 𝜓 𝜒)))",
    ),
    "pm2.83": Target(
        25,
        "(→ (→ 𝜑 (→ 𝜓 𝜒)) (→ (→ 𝜑 (→ 𝜒 𝜃)) (→ 𝜑 (→ 𝜓 𝜃))))",
    ),
    "loolin": Target(
        26,
        "(→ (→ (→ 𝜑 𝜓) (→ 𝜓 𝜑)) (→ 𝜓 𝜑))",
    ),
}

HF_BY_TARGET: dict[str, tuple[int, ...]] = {
    "jarr": (1, 4, 6),
    "imim1": (1, 6, 7),
    "loowoz": (6, 8),
    "pm2.83": (9, 11, 12),
    "loolin": (9, 11, 12),
}


def peano(n: int) -> str:
    out = "Z"
    for _ in range(n):
        out = f"(S {out})"
    return out


def add_table() -> str:
    lines = [
        ";; addFn A B C means A + B = C for decimal-symbol integers.",
        f";; The table is finite because this experiment uses proof sizes <= {MAX_N}.",
    ]
    for a in range(MAX_N + 1):
        for b in range(MAX_N + 1 - a):
            lines.append(f"(addFn {a} {b} {a + b})")
    return "\n".join(lines)


def gt_table() -> str:
    lines = [
        ";; gtFn A B means A > B for decimal-symbol integers.",
        ";; This keeps split's remaining-size cap as a small ground-prefixed guard.",
    ]
    for a in range(MAX_N + 1):
        for b in range(MAX_N + 1):
            if a > b:
                lines.append(f"(gtFn {a} {b})")
    return "\n".join(lines)


def arithmetic_tables() -> str:
    return "\n".join([add_table(), "", gt_table()])


def scheduler_fuel(target: Target, hf: int) -> str:
    return peano(target.bound)


def forward_fuel(hf: int) -> str:
    return peano(hf + 6)


LOAD_ARITH = """;; Load decimal <-> Peano numerals and lte from the same ACT tables used by
;; backward-via-forward/bfc-xp.mm2. Run gen-fromNumber.mm2 and gen-lte.mm2 first.
(exec (0)
      (I (ACT fromNumber (fromNumberFn $x $k)))
      (, (fromNumberFn $x $k)
         (toNumberFn $k $x)))

(exec (1)
      (I (ACT lte (lte $k $l))
         (BTM (toNumberFn $k $x))
         (BTM (toNumberFn $l $y)))
      (, (lte $x $y)))

;; Precompute increment and decrement tables for decimal integers.
(exec (1)
      (, (toNumberFn $x $y)
         (toNumberFn (S $x) $sy))
      (, (incFn $y $sy)
         (decFn $sy $y)))
"""


FWD_SEEDS = """;; Exact-size forward schematic theorem strata.
(fwd 1 (: ax₁ (→ $𝜑 (→ $𝜓 $𝜑))))
(fwd 1 (: ax₂ (→ (→ $𝜑 (→ $𝜓 $𝜒))
                 (→ (→ $𝜑 $𝜓) (→ $𝜑 $𝜒)))))
(fwd 1 (: ax₃ (→ (→ (¬ $𝜑) (¬ $𝜓))
                 (→ $𝜓 $𝜑))))
"""


BFC_RULES = """;; Turn target into initial source and run the same size-layered loop as bfc-xp.
(exec (2)
      (, (target $mps (c: $ta $tx))
         (fromNumberFn $mps $mps_n))
      (, (sol $mps 1 (c: (-> $ta $ta) I))
         (exec (2 $mps_n)
               (, (exec (2 (S $k)) $ptrn $tplt)
                  (toNumberFn (S $k) $ski)
                  (toNumberFn $k $ki))
               (, {axiom_execs}
                  {extra_execs}
                  (exec (6 0)
                        (, (sol $ski $hi (c: (-> $b $c) $f))
                           {split_guard}
                           (incFn $hi $shi))
                        (, (sol $ki $shi (c: (-> (→ $a $b) (-> $a $c)) (mpⁱ $f)))))
                  (exec (7 0) (,)
                        (, (exec (2 $k) $ptrn $tplt)))))))

;; Completion is byte-for-byte the bfc-xp target shape.
(exec (3 0 0)
      (, (target $mps (c: $ta $tx))
         (sol 0 0 (c: $ta $tx)))
      (, (final 0 0 (c: $ta $tx))))
"""


AXIOM_DISCHARGE_EXECS = """(exec (3 0)
                        (, (sol $ski $shi (c: (-> (→ $𝜑 (→ $𝜓 $𝜑)) $b) $f))
                           (decFn $shi $hi)
                           (lte $hi $ki))
                        (, (sol $ki $hi (c: $b ($f ax₁)))))
                  (exec (4 0)
                        (, (sol $ski $shi (c: (-> (→ (→ $𝜑 (→ $𝜓 $𝜒)) (→ (→ $𝜑 $𝜓) (→ $𝜑 $𝜒))) $b) $f))
                           (decFn $shi $hi)
                           (lte $hi $ki))
                        (, (sol $ki $hi (c: $b ($f ax₂)))))
                  (exec (5 0)
                        (, (sol $ski $shi (c: (-> (→ (→ (¬ $𝜑) (¬ $𝜓)) (→ $𝜓 $𝜑)) $b) $f))
                           (decFn $shi $hi)
                           (lte $hi $ki))
                        (, (sol $ki $hi (c: $b ($f ax₃)))))"""


BFC_SPLIT_GUARD = """(lte $hi $ki)"""


def capped_split_guard(hf: int) -> str:
    return f"(lte $hi $ki)\n                           (lte {hf + 1} $ski)"


FWD_CLOSURE_EXEC = """;; Forward closure. A forward ax-mp proof has size i + j + 1.
                  (exec (20 fwd-closure)
                        (, (fwd $fi (: $fp (→ $fa $fb)))
                           (addFn $fi $fj $fs_minus_one)
                           (incFn $fs_minus_one $fs)
                           (lte $fs {hf})
                           (fwd $fj (: $xp $fa)))
                        (, (fwd $fs (: (ax-mp $fp $xp) $fb))))
                  """


ABOVE_CAP_MEET_EXEC_TEMPLATE = """;; Above the cap, fwd K=1 is the axiom-discharge meet.
                  (exec ({exec_id} meet-above-cap)
                        (, (sol $ski $shi (c: (-> $ma $mb) $mf))
                           (gtFn $ski {hf})
                           (addFn $meet_remaining 1 $ski)
                           (decFn $shi $meet_h)
                           (lte $meet_h $meet_remaining)
                           (fwd 1 (: $mp $ma)))
                        (, (sol $meet_remaining $meet_h (c: $mb ($mf $mp)))))"""


TAIL_MEET_EXEC_TEMPLATE = """;; At or below the cap, close the tail with any forward theorem that fits.
                  (exec ({exec_id} meet-tail)
                        (, (sol $ski $shi (c: (-> $ma $mb) $mf))
                           (lte $ski {hf})
                           (lte $mk $ski)
                           (addFn $meet_remaining $mk $ski)
                           (decFn $shi $meet_h)
                           (lte $meet_h $meet_remaining)
                           (fwd $mk (: $mp $ma)))
                        (, (sol $meet_remaining $meet_h (c: $mb ($mf $mp)))))"""


def meet_execs(hf: int) -> list[str]:
    return [
        ABOVE_CAP_MEET_EXEC_TEMPLATE.format(exec_id=30, hf=hf),
        TAIL_MEET_EXEC_TEMPLATE.format(exec_id=31, hf=hf),
    ]


SPLIT_EXEC_TEMPLATE = """;; Backward mpⁱ split. Splits stop above the forward cap only.
                  (exec (40 split)
                        (, (sol $ski $hi (c: (-> $b $c) $f))
                           (gtFn $ski {hf})
                           (decFn $ski $ki)
                           (lte $hi $ki)
                           (incFn $hi $shi))
                        (, (sol $ki $shi (c: (-> (→ $a $b) (-> $a $c)) (mpⁱ $f)))))"""


COMPLETION_EXEC = """;; Completion is the bfc-xp target shape.
                  (exec (50 complete)
                        (, (target $mps (c: $ta $tx))
                           (sol 0 0 (c: $ta $tx)))
                        (, (final 0 0 (c: $ta $tx))))"""


def split_exec(hf: int) -> str:
    return SPLIT_EXEC_TEMPLATE.format(hf=hf)


def static_scheduler(fuel: str, execs: list[str]) -> str:
    body = "\n".join(execs)
    return f"""(exec (99 {fuel})
               (, (exec (99 (S $sched_k)) $sched_ptrn $sched_tplt))
               (, {body}
                  ;; Respawn the same scheduler body with one less fuel tick.
                  (exec (99 $sched_k) $sched_ptrn $sched_tplt)))"""


def scheduled_mitm_rules(hf: int, include_forward_closure: bool) -> list[str]:
    execs: list[str] = []
    if include_forward_closure:
        execs.append(FWD_CLOSURE_EXEC.format(hf=hf))
    execs.extend([*meet_execs(hf), split_exec(hf), COMPLETION_EXEC])
    return execs


def program_header(kind: str, target_name: str, target: Target, hf: int | None) -> str:
    hf_line = "" if hf is None else f"\n(forward-bound {hf})"
    hf_comment = "" if hf is None else f";; Forward bound Hf: {hf}\n"
    return (
        f";; Generated by build_mitm_programs.py.\n"
        f";; Kind: {kind}\n"
        f";; Target: {target_name}, exact proof size {target.bound}\n"
        f"{hf_comment}"
        f"(target {target.bound} (c: {target.formula} $x))"
        f"{hf_line}\n\n"
    )


def render_bfc_program(target_name: str, target: Target) -> str:
    return "\n".join(
        [
            program_header("bfc", target_name, target, None),
            LOAD_ARITH,
            "",
            BFC_RULES.format(
                axiom_execs=AXIOM_DISCHARGE_EXECS,
                split_guard=BFC_SPLIT_GUARD,
                extra_execs="",
            ),
            "",
        ]
    )


def scheduled_seed(target: Target, hf: int, include_forward_closure: bool) -> str:
    fuel = scheduler_fuel(target, hf)
    scheduler = static_scheduler(
        "$init_fuel", scheduled_mitm_rules(hf, include_forward_closure)
    )
    return f"""(scheduler-fuel {fuel})

;; Seed the backward table and install one static scheduler. The Peano fuel lives
;; only in the scheduler exec location; transition rule bodies contain no round
;; constants and are re-emitted byte-identically each cycle.
(exec (2 init)
      (, (target $mps (c: $ta $tx))
         (scheduler-fuel $init_fuel))
      (, (sol $mps 1 (c: (-> $ta $ta) I))
         {scheduler}))"""


def render_scheduled_raw_mitm(target_name: str, target: Target, hf: int) -> str:
    return "\n".join(
        [
            program_header("mitm-r3-raw", target_name, target, hf),
            LOAD_ARITH,
            arithmetic_tables(),
            "",
            FWD_SEEDS,
            scheduled_seed(target, hf, include_forward_closure=True),
            "",
        ]
    )


def render_forward_only(hf: int) -> str:
    scheduler = static_scheduler("$init_fuel", [FWD_CLOSURE_EXEC.format(hf=hf)])
    return "\n".join(
        [
            f";; Generated by build_mitm_programs.py.\n;; Kind: forward-only\n(forward-bound {hf})\n",
            LOAD_ARITH,
            arithmetic_tables(),
            "",
            FWD_SEEDS,
            f"""(scheduler-fuel {forward_fuel(hf)})

;; Precompute forward strata with the same static scheduler idiom used by round 3.
(exec (20 init-forward)
      (, (forward-bound $hf)
         (scheduler-fuel $init_fuel))
      (, {scheduler}))""",
            "",
        ]
    )


def read_materialized_fwd(hf: int) -> str | None:
    dump = ROOT / f"fwd-only-hf{hf}.dump"
    if not dump.exists():
        return None
    facts = [
        line
        for line in dump.read_text(encoding="utf-8").splitlines()
        if line.startswith("(fwd ")
    ]
    return "\n".join(facts) + ("\n" if facts else "")


def render_scheduled_static_mitm(target_name: str, target: Target, hf: int) -> str | None:
    facts = read_materialized_fwd(hf)
    if facts is None:
        return None
    return "\n".join(
        [
            program_header("mitm-r3-static", target_name, target, hf),
            LOAD_ARITH,
            arithmetic_tables(),
            "",
            ";; Materialized fwd facts computed by fwd-only-hf*.mm2.",
            facts,
            scheduled_seed(target, hf, include_forward_closure=False),
            "",
        ]
    )


BARRIER_ABOVE_CAP_MEET_RULE_TEMPLATE = """((p30 above rule)
  (, ((p30 above rule) $self_p $self_t)
     (active-budget $ski)
     (sol $ski $shi (c: (-> $ma $mb) $mf))
     (gtFn $ski {hf})
     (addFn $meet_remaining 1 $ski)
     (decFn $shi $meet_h)
     (lte $meet_h $meet_remaining)
     (fwd 1 (: $mp $ma)))
  (, (sol $meet_remaining $meet_h (c: $mb ($mf $mp)))
     (exec (p30 above) $self_p $self_t)))"""


BARRIER_ABOVE_CAP_MEET_EXEC_TEMPLATE = """(exec (p30 above)
      (, ((p30 above rule) $self_p $self_t)
         (active-budget $ski)
         (sol $ski $shi (c: (-> $ma $mb) $mf))
         (gtFn $ski {hf})
         (addFn $meet_remaining 1 $ski)
         (decFn $shi $meet_h)
         (lte $meet_h $meet_remaining)
         (fwd 1 (: $mp $ma)))
      (, (sol $meet_remaining $meet_h (c: $mb ($mf $mp)))
         (exec (p30 above) $self_p $self_t)))"""


BARRIER_TAIL_MEET_RULE_TEMPLATE = """((p31 tail rule)
  (, ((p31 tail rule) $self_p $self_t)
     (active-budget $ski)
     (sol $ski $shi (c: (-> $ma $mb) $mf))
     (lte $ski {hf})
     (lte $mk $ski)
     (addFn $meet_remaining $mk $ski)
     (decFn $shi $meet_h)
     (lte $meet_h $meet_remaining)
     (fwd $mk (: $mp $ma)))
  (, (sol $meet_remaining $meet_h (c: $mb ($mf $mp)))
     (exec (p31 tail) $self_p $self_t)))"""


BARRIER_TAIL_MEET_EXEC_TEMPLATE = """(exec (p31 tail)
      (, ((p31 tail rule) $self_p $self_t)
         (active-budget $ski)
         (sol $ski $shi (c: (-> $ma $mb) $mf))
         (lte $ski {hf})
         (lte $mk $ski)
         (addFn $meet_remaining $mk $ski)
         (decFn $shi $meet_h)
         (lte $meet_h $meet_remaining)
         (fwd $mk (: $mp $ma)))
      (, (sol $meet_remaining $meet_h (c: $mb ($mf $mp)))
         (exec (p31 tail) $self_p $self_t)))"""


BARRIER_SPLIT_RULE_TEMPLATE = """((p40 split rule)
  (, ((p40 split rule) $self_p $self_t)
     (active-budget $ski)
     (sol $ski $hi (c: (-> $b $c) $f))
     (gtFn $ski {hf})
     (decFn $ski $ki)
     (lte $hi $ki)
     (incFn $hi $shi))
  (, (sol $ki $shi (c: (-> (→ $a $b) (-> $a $c)) (mpⁱ $f)))
     (exec (p40 split) $self_p $self_t)))"""


BARRIER_SPLIT_EXEC_TEMPLATE = """(exec (p40 split)
      (, ((p40 split rule) $self_p $self_t)
         (active-budget $ski)
         (sol $ski $hi (c: (-> $b $c) $f))
         (gtFn $ski {hf})
         (decFn $ski $ki)
         (lte $hi $ki)
         (incFn $hi $shi))
      (, (sol $ki $shi (c: (-> (→ $a $b) (-> $a $c)) (mpⁱ $f)))
         (exec (p40 split) $self_p $self_t)))"""


BARRIER_COMPLETION_RULE = """((p50 complete rule)
  (, ((p50 complete rule) $self_p $self_t)
     (active-budget 0)
     (target $mps (c: $ta $tx))
     (sol 0 0 (c: $ta $tx)))
  (, (final 0 0 (c: $ta $tx))
     (exec (p50 complete) $self_p $self_t)))"""


BARRIER_COMPLETION_EXEC = """(exec (p50 complete)
      (, ((p50 complete rule) $self_p $self_t)
         (active-budget 0)
         (target $mps (c: $ta $tx))
         (sol 0 0 (c: $ta $tx)))
      (, (final 0 0 (c: $ta $tx))
         (exec (p50 complete) $self_p $self_t)))"""


BARRIER_ADVANCE_RULE = """((budget barrier rule)
  (, ((budget barrier rule) $barrier_p $barrier_t)
     ((p30 above rule) $p30_p $p30_t)
     ((p31 tail rule) $p31_p $p31_t)
     ((p40 split rule) $p40_p $p40_t)
     ((p50 complete rule) $p50_p $p50_t)
     (active-budget $ski)
     (decFn $ski $ki))
  (O (- (active-budget $ski))
     (+ (active-budget $ki))
     (+ (exec (p30 above) $p30_p $p30_t))
     (+ (exec (p31 tail) $p31_p $p31_t))
     (+ (exec (p40 split) $p40_p $p40_t))
     (+ (exec (p50 complete) $p50_p $p50_t))
     (+ (exec (quiesce budget step) $barrier_p $barrier_t))))"""


def barrier_budget_rule_execs(hf: int) -> list[str]:
    return [
        BARRIER_ABOVE_CAP_MEET_EXEC_TEMPLATE.format(hf=hf),
        BARRIER_TAIL_MEET_EXEC_TEMPLATE.format(hf=hf),
        BARRIER_SPLIT_EXEC_TEMPLATE.format(hf=hf),
        BARRIER_COMPLETION_EXEC,
    ]


def barrier_budget_rule_facts(hf: int) -> list[str]:
    return [
        BARRIER_ABOVE_CAP_MEET_RULE_TEMPLATE.format(hf=hf),
        BARRIER_TAIL_MEET_RULE_TEMPLATE.format(hf=hf),
        BARRIER_SPLIT_RULE_TEMPLATE.format(hf=hf),
        BARRIER_COMPLETION_RULE,
        BARRIER_ADVANCE_RULE,
    ]


def barrier_budget_execs_block(hf: int, indent: str = "         ") -> str:
    return ("\n" + indent).join(barrier_budget_rule_execs(hf))


def barrier_budget_barrier(hf: int) -> str:
    return f"""(exec (quiesce budget step)
      (, ((budget barrier rule) $barrier_p $barrier_t)
         ((p30 above rule) $p30_p $p30_t)
         ((p31 tail rule) $p31_p $p31_t)
         ((p40 split rule) $p40_p $p40_t)
         ((p50 complete rule) $p50_p $p50_t)
         (active-budget $ski)
         (decFn $ski $ki))
      (O (- (active-budget $ski))
         (+ (active-budget $ki))
         (+ (exec (p30 above) $p30_p $p30_t))
         (+ (exec (p31 tail) $p31_p $p31_t))
         (+ (exec (p40 split) $p40_p $p40_t))
         (+ (exec (p50 complete) $p50_p $p50_t))
         (+ (exec (quiesce budget step) $barrier_p $barrier_t))))"""


def barrier_budget_seed(target: Target, hf: int) -> str:
    return f""";; Seed the backward table, install static budget rules, and process each
;; active budget to quiescence before the barrier advances to the next one.
(exec (init barrier)
      (, (target $mps (c: $ta $tx)))
      (, (sol $mps 1 (c: (-> $ta $ta) I))
         (active-budget $mps)
         {barrier_budget_execs_block(hf)}
         {barrier_budget_barrier(hf)}))"""


def render_barrier_static_mitm(target_name: str, target: Target, hf: int) -> str | None:
    facts = read_materialized_fwd(hf)
    if facts is None:
        return None
    return "\n".join(
        [
            program_header("mitm-barrier-static", target_name, target, hf),
            LOAD_ARITH,
            arithmetic_tables(),
            "",
            ";; Materialized fwd facts computed by fwd-only-hf*.mm2.",
            facts,
            ";; Stable rule bytes used to re-arm static execs at each quiescent boundary.",
            "\n\n".join(barrier_budget_rule_facts(hf)),
            barrier_budget_seed(target, hf),
            "",
        ]
    )


def build() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in ROOT.glob("mitm-*.mm2"):
        path.unlink()
    (ROOT / "mitm-add-table.mm2").write_text(
        arithmetic_tables() + "\n", encoding="utf-8"
    )
    for name, target in TARGETS.items():
        (ROOT / f"bfc-{name}.mm2").write_text(
            render_bfc_program(name, target), encoding="utf-8"
        )
    for name, hfs in HF_BY_TARGET.items():
        target = TARGETS[name]
        for hf in hfs:
            raw = render_scheduled_raw_mitm(name, target, hf)
            for prefix in ("mitm-r3-raw", "mitm-capped-raw", "mitm-raw", "mitm"):
                (ROOT / f"{prefix}-{name}-hf{hf}.mm2").write_text(
                    raw, encoding="utf-8"
                )
    for hf in sorted({hf for hfs in HF_BY_TARGET.values() for hf in hfs}):
        (ROOT / f"fwd-only-hf{hf}.mm2").write_text(
            render_forward_only(hf), encoding="utf-8"
        )
    for name, hfs in HF_BY_TARGET.items():
        target = TARGETS[name]
        for hf in hfs:
            static = render_scheduled_static_mitm(name, target, hf)
            if static is not None:
                for prefix in ("mitm-r3-static", "mitm-capped-static", "mitm-static"):
                    (ROOT / f"{prefix}-{name}-hf{hf}.mm2").write_text(
                        static, encoding="utf-8"
                    )
            barrier_static = render_barrier_static_mitm(name, target, hf)
            if barrier_static is not None:
                (ROOT / f"mitm-barrier-static-{name}-hf{hf}.mm2").write_text(
                    barrier_static, encoding="utf-8"
                )


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import plan_oracle


ROOT = Path(__file__).resolve().parent
TRANSITIONS: list[tuple[str, int, int]] = []

for blank in range(9):
    for action, delta in plan_oracle.MOVES:
        target = blank + delta
        if plan_oracle.valid_swap(blank, delta):
            TRANSITIONS.append((action, blank, target))


def state_term(cells: list[str]) -> str:
    return "(state " + " ".join(cells) + ")"


def move_schema(blank: int, target: int) -> tuple[str, str]:
    cells = [f"$c{i}" for i in range(9)]
    cells[blank] = "_"
    src = cells[:]
    dst = cells[:]
    dst[blank], dst[target] = dst[target], dst[blank]
    return state_term(src), state_term(dst)


def arithmetic_tables(max_depth: int) -> str:
    lines = [";; Decimal tables used by the static depth scheduler."]
    for value in range(1, max_depth + 1):
        lines.append(f"(decFn {value} {value - 1})")
    lines.append("")
    lines.append(";; addFn is only used for meet-depth reporting.")
    for left in range(max_depth + 1):
        for right in range(max_depth + 1 - left):
            lines.append(f"(addFn {left} {right} {left + right})")
    return "\n".join(lines)


def side_layer_table(side: str, max_depth: int) -> str:
    lines = [
        f";; {side} layer schedule. Rules expand CURRENT and emit NEXT."
    ]
    for remaining in range(1, max_depth + 1):
        lines.append(f"({side}-currentLayerFn {remaining} {max_depth - remaining})")
        lines.append(f"({side}-nextLayerFn {remaining} {max_depth - remaining + 1})")
    return "\n".join(lines)


def side_move_rule_fact(side: str, index: int, src: str, dst: str) -> str:
    atom = f"{side}-move-{index:02d}"
    return f"""(({atom} rule)
  (, (({atom} rule) $self_p $self_t)
     (active-{side}-budget $budget)
     (decFn $budget $next_budget)
     ({side}-currentLayerFn $budget $current_depth)
     ({side}-nextLayerFn $budget $next_depth)
     ({side}-layer $current_depth {src}))
  (O (guard ({side}-seen {dst}) ({side}-seen {dst}))
     (guard ({side}-seen {dst}) ({side}-layer $next_depth {dst}))
     (+ (exec ({atom}) $self_p $self_t))))"""


def side_move_rule_facts(side: str) -> list[str]:
    facts: list[str] = []
    for index, (_, blank, target) in enumerate(TRANSITIONS):
        src, dst = move_schema(blank, target)
        if side == "bwd":
            src, dst = dst, src
        facts.append(side_move_rule_fact(side, index, src, dst))
    return facts


def rule_ref(atom: str, prefix: str) -> str:
    return f"(({atom} rule) ${prefix}_p ${prefix}_t)"


def exec_emit(atom: str, prefix: str) -> str:
    return f"(+ (exec ({atom}) ${prefix}_p ${prefix}_t))"


def exec_bare(atom: str, prefix: str) -> str:
    return f"(exec ({atom}) ${prefix}_p ${prefix}_t)"


def move_rule_refs(side: str) -> list[str]:
    return [
        rule_ref(f"{side}-move-{index:02d}", f"m{index:02d}")
        for index in range(len(TRANSITIONS))
    ]


def move_exec_emits(side: str) -> list[str]:
    return [
        exec_emit(f"{side}-move-{index:02d}", f"m{index:02d}")
        for index in range(len(TRANSITIONS))
    ]


def indented(lines: list[str], spaces: int = 5) -> str:
    pad = " " * spaces
    return ("\n" + pad).join(lines)


def budget_barrier_rule_fact(side: str, include_meet: bool) -> str:
    meet_pattern = "\n     ((meet rule) $meet_p $meet_t)" if include_meet else ""
    meet_emit = "\n     (+ (exec (meet search) $meet_p $meet_t))" if include_meet else ""
    refs = "\n     ".join(move_rule_refs(side))
    emits = indented(
        [
            *move_exec_emits(side),
            "(+ (exec (quiesce {side} budget step) $self_p $self_t))".format(side=side),
        ]
    )
    return f"""(({side}-budget-barrier rule)
  (, (({side}-budget-barrier rule) $self_p $self_t)
     {refs}{meet_pattern}
     (active-{side}-budget $budget)
     (decFn $budget $next_budget))
  (O (- (active-{side}-budget $budget))
     (+ (active-{side}-budget $next_budget)){meet_emit}
     {emits}))"""


def side_rule_facts(side: str, include_meet: bool) -> str:
    facts = [
        *side_move_rule_facts(side),
        budget_barrier_rule_fact(side, include_meet),
    ]
    return "\n\n".join(facts)


def side_seed_exec(side: str, seed_fact: str) -> str:
    refs = "\n         ".join([rule_ref(f"{side}-budget-barrier", "barrier"), *move_rule_refs(side)])
    emits = indented(
        [
            *[exec_bare(f"{side}-move-{index:02d}", f"m{index:02d}") for index in range(len(TRANSITIONS))],
            "(exec (quiesce {side} budget step) $barrier_p $barrier_t)".format(side=side),
        ],
        9,
    )
    return f"""(exec (init {side})
      (, ({seed_fact} $seed_state)
         ({side}-depth $depth)
         {refs})
      (, ({side}-seen $seed_state)
         ({side}-layer 0 $seed_state)
         (active-{side}-budget $depth)
         {emits}))"""


def meet_rule_fact() -> str:
    return """((meet rule)
  (, ((meet rule) $self_p $self_t)
     (fwd-layer $fwd_depth $state)
     (bwd-layer $bwd_depth $state)
     (addFn $fwd_depth $bwd_depth $meet_depth))
  (, (meet $meet_depth $fwd_depth $bwd_depth $state)
     (exec (meet search) $self_p $self_t)))"""


def header(kind: str, instance: plan_oracle.Instance) -> str:
    return "\n".join(
        [
            ";; Generated by build_plan_programs.py.",
            f";; Kind: {kind}",
            f";; Instance: {instance.name}",
            f";; Oracle optimal distance: {instance.optimal_distance}",
            f"(start {plan_oracle.state_sexpr(instance.state)})",
            f"(goal {plan_oracle.state_sexpr(instance.goal)})",
            "",
        ]
    )


def render_forward(instance: plan_oracle.Instance) -> str:
    depth = instance.optimal_distance
    return "\n".join(
        [
            header("forward-only BFS", instance),
            f"(fwd-depth {depth})",
            arithmetic_tables(depth),
            side_layer_table("fwd", depth),
            "",
            side_rule_facts("fwd", include_meet=False),
            side_seed_exec("fwd", "start"),
            "",
        ]
    )


def render_backward(instance: plan_oracle.Instance) -> str:
    depth = instance.optimal_distance
    return "\n".join(
        [
            header("backward-only BFS", instance),
            f"(bwd-depth {depth})",
            arithmetic_tables(depth),
            side_layer_table("bwd", depth),
            "",
            side_rule_facts("bwd", include_meet=False),
            side_seed_exec("bwd", "goal"),
            "",
        ]
    )


def render_mitm(instance: plan_oracle.Instance) -> str:
    distance = instance.optimal_distance
    fwd_depth = distance // 2
    bwd_depth = distance - fwd_depth
    return "\n".join(
        [
            header("MITM bidirectional BFS", instance),
            f"(fwd-depth {fwd_depth})",
            f"(bwd-depth {bwd_depth})",
            arithmetic_tables(distance),
            side_layer_table("fwd", fwd_depth),
            side_layer_table("bwd", bwd_depth),
            "",
            meet_rule_fact(),
            "",
            side_rule_facts("fwd", include_meet=True),
            "",
            side_rule_facts("bwd", include_meet=True),
            "",
            side_seed_exec("fwd", "start"),
            side_seed_exec("bwd", "goal"),
            "",
        ]
    )


def program_paths(instance: plan_oracle.Instance) -> dict[str, Path]:
    distance = instance.optimal_distance
    fwd_depth = distance // 2
    bwd_depth = distance - fwd_depth
    return {
        "forward": ROOT / f"{instance.name}-forward-d{distance}.mm2",
        "backward": ROOT / f"{instance.name}-backward-d{distance}.mm2",
        "mitm": ROOT / f"{instance.name}-mitm-d{distance}-f{fwd_depth}-b{bwd_depth}.mm2",
    }


def build(instances: list[plan_oracle.Instance]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    plan_oracle.write_instances(ROOT / "instances.json", instances)
    for instance in instances:
        paths = program_paths(instance)
        paths["forward"].write_text(render_forward(instance), encoding="utf-8")
        paths["backward"].write_text(render_backward(instance), encoding="utf-8")
        paths["mitm"].write_text(render_mitm(instance), encoding="utf-8")
        print(
            f"{instance.name}: wrote {paths['forward'].name}, "
            f"{paths['backward'].name}, {paths['mitm'].name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 8-puzzle MM2 planning demos")
    parser.add_argument("--seed", type=int, default=plan_oracle.DEFAULT_SEED)
    parser.add_argument("--targets", type=int, nargs="*", default=list(plan_oracle.DEFAULT_TARGET_DEPTHS))
    args = parser.parse_args()
    build(plan_oracle.generate_instances(args.targets, args.seed))


if __name__ == "__main__":
    main()

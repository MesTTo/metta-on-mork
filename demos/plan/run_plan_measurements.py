#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import build_plan_programs
import plan_oracle


ROOT = Path(__file__).resolve().parent
MORK = Path(os.environ.get("MORK_BIN", "mork"))
STATE_RE = r"\(state ([^)]+)\)"
LAYER_RE = re.compile(rf"^\((fwd|bwd)-layer ([0-9]+) {STATE_RE}\)$")
MEET_RE = re.compile(rf"^\(meet ([0-9]+) ([0-9]+) ([0-9]+) {STATE_RE}\)$")


def parse_state(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def dump_path(program: Path) -> Path:
    return program.with_suffix(".dump")


def log_path(program: Path) -> Path:
    return program.with_suffix(".log")


def run_mork(program: Path) -> dict[str, object]:
    dump = dump_path(program)
    log = log_path(program)
    command = [str(MORK), "run", str(program), str(dump)]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with exit {completed.returncode}: "
            f"{completed.stdout.strip()}"
        )
    return {
        "command": " ".join(command),
        "program": program.name,
        "dump": dump.name,
        "log": log.name,
        "wall_ms": wall_ms,
        "stdout": completed.stdout.strip(),
    }


def parse_dump(path: Path) -> dict[str, object]:
    layers: dict[str, dict[int, set[tuple[str, ...]]]] = {"fwd": {}, "bwd": {}}
    meets: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        layer_match = LAYER_RE.match(line)
        if layer_match:
            side, depth_text, state_text = layer_match.groups()
            depth = int(depth_text)
            layers[side].setdefault(depth, set()).add(parse_state(state_text))
            continue
        meet_match = MEET_RE.match(line)
        if meet_match:
            total, fwd_depth, bwd_depth, state_text = meet_match.groups()
            meets.append(
                {
                    "total": int(total),
                    "fwd_depth": int(fwd_depth),
                    "bwd_depth": int(bwd_depth),
                    "state": parse_state(state_text),
                }
            )
    return {
        "layers": {
            side: {str(depth): len(states) for depth, states in sorted(by_depth.items())}
            for side, by_depth in layers.items()
        },
        "layer_states": layers,
        "meets": meets,
    }


def min_depth_for_state(
    layers: dict[int, set[tuple[str, ...]]], target: tuple[str, ...]
) -> int | None:
    for depth, states in sorted(layers.items()):
        if target in states:
            return depth
    return None


def summarize_run(
    instance: plan_oracle.Instance, kind: str, run: dict[str, object]
) -> dict[str, object]:
    parsed = parse_dump(ROOT / str(run["dump"]))
    layer_states = parsed.pop("layer_states")
    if kind == "forward":
        found_depth = min_depth_for_state(layer_states["fwd"], instance.goal)
    elif kind == "backward":
        found_depth = min_depth_for_state(layer_states["bwd"], instance.state)
    else:
        meet_totals = [int(meet["total"]) for meet in parsed["meets"]]
        found_depth = min(meet_totals) if meet_totals else None
    return {
        **run,
        "kind": kind,
        "layers": parsed["layers"],
        "meet_count": len(parsed["meets"]),
        "found_depth": found_depth,
    }


def theoretical_nodes(branching: float, depth: int) -> tuple[float, float]:
    forward = branching**depth
    mitm = 2 * (branching ** (depth / 2))
    return forward, mitm


def build_report(results: dict[str, object]) -> str:
    lines: list[str] = [
        "# 8-Puzzle MITM Planning Demonstrator",
        "",
        "This directory builds a STRIPS-style sliding-tile planning demo on MORK. States are ground `(state c0 ... c8)` tuples and `_` is the blank. The generated MM2 uses static move and barrier rule facts. Layer depth changes through `active-*-budget`, `decFn`, `*-currentLayerFn`, and `*-nextLayerFn` facts, so round counters are not baked into transition rule bytes.",
        "",
        "The claim tested here is layer composition on MORK: barrier staging plus joins can expose the expected meet-in-the-middle shape. This is not a claim that this MM2 encoding beats specialized 8-puzzle planners.",
        "",
        "## Commands",
        "",
        "```sh",
        "python3 demos/plan/plan_oracle.py --write-instances demos/plan/instances.json",
        "python3 demos/plan/build_plan_programs.py",
        "python3 demos/plan/run_plan_measurements.py",
        "```",
        "",
        "The driver then executed these MORK commands:",
        "",
        "```sh",
    ]
    for key in sorted(results["runs"]):
        lines.append(results["runs"][key]["command"])
    lines.extend(
        [
            "```",
            "",
            "The wall-time numbers below are measured by the Python driver around each command. The per-program logs are written beside the dumps.",
        ]
    )
    lines.extend(
        [
            "",
            "## Instances",
            "",
            "| instance | optimal | start state | scramble |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for item in results["instances"]:
        lines.append(
            f"| {item['name']} | {item['optimal_distance']} | "
            f"`{plan_oracle.state_sexpr(tuple(item['state']))}` | "
            f"`{''.join(item['scramble_actions'])}` |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| instance | oracle | forward depth | MITM split | MITM meet depth | forward ms | backward ms | MITM ms | forward states | MITM fwd states | MITM bwd states | meet facts |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results["rows"]:
        lines.append(
            f"| {row['instance']} | {row['oracle_distance']} | {row['forward_depth']} | "
            f"{row['mitm_split']} | {row['mitm_depth']} | "
            f"{row['forward_ms']:.2f} | {row['backward_ms']:.2f} | {row['mitm_ms']:.2f} | "
            f"{row['forward_states']} | {row['mitm_fwd_states']} | {row['mitm_bwd_states']} | "
            f"{row['meet_count']} |"
        )
    lines.extend(
        [
            "",
            "## Strata",
            "",
        ]
    )
    for row in results["rows"]:
        lines.append(f"### {row['instance']}")
        lines.append("")
        lines.append(f"- Forward-only fwd-layer counts: `{row['forward_layer_counts']}`")
        lines.append(f"- Backward-only bwd-layer counts: `{row['backward_layer_counts']}`")
        lines.append(f"- MITM fwd-layer counts: `{row['mitm_fwd_layer_counts']}`")
        lines.append(f"- MITM bwd-layer counts: `{row['mitm_bwd_layer_counts']}`")
        lines.append("")
    lines.extend(
        [
            "## Branching Shape",
            "",
            "The measured distinct layer counts are small because the 8-puzzle state graph has many duplicate paths and the guard table keeps first visits only. To show the usual search-tree intuition, the table below derives an effective `b` from the measured final forward stratum, so `b^d` equals that final layer count, then compares it with `2*b^(d/2)` at the same oracle depth.",
            "",
            "| instance | measured b | d | b^d | 2*b^(d/2) | ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results["rows"]:
        lines.append(
            f"| {row['instance']} | {row['branching']:.4f} | {row['oracle_distance']} | "
            f"{row['tree_forward']:.2f} | {row['tree_mitm']:.2f} | "
            f"{row['tree_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Verification",
            "",
        ]
    )
    for check in results["checks"]:
        lines.append(f"- {check}")
    return "\n".join(lines) + "\n"


def main() -> None:
    instances = plan_oracle.generate_instances()
    build_plan_programs.build(instances)
    rows: list[dict[str, object]] = []
    checks: list[str] = []
    runs: dict[str, dict[str, object]] = {}
    for instance in instances:
        paths = build_plan_programs.program_paths(instance)
        summaries: dict[str, dict[str, object]] = {}
        for kind, program in paths.items():
            run = run_mork(program)
            summary = summarize_run(instance, kind, run)
            summaries[kind] = summary
            runs[f"{instance.name}:{kind}"] = summary
        forward_depth = summaries["forward"]["found_depth"]
        backward_depth = summaries["backward"]["found_depth"]
        mitm_depth = summaries["mitm"]["found_depth"]
        if forward_depth != instance.optimal_distance:
            raise AssertionError(f"{instance.name}: forward depth {forward_depth} != oracle {instance.optimal_distance}")
        if backward_depth != instance.optimal_distance:
            raise AssertionError(f"{instance.name}: backward depth {backward_depth} != oracle {instance.optimal_distance}")
        if mitm_depth != instance.optimal_distance:
            raise AssertionError(f"{instance.name}: MITM depth {mitm_depth} != oracle {instance.optimal_distance}")
        checks.append(
            f"{instance.name}: oracle={instance.optimal_distance}, forward={forward_depth}, backward={backward_depth}, MITM={mitm_depth}"
        )
        fwd_counts = {int(k): int(v) for k, v in summaries["forward"]["layers"]["fwd"].items()}
        mitm_fwd_counts = {int(k): int(v) for k, v in summaries["mitm"]["layers"]["fwd"].items()}
        mitm_bwd_counts = {int(k): int(v) for k, v in summaries["mitm"]["layers"]["bwd"].items()}
        bwd_counts = {int(k): int(v) for k, v in summaries["backward"]["layers"]["bwd"].items()}
        branching = fwd_counts[instance.optimal_distance] ** (
            1.0 / instance.optimal_distance
        )
        tree_forward, tree_mitm = theoretical_nodes(branching, instance.optimal_distance)
        rows.append(
            {
                "instance": instance.name,
                "oracle_distance": instance.optimal_distance,
                "forward_depth": forward_depth,
                "backward_depth": backward_depth,
                "mitm_depth": mitm_depth,
                "mitm_split": f"{instance.optimal_distance // 2}+{instance.optimal_distance - instance.optimal_distance // 2}",
                "forward_ms": summaries["forward"]["wall_ms"],
                "backward_ms": summaries["backward"]["wall_ms"],
                "mitm_ms": summaries["mitm"]["wall_ms"],
                "forward_states": sum(fwd_counts.values()),
                "mitm_fwd_states": sum(mitm_fwd_counts.values()),
                "mitm_bwd_states": sum(mitm_bwd_counts.values()),
                "meet_count": summaries["mitm"]["meet_count"],
                "forward_layer_counts": fwd_counts,
                "backward_layer_counts": bwd_counts,
                "mitm_fwd_layer_counts": mitm_fwd_counts,
                "mitm_bwd_layer_counts": mitm_bwd_counts,
                "branching": branching,
                "tree_forward": tree_forward,
                "tree_mitm": tree_mitm,
                "tree_ratio": tree_forward / tree_mitm,
            }
        )
    results = {
        "instances": [asdict(instance) for instance in instances],
        "rows": rows,
        "runs": runs,
        "checks": checks,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (ROOT / "REPORT.md").write_text(build_report(results), encoding="utf-8")
    print(f"wrote {ROOT / 'results.json'}")
    print(f"wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()

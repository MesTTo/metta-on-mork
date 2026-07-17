from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = ROOT / "rules" / "xor_tick.mm2"

BASE_PHASE_ORDER = (
    "P00_CLEAR_PCS_H",
    "P00_CLEAR_PCS_Y",
    "P00_CLEAR_PCPHI_H",
    "P00_CLEAR_PREH",
    "P00_CLEAR_SH",
    "P00_CLEAR_PHIH",
    "P00_CLEAR_PREY",
    "P00_CLEAR_SY",
    "P00_CLEAR_GY",
    "P00_CLEAR_PCG_Y",
    "P00_CLEAR_BACKH",
    "P00_CLEAR_PCB_H",
    "P00_CLEAR_BH",
    "P00_CLEAR_PHI2H",
    "P00_CLEAR_PRIMEH",
    "P00_CLEAR_BPH",
    "P00_CLEAR_PCG_H",
    "P00_CLEAR_GH",
    "P10_PREH",
    "P10_STATE",
    "P10_PHIH",
    "P10_ALIAS_PCPHI_H",
    "P10_PREY",
    "P10_OUTPUT",
    "P20_GY",
    "P20_BACK",
    "P20_PCB",
    "P30_PHI2",
    "P30_PRIME",
    "P30_BP",
    "P30_GH",
    "P35_HIST_EH",
    "P35_HIST_EY",
    "P40_EH",
    "P40_EY",
)
TRAIN_HISTORY_PHASES = (
    "P35_TRAIN_SH_H",
    "P35_TRAIN_SH_Y",
    "P35_TRAIN_EH",
    "P35_TRAIN_EY",
)
WEIGHT_PHASES = (
    "P60_CLEAR_DWXH",
    "P60_CLEAR_DWHY",
    "P60_CLEAR_SDWXH",
    "P60_CLEAR_SDWHY",
    "P60_DWXH",
    "P60_DWHY",
    "P60_SCALE_DWXH",
    "P60_SCALE_DWHY",
    "P60_FOLD_WXH",
    "P60_FOLD_WHY",
    "P60_CLEAR_PCW_XH",
    "P60_SYNC_PCW_XH",
    "P60_CLEAR_PCW_HY",
    "P60_SYNC_PCW_HY",
)
POST_WEIGHT_TICK_PHASES = (
    "P69_ADVANCE_TICK",
)
RELOAD_PHASES = (
    "P70_ADVANCE_UPDATE",
    "P70_CLEAR_PCIN_X",
    "P70_CLEAR_PCS_X",
    "P70_CLEAR_PCPHI_X",
    "P70_CLEAR_PHIX",
    "P70_CLEAR_PCIN_Y",
    "P70_CLEAR_YT",
    "P70_CLEAR_PCE_H",
    "P70_CLEAR_EH",
    "P70_CLEAR_PCE_Y",
    "P70_CLEAR_EY",
    "P71_LOAD_X",
    "P71_LOAD_Y",
    "P71_RESET_EH",
    "P71_RESET_EY",
)
TRAIN_TICK_PHASE_ORDER = BASE_PHASE_ORDER[:-2] + TRAIN_HISTORY_PHASES + BASE_PHASE_ORDER[-2:]
TRAIN_PHASE_ORDER = TRAIN_TICK_PHASE_ORDER + WEIGHT_PHASES + POST_WEIGHT_TICK_PHASES + RELOAD_PHASES
PHASE_ORDER = BASE_PHASE_ORDER

DEFAULT_SPEC: dict[str, Any] = {
    "nodes": [
        {"name": "x", "dim": 2, "role": "input"},
        {"name": "h", "dim": 2, "role": "hidden", "activation": "tanh"},
        {"name": "y", "dim": 2, "role": "output"},
    ],
    "edges": [
        {
            "name": "xh",
            "src": "x",
            "dst": "h",
            "weight": "wxh",
            "pre": "preh",
            "src_activation": "phix",
            "forward": "ab,b->a",
        },
        {
            "name": "hy",
            "src": "h",
            "dst": "y",
            "weight": "why",
            "pre": "prey",
            "src_activation": "phih",
            "forward": "ab,b->a",
        },
    ],
    "activation_lowering": "tensor-op-unary",
    "error_distribution": "gaussian",
    "output_family": "free-output-mse",
}


@dataclass(frozen=True)
class NodeSpec:
    name: str
    dim: int
    role: str
    activation: str | None = None


@dataclass(frozen=True)
class EdgeSpec:
    name: str
    src: str
    dst: str
    weight: str
    pre: str
    src_activation: str
    forward: str


@dataclass(frozen=True)
class NetworkSpec:
    nodes: Mapping[str, NodeSpec]
    edges: Mapping[str, EdgeSpec]
    activation_lowering: str
    error_distribution: str
    output_family: str


@dataclass(frozen=True)
class EinsumSpec:
    inputs: tuple[str, ...]
    output: str

    @classmethod
    def parse(cls, text: str) -> "EinsumSpec":
        left, output = text.split("->", 1)
        inputs = tuple(part.strip() for part in left.split(","))
        if len(inputs) != 2 or not inputs[0] or not inputs[1] or not output:
            raise ValueError(f"unsupported einsum spec: {text}")
        return cls(inputs=inputs, output=output.strip())

    def hidden_backpressure(self) -> str:
        return f"{self.inputs[0]},{self.output}->{self.inputs[1]}"

    def weight_delta(self) -> str:
        return f"{self.output},{self.inputs[1]}->{self.inputs[0]}"


def load_spec(path: Path | None, activation_lowering: str | None) -> NetworkSpec:
    data = DEFAULT_SPEC if path is None else json.loads(path.read_text())
    lowering = activation_lowering or str(data.get("activation_lowering", DEFAULT_SPEC["activation_lowering"]))
    nodes = {
        str(node["name"]): NodeSpec(
            name=str(node["name"]),
            dim=int(node["dim"]),
            role=str(node["role"]),
            activation=node.get("activation"),
        )
        for node in data["nodes"]
    }
    edges = {
        str(edge["name"]): EdgeSpec(
            name=str(edge["name"]),
            src=str(edge["src"]),
            dst=str(edge["dst"]),
            weight=str(edge["weight"]),
            pre=str(edge["pre"]),
            src_activation=str(edge["src_activation"]),
            forward=str(edge["forward"]),
        )
        for edge in data["edges"]
    }
    spec = NetworkSpec(
        nodes=nodes,
        edges=edges,
        activation_lowering=lowering,
        error_distribution=str(data.get("error_distribution", DEFAULT_SPEC["error_distribution"])),
        output_family=str(data.get("output_family", DEFAULT_SPEC["output_family"])),
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: NetworkSpec) -> None:
    if tuple(spec.nodes) != ("x", "h", "y"):
        raise ValueError("pcgraph derivation currently expects nodes x, h, y in that order")
    if tuple(spec.edges) != ("xh", "hy"):
        raise ValueError("pcgraph derivation currently expects edges xh and hy in that order")
    if spec.nodes["x"].dim != 2 or spec.nodes["h"].dim != 2 or spec.nodes["y"].dim != 2:
        raise ValueError("pcgraph derivation currently targets the XOR 2-2-2 chain")
    if spec.nodes["h"].activation != "tanh":
        raise ValueError("pcgraph hidden activation must be tanh")
    if spec.activation_lowering not in {"tensor-op-unary", "pure"}:
        raise ValueError("activation_lowering must be tensor-op-unary or pure")
    if spec.error_distribution != "gaussian":
        raise ValueError("only the gaussian psi(eps)=eps map is implemented")
    if spec.output_family != "free-output-mse":
        raise ValueError("only the free-output MSE family is implemented")
    for edge in spec.edges.values():
        parsed = EinsumSpec.parse(edge.forward)
        if parsed.inputs != ("ab", "b") or parsed.output != "a":
            raise ValueError(f"{edge.name} must use forward spec ab,b->a")


def phase(name: str, body: str) -> str:
    return f"; %%PHASE {name}\n{body.strip()}\n; %%END\n"


def tensor_op(
    priority: int,
    pattern: Sequence[str],
    op: str,
    inputs: Sequence[str],
    output: str,
    sources: Sequence[str],
) -> str:
    return f"""(exec {priority}
  (, {" ".join(pattern)})
  (O (tensor-op-f32
        (op {op})
        (inputs {" ".join(inputs)})
        (output {output})
        (from {" ".join(sources)})
        (backend auto))))"""


def clear_phase(name: str, priority: int, relation: str, variables: Sequence[str]) -> str:
    cells = " ".join((relation, *variables))
    return phase(
        name,
        f"""(exec {priority}
  (, ({cells}))
  (O (- ({cells}))))""",
    )


def emit_header(spec: NetworkSpec) -> str:
    return f"""; One jpc-native ePC inner-loop tick for the fixed XOR chain.
; This file is derived from the forward specs in derive_rules.py.
; Forward xh: {spec.edges["xh"].forward}; forward hy: {spec.edges["hy"].forward}.
; The hidden tanh activation is lowered as {spec.activation_lowering}.
; The gaussian error distribution uses psi(eps)=eps.
"""


def emit_clear_phases() -> list[str]:
    clear_specs = [
        ("P00_CLEAR_PCS_H", "pcs", ("h", "$i", "$v")),
        ("P00_CLEAR_PCS_Y", "pcs", ("y", "$i", "$v")),
        ("P00_CLEAR_PCPHI_H", "pcphi", ("h", "$i", "$v")),
        ("P00_CLEAR_PREH", "preh", ("$i", "$v")),
        ("P00_CLEAR_SH", "sh", ("$i", "$v")),
        ("P00_CLEAR_PHIH", "phih", ("$i", "$v")),
        ("P00_CLEAR_PREY", "prey", ("$i", "$v")),
        ("P00_CLEAR_SY", "sy", ("$i", "$v")),
        ("P00_CLEAR_GY", "gy", ("$i", "$v")),
        ("P00_CLEAR_PCG_Y", "pcg", ("y", "$i", "$v")),
        ("P00_CLEAR_BACKH", "backh", ("$i", "$v")),
        ("P00_CLEAR_PCB_H", "pcb", ("h", "$i", "$v")),
        ("P00_CLEAR_BH", "bh", ("$i", "$v")),
        ("P00_CLEAR_PHI2H", "phi2h", ("$i", "$v")),
        ("P00_CLEAR_PRIMEH", "primeh", ("$i", "$v")),
        ("P00_CLEAR_BPH", "bph", ("$i", "$v")),
        ("P00_CLEAR_PCG_H", "pcg", ("h", "$i", "$v")),
        ("P00_CLEAR_GH", "gh", ("$i", "$v")),
    ]
    return [clear_phase(name, 1, relation, variables) for name, relation, variables in clear_specs]


def emit_forward_pre_phase(name: str, priority: int, edge: EdgeSpec) -> str:
    return phase(
        name,
        tensor_op(
            priority,
            [f"({edge.weight} $i $j $w)", f"({edge.src_activation} $j $x)"],
            f"einsum {edge.forward}",
            [f"({edge.weight} dense 2 2)", f"({edge.src_activation} dense 2)"],
            f"({edge.pre} dense)",
            [f"({edge.weight} $i $j $w)", f"({edge.src_activation} $j $x)"],
        ),
    )


def emit_hidden_state_phase() -> str:
    return phase(
        "P10_STATE",
        """(exec 11
  (, (preh $i $p)
     (eh $i $e)
     (pctick $t))
  (O (pure (pcs h $i $s) $s
           (f32_to_string (sum_f32 (f32_from_string $p) (f32_from_string $e))))
     (pure (sh $i $s) $s
           (f32_to_string (sum_f32 (f32_from_string $p) (f32_from_string $e))))
     (pure (pcsh $t h $i $s) $s
           (f32_to_string (sum_f32 (f32_from_string $p) (f32_from_string $e))))))""",
    )


def emit_hidden_activation_phase(lowering: str) -> list[str]:
    if lowering == "pure":
        return [
            phase(
                "P10_PHIH",
                """(exec 14
  (, (sh $i $s))
  (O (pure (phih $i $phi) $phi
           (f32_to_string (tanh_f32 (f32_from_string $s))))))""",
            ),
            phase(
                "P10_ALIAS_PCPHI_H",
                """(exec 15
  (, (phih $i $phi))
  (O (+ (pcphi h $i $phi))))""",
            ),
        ]
    return [
        phase(
            "P10_PHIH",
            """(exec 14
  (, (sh $i $s))
  (O (tensor-op-f32
        (op unary tanh)
        (inputs (sh dense 2))
        (output (phih dense))
        (from (sh $i $s))
        (backend auto))))""",
        ),
        phase(
            "P10_ALIAS_PCPHI_H",
            """(exec 15
  (, (phih $i $phi))
  (O (+ (pcphi h $i $phi))))""",
        ),
    ]


def emit_output_state_phase() -> str:
    return phase(
        "P10_OUTPUT",
        """(exec 13
  (, (prey $i $p)
     (ey $i $e)
     (pctick $t))
  (O (pure (pcs y $i $s) $s
           (f32_to_string (sub_f32 (f32_from_string $p) (f32_from_string $e))))
     (pure (sy $i $s) $s
           (f32_to_string (sub_f32 (f32_from_string $p) (f32_from_string $e))))
     (pure (pcsh $t y $i $s) $s
           (f32_to_string (sub_f32 (f32_from_string $p) (f32_from_string $e))))))""",
    )


def emit_output_gradient_phase() -> str:
    return phase(
        "P20_GY",
        """(exec 20
  (, (yt $i $target)
     (sy $i $s))
  (O (pure (pcg y $i $g) $g
           (f32_to_string (sub_f32 (f32_from_string $target) (f32_from_string $s))))
     (pure (gy $i $g) $g
           (f32_to_string (sub_f32 (f32_from_string $target) (f32_from_string $s))))))""",
    )


def emit_backpressure_phase(edge: EdgeSpec) -> str:
    backpressure = EinsumSpec.parse(edge.forward).hidden_backpressure()
    return phase(
        "P20_BACK",
        tensor_op(
            21,
            [f"({edge.weight} $i $j $w)", "(gy $i $g)"],
            f"einsum {backpressure}",
            [f"({edge.weight} dense 2 2)", "(gy dense 2)"],
            "(backh dense)",
            [f"({edge.weight} $i $j $w)", "(gy $i $g)"],
        ),
    )


def emit_hidden_back_alias_phase() -> str:
    return phase(
        "P20_PCB",
        """(exec 22
  (, (backh $i $b))
  (O (+ (pcb h $i $b))
     (+ (bh $i $b))))""",
    )


def emit_hidden_gradient_phases() -> list[str]:
    return [
        phase(
            "P30_PHI2",
            """(exec 30
  (, (phih $i $phi))
  (O (pure (phi2h $i $phi2) $phi2
           (f32_to_string
             (product_f32 (f32_from_string $phi) (f32_from_string $phi))))))""",
        ),
        phase(
            "P30_PRIME",
            """(exec 31
  (, (phi2h $i $phi2)
     (one $one))
  (O (pure (primeh $i $prime) $prime
           (f32_to_string
             (sub_f32 (f32_from_string $one) (f32_from_string $phi2))))))""",
        ),
        phase(
            "P30_BP",
            """(exec 32
  (, (primeh $i $prime)
     (bh $i $b))
  (O (pure (bph $i $bp) $bp
           (f32_to_string
             (product_f32 (f32_from_string $prime) (f32_from_string $b))))))""",
        ),
        phase(
            "P30_GH",
            """(exec 33
  (, (eh $i $e)
     (bph $i $bp))
  (O (pure (pcg h $i $g) $g
           (f32_to_string (sub_f32 (f32_from_string $e) (f32_from_string $bp))))
     (pure (gh $i $g) $g
           (f32_to_string (sub_f32 (f32_from_string $e) (f32_from_string $bp))))))""",
        ),
    ]


def emit_history_phases() -> list[str]:
    return [
        phase(
            "P35_HIST_EH",
            """(exec 35
  (, (pctick $t)
     (pce h $i $e))
  (O (+ (pceh $t h $i $e))))""",
        ),
        phase(
            "P35_HIST_EY",
            """(exec 36
  (, (pctick $t)
     (pce y $i $e))
  (O (+ (pceh $t y $i $e))))""",
        ),
        phase(
            "P35_TRAIN_SH_H",
            """(exec 37
  (, (pcupdate $u)
     (pctick $t)
     (sh $i $s))
  (O (+ (pctrsh $u $t h $i $s))))""",
        ),
        phase(
            "P35_TRAIN_SH_Y",
            """(exec 38
  (, (pcupdate $u)
     (pctick $t)
     (sy $i $s))
  (O (+ (pctrsh $u $t y $i $s))))""",
        ),
        phase(
            "P35_TRAIN_EH",
            """(exec 39
  (, (pcupdate $u)
     (pctick $t)
     (pce h $i $e))
  (O (+ (pctreh $u $t h $i $e))))""",
        ),
        phase(
            "P35_TRAIN_EY",
            """(exec 40
  (, (pcupdate $u)
     (pctick $t)
     (pce y $i $e))
  (O (+ (pctreh $u $t y $i $e))))""",
        ),
    ]


def emit_error_update_phases() -> list[str]:
    def one_node(name: str, node: str, helper: str, gradient: str, priority: int) -> str:
        return phase(
            name,
            f"""(exec {priority}
  (, (pchp error-lr $lr)
     (pce {node} $i $old)
     ({helper} $i $old)
     ({gradient} $i $g))
  (O (- (pce {node} $i $old))
     (- ({helper} $i $old))
     (pure (pce {node} $i $new) $new
           (f32_to_string
             (sub_f32
               (f32_from_string $old)
               (product_f32 (f32_from_string $lr) (f32_from_string $g)))))
     (pure ({helper} $i $new) $new
           (f32_to_string
             (sub_f32
               (f32_from_string $old)
               (product_f32 (f32_from_string $lr) (f32_from_string $g)))))))""",
        )

    return [
        one_node("P40_EH", "h", "eh", "gh", 40),
        one_node("P40_EY", "y", "ey", "gy", 41),
    ]


def emit_weight_clear_phases() -> list[str]:
    clear_specs = [
        ("P60_CLEAR_DWXH", 60, "dwxh", ("$i", "$j", "$v")),
        ("P60_CLEAR_DWHY", 61, "dwhy", ("$i", "$j", "$v")),
        ("P60_CLEAR_SDWXH", 62, "sdwxh", ("$i", "$j", "$v")),
        ("P60_CLEAR_SDWHY", 63, "sdwhy", ("$i", "$j", "$v")),
    ]
    return [clear_phase(name, priority, relation, variables) for name, priority, relation, variables in clear_specs]


def emit_delta_phase(name: str, priority: int, edge: EdgeSpec, left: str, left_value: str, right: str, right_value: str, output: str) -> str:
    delta = EinsumSpec.parse(edge.forward).weight_delta()
    return phase(
        name,
        tensor_op(
            priority,
            [f"({left} $i {left_value})", f"({right} $j {right_value})"],
            f"einsum {delta}",
            [f"({left} dense 2)", f"({right} dense 2)"],
            f"({output} dense 2 2)",
            [f"({left} $i {left_value})", f"({right} $j {right_value})"],
        ),
    )


def emit_scale_phase(name: str, priority: int, source: str, output: str) -> str:
    return phase(
        name,
        f"""(exec {priority}
  (, (pchp weight-lr $lr)
     ({source} $i $j $d))
  (O (pure ({output} $i $j $sd) $sd
           (f32_to_string
             (product_f32 (f32_from_string $lr) (f32_from_string $d))))))""",
    )


def emit_fold_phase(name: str, priority: int, weight: str, scaled_delta: str) -> str:
    return phase(
        name,
        tensor_op(
            priority,
            [f"({weight} $i $j $w)", f"({scaled_delta} $i $j $dw)"],
            "add",
            [f"({weight} dense 2 2)", f"({scaled_delta} dense 2 2)"],
            f"({weight} dense 2 2)",
            [f"({weight} $i $j $w)", f"({scaled_delta} $i $j $dw)"],
        ),
    )


def emit_pcw_phase(name: str, priority: int, edge_name: str, weight: str, clear: bool) -> str:
    if clear:
        body = f"""(exec {priority}
  (, (pcw {edge_name} $i $j $w))
  (O (- (pcw {edge_name} $i $j $w))))"""
    else:
        body = f"""(exec {priority}
  (, ({weight} $i $j $w))
  (O (+ (pcw {edge_name} $i $j $w))))"""
    return phase(name, body)


def emit_weight_phases(spec: NetworkSpec) -> list[str]:
    xh = spec.edges["xh"]
    hy = spec.edges["hy"]
    return [
        *emit_weight_clear_phases(),
        emit_delta_phase("P60_DWXH", 64, xh, "eh", "$e", "phix", "$x", "dwxh"),
        emit_delta_phase("P60_DWHY", 65, hy, "gy", "$g", "phih", "$h", "dwhy"),
        emit_scale_phase("P60_SCALE_DWXH", 66, "dwxh", "sdwxh"),
        emit_scale_phase("P60_SCALE_DWHY", 67, "dwhy", "sdwhy"),
        emit_fold_phase("P60_FOLD_WXH", 68, "wxh", "sdwxh"),
        emit_fold_phase("P60_FOLD_WHY", 69, "why", "sdwhy"),
        emit_pcw_phase("P60_CLEAR_PCW_XH", 70, "xh", "wxh", clear=True),
        emit_pcw_phase("P60_SYNC_PCW_XH", 71, "xh", "wxh", clear=False),
        emit_pcw_phase("P60_CLEAR_PCW_HY", 72, "hy", "why", clear=True),
        emit_pcw_phase("P60_SYNC_PCW_HY", 73, "hy", "why", clear=False),
    ]


def emit_tick_and_reload_phases() -> list[str]:
    return [
        phase(
            "P69_ADVANCE_TICK",
            """(exec 79
  (, (pctick $tick)
     (pcnext $tick $next))
  (O (- (pctick $tick))
     (+ (pctick $next))
     (+ (pcadvanced $tick $next))))""",
        ),
        phase(
            "P70_ADVANCE_UPDATE",
            """(exec 80
  (, (pcupdate $u)
     (pcnextupdate $u $next)
     (pctick $t))
  (O (- (pcupdate $u))
     (+ (pcupdate $next))
     (- (pctick $t))
     (+ (pctick 0))))""",
        ),
        clear_phase("P70_CLEAR_PCIN_X", 81, "pcin", ("x", "$i", "$v")),
        clear_phase("P70_CLEAR_PCS_X", 82, "pcs", ("x", "$i", "$v")),
        clear_phase("P70_CLEAR_PCPHI_X", 83, "pcphi", ("x", "$i", "$v")),
        clear_phase("P70_CLEAR_PHIX", 84, "phix", ("$i", "$v")),
        clear_phase("P70_CLEAR_PCIN_Y", 85, "pcin", ("y", "$i", "$v")),
        clear_phase("P70_CLEAR_YT", 86, "yt", ("$i", "$v")),
        clear_phase("P70_CLEAR_PCE_H", 87, "pce", ("h", "$i", "$v")),
        clear_phase("P70_CLEAR_EH", 88, "eh", ("$i", "$v")),
        clear_phase("P70_CLEAR_PCE_Y", 89, "pce", ("y", "$i", "$v")),
        clear_phase("P70_CLEAR_EY", 90, "ey", ("$i", "$v")),
        phase(
            "P71_LOAD_X",
            """(exec 91
  (, (pcupdate $u)
     (pcupdate-sample $u $sample)
     (pcx $sample $i $v))
  (O (+ (pcin x $i $v))
     (+ (pcs x $i $v))
     (+ (pcphi x $i $v))
     (+ (phix $i $v))))""",
        ),
        phase(
            "P71_LOAD_Y",
            """(exec 92
  (, (pcupdate $u)
     (pcupdate-sample $u $sample)
     (pcy $sample $i $v))
  (O (+ (pcin y $i $v))
     (+ (yt $i $v))))""",
        ),
        phase(
            "P71_RESET_EH",
            """(exec 93
  (, (pczero h $i $z))
  (O (+ (pce h $i $z))
     (+ (eh $i $z))))""",
        ),
        phase(
            "P71_RESET_EY",
            """(exec 94
  (, (pczero y $i $z))
  (O (+ (pce y $i $z))
     (+ (ey $i $z))))""",
        ),
    ]


def derive_rules(spec: NetworkSpec) -> str:
    phases = [
        emit_header(spec),
        *emit_clear_phases(),
        emit_forward_pre_phase("P10_PREH", 10, spec.edges["xh"]),
        emit_hidden_state_phase(),
        *emit_hidden_activation_phase(spec.activation_lowering),
        emit_forward_pre_phase("P10_PREY", 12, spec.edges["hy"]),
        emit_output_state_phase(),
        emit_output_gradient_phase(),
        emit_backpressure_phase(spec.edges["hy"]),
        emit_hidden_back_alias_phase(),
        *emit_hidden_gradient_phases(),
        *emit_history_phases(),
        *emit_error_update_phases(),
        *emit_weight_phases(spec),
        *emit_tick_and_reload_phases(),
    ]
    return "\n".join(part.rstrip() for part in phases) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, help="JSON network spec. Defaults to the XOR 2-2-2 spec.")
    parser.add_argument("--output", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--activation-lowering", choices=("tensor-op-unary", "pure"))
    args = parser.parse_args(argv)

    spec = load_spec(args.spec, args.activation_lowering)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(derive_rules(spec))


if __name__ == "__main__":
    main()

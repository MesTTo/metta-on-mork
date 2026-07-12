#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
GOAL: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "_")
DEFAULT_TARGET_DEPTHS = (8, 8, 12, 12, 16, 20)
DEFAULT_SEED = 20260712

MOVES: tuple[tuple[str, int], ...] = (
    ("U", -3),
    ("D", 3),
    ("L", -1),
    ("R", 1),
)


@dataclass(frozen=True)
class Instance:
    name: str
    state: tuple[str, ...]
    goal: tuple[str, ...]
    target_depth: int
    optimal_distance: int
    scramble_actions: tuple[str, ...]


def valid_swap(blank: int, delta: int) -> bool:
    target = blank + delta
    if target < 0 or target >= 9:
        return False
    if delta == -1 and blank % 3 == 0:
        return False
    if delta == 1 and blank % 3 == 2:
        return False
    return True


def apply_move(state: tuple[str, ...], action: str) -> tuple[str, ...]:
    delta_by_action = dict(MOVES)
    delta = delta_by_action[action]
    blank = state.index("_")
    if not valid_swap(blank, delta):
        raise ValueError(f"move {action} is invalid for {state}")
    target = blank + delta
    cells = list(state)
    cells[blank], cells[target] = cells[target], cells[blank]
    return tuple(cells)


def neighbors(state: tuple[str, ...]) -> Iterable[tuple[str, tuple[str, ...]]]:
    blank = state.index("_")
    for action, delta in MOVES:
        if valid_swap(blank, delta):
            yield action, apply_move(state, action)


def bfs_distances(seed: tuple[str, ...] = GOAL) -> dict[tuple[str, ...], int]:
    distances = {seed: 0}
    queue: deque[tuple[str, ...]] = deque([seed])
    while queue:
        state = queue.popleft()
        next_depth = distances[state] + 1
        for _, nxt in neighbors(state):
            if nxt not in distances:
                distances[nxt] = next_depth
                queue.append(nxt)
    return distances


def optimal_distance(start: tuple[str, ...], goal: tuple[str, ...] = GOAL) -> int:
    if start == goal:
        return 0
    seen = {start}
    queue: deque[tuple[tuple[str, ...], int]] = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        for _, nxt in neighbors(state):
            if nxt == goal:
                return depth + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    raise ValueError("goal is unreachable from start")


def random_walk(
    rng: random.Random, length: int, seed: tuple[str, ...] = GOAL
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    state = seed
    previous: tuple[str, ...] | None = None
    actions: list[str] = []
    for _ in range(length):
        options = [(action, nxt) for action, nxt in neighbors(state) if nxt != previous]
        if not options:
            options = list(neighbors(state))
        action, nxt = rng.choice(options)
        previous = state
        state = nxt
        actions.append(action)
    return state, tuple(actions)


def instance_name(depth: int, ordinal: int) -> str:
    return f"d{depth:02d}{chr(ord('a') + ordinal)}"


def generate_instances(
    target_depths: Iterable[int] = DEFAULT_TARGET_DEPTHS,
    seed: int = DEFAULT_SEED,
) -> list[Instance]:
    rng = random.Random(seed)
    distances = bfs_distances(GOAL)
    used: set[tuple[str, ...]] = set()
    ordinals: dict[int, int] = {}
    instances: list[Instance] = []
    for target_depth in target_depths:
        for _ in range(200_000):
            state, actions = random_walk(rng, target_depth)
            distance = distances[state]
            if distance == target_depth and state not in used:
                ordinal = ordinals.get(target_depth, 0)
                ordinals[target_depth] = ordinal + 1
                used.add(state)
                instances.append(
                    Instance(
                        name=instance_name(target_depth, ordinal),
                        state=state,
                        goal=GOAL,
                        target_depth=target_depth,
                        optimal_distance=distance,
                        scramble_actions=actions,
                    )
                )
                break
        else:
            raise RuntimeError(f"could not find random-walk instance at depth {target_depth}")
    return instances


def state_sexpr(state: tuple[str, ...]) -> str:
    return "(state " + " ".join(state) + ")"


def write_instances(path: Path, instances: list[Instance]) -> None:
    path.write_text(
        json.dumps([asdict(instance) for instance in instances], indent=2) + "\n",
        encoding="utf-8",
    )


def load_instances(path: Path) -> list[Instance]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Instance(
            name=item["name"],
            state=tuple(item["state"]),
            goal=tuple(item["goal"]),
            target_depth=int(item["target_depth"]),
            optimal_distance=int(item["optimal_distance"]),
            scramble_actions=tuple(item["scramble_actions"]),
        )
        for item in raw
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="8-puzzle oracle for plan demos")
    parser.add_argument("--write-instances", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="*",
        default=list(DEFAULT_TARGET_DEPTHS),
        help="target optimal depths for generated scrambles",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    instances = generate_instances(args.targets, args.seed)
    if args.write_instances:
        write_instances(args.write_instances, instances)
    if args.json:
        print(json.dumps([asdict(instance) for instance in instances], indent=2))
    else:
        for instance in instances:
            print(
                f"{instance.name}: distance={instance.optimal_distance} "
                f"state={state_sexpr(instance.state)} "
                f"scramble={''.join(instance.scramble_actions)}"
            )


if __name__ == "__main__":
    main()

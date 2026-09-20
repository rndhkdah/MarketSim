"""T4.05 — static follow-up graph checks (§4.2)."""

from __future__ import annotations

from marketsim.core.errors import ConfigError
from marketsim.events.schema import EventSpec

# §4.2: Σ follow-up probabilities ≤ 0.9; DAG depth ≤ max_depth.
P_SUM_CAP = 0.9


def followup_graph(catalog: dict[str, EventSpec]) -> dict[str, list[tuple[str, float]]]:
    """Directed edges ``src → (dst, probability)`` in catalog id order."""
    graph: dict[str, list[tuple[str, float]]] = {eid: [] for eid in catalog}
    for eid in sorted(catalog):
        spec = catalog[eid]
        for fu in spec.followups:
            graph[eid].append((fu.event_id, float(fu.probability)))
    return graph


def check_subcritical(
    catalog: dict[str, EventSpec],
    *,
    max_depth: int = 3,
    p_sum_cap: float = P_SUM_CAP,
) -> None:
    """Reject cycles, per-node ``Σ p > p_sum_cap``, and DAG depth ``> max_depth``."""
    graph = followup_graph(catalog)
    for src, edges in graph.items():
        total = sum(p for _, p in edges)
        if total > p_sum_cap + 1e-12:
            raise ConfigError(
                f"event {src!r} follow-up probabilities sum to {total}, cap is {p_sum_cap}"
            )
        for dst, _p in edges:
            if dst not in catalog:
                raise ConfigError(f"dangling follow-up {dst!r} on event {src!r}")
    _assert_dag(graph)
    depth = _longest_path(graph)
    if depth > int(max_depth):
        raise ConfigError(f"follow-up DAG depth {depth} exceeds max_depth {max_depth}")


def _assert_dag(graph: dict[str, list[tuple[str, float]]]) -> None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def visit(node: str) -> None:
        color[node] = GRAY
        for dst, _p in graph.get(node, []):
            c = color.get(dst, WHITE)
            if c == GRAY:
                raise ConfigError(f"follow-up graph has a cycle at {node!r} → {dst!r}")
            if c == WHITE:
                visit(dst)
        color[node] = BLACK

    for node in sorted(graph):
        if color[node] == WHITE:
            visit(node)


def _longest_path(graph: dict[str, list[tuple[str, float]]]) -> int:
    """Longest hop-count in a DAG."""
    memo: dict[str, int] = {}

    def depth(node: str) -> int:
        if node in memo:
            return memo[node]
        kids = graph.get(node, [])
        if not kids:
            memo[node] = 0
            return 0
        best = 1 + max(depth(dst) for dst, _p in kids)
        memo[node] = best
        return best

    if not graph:
        return 0
    return max(depth(n) for n in graph)


def expected_cascade_size(catalog: dict[str, EventSpec], damping: float = 0.7) -> dict[str, float]:
    """Branching-process mean extra events by root (finite on a DAG)."""
    graph = followup_graph(catalog)
    memo: dict[tuple[str, int], float] = {}

    def size(node: str, depth: int) -> float:
        key = (node, depth)
        if key in memo:
            return memo[key]
        total = 1.0
        for dst, p in graph.get(node, []):
            total += float(p) * (float(damping) ** depth) * size(dst, depth + 1)
        memo[key] = total
        return total

    return {eid: size(eid, 0) for eid in catalog}

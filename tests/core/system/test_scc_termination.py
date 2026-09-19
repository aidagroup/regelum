"""SCC termination checked against explicit finite configuration graphs."""

import random
from itertools import product

import pytest
import z3

import regelum as rg
from regelum.core import _checks


class Bit(rg.Node):
    class State(rg.NodeState):
        x: bool = rg.var(init=False)


def check(phases, nodes=(), *, depth=None, max_depth=64):
    return _checks._check_c2star(tuple(phases), tuple(nodes), depth=depth, max_depth=max_depth)


def split_phases():
    from examples.compile_checks.c2star_split_writes import build_system

    return build_system()


def test_scc_selection_and_long_graph():
    assert _checks._cyclic_sccs(list("ABCD"), [("A", "B"), ("B", "A"), ("B", "C"), ("C", "D")]) == (
        ("A", "B"),
    )
    assert _checks._cyclic_sccs(
        list("ABCD"), [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("C", "D")]
    ) == (("A", "B", "C"),)
    assert _checks._cyclic_sccs(["A", "B"], [("A", "A")]) == (("A",),)
    names = [str(i) for i in range(2000)]
    assert not _checks._cyclic_sccs(names, list(zip(names, names[1:])))


def test_acyclic_and_singleton_loop():
    assert not check([rg.Phase("a", nodes=(), transitions=(rg.Goto(rg.terminate),))])
    issues = check([rg.Phase("a", nodes=(), transitions=(rg.Goto("a"),))])
    assert "N_S=1" in issues[0].message
    assert "reachable entry" in issues[0].message


def test_split_writes_and_early_unsat_with_large_bound():
    system = split_phases()
    assert system.compile_report.ok
    assert check(system.phases, system.nodes, max_depth=2)
    # N_S=8; no need to allocate/unroll all eight steps.
    assert not check(system.phases, system.nodes, max_depth=3)
    assert check(system.phases, system.nodes, depth=1)


def test_alternating_cycles_regression():
    bit = Bit()
    x = rg.V(Bit.State.x)
    phases = [
        rg.Phase("A", nodes=(), transitions=(rg.If(~x, "B"), rg.Else("C")), is_initial=True),
        rg.Phase("B", nodes=(bit,), transitions=(rg.If(x, "A"), rg.Else(rg.terminate))),
        rg.Phase("C", nodes=(bit,), transitions=(rg.If(~x, "A"), rg.Else(rg.terminate))),
    ]
    # A0 -> B0 -> A1 -> C1 -> A0 can repeat, although neither simple
    # phase cycle can repeat consecutively. The old depth=2 test passed.
    system = rg.PhasedReactiveSystem(phases=phases, c2star_depth=2, strict=False)
    assert not system.compile_report.ok
    assert any("C2*" in i.message for i in system.compile_report.issues)
    assert "N_S=6" in check(phases, [bit])[0].message


def test_read_only_guard_stays_fixed_and_guards_use_post_state():
    bit = Bit()
    x = rg.V(Bit.State.x)
    phases = [
        rg.Phase("A", nodes=(), transitions=(rg.If(x, "B"), rg.Else(rg.terminate))),
        rg.Phase("B", nodes=(), transitions=(rg.If(~x, "A"), rg.Else(rg.terminate))),
    ]
    assert not check(phases, [bit])
    # B may change x after entry: x=true in A, x=false after B.
    phases[1] = rg.Phase("B", nodes=(bit,), transitions=(rg.If(~x, "A"), rg.Else(rg.terminate)))
    # Still terminates: A retains false. If A also writes, the loop is feasible.
    assert not check(phases, [bit])
    phases[0] = rg.Phase("A", nodes=(bit,), transitions=(rg.If(x, "B"), rg.Else(rg.terminate)))
    assert check(phases, [bit])


def test_unknown_never_certifies(monkeypatch):
    class UnknownSolver:
        def add(self, *args):
            pass

        def check(self):
            return z3.unknown

        def reason_unknown(self):
            return "test resource limit"

    monkeypatch.setattr(_checks.z3, "Solver", UnknownSolver)
    issues = check([rg.Phase("A", nodes=(), transitions=(rg.Goto("A"),))])
    assert "UNKNOWN" in issues[0].message


def test_non_symbolic_internal_guard_is_inconclusive():
    phases = [
        rg.Phase("A", nodes=(), transitions=(rg.If(lambda state: True, "A"), rg.Else(rg.terminate)))
    ]
    assert "non-symbolic" in check(phases)[0].message


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        check([], max_depth=value)
    with pytest.raises(ValueError):
        check([], depth=value)


def test_unknown_domain_uses_budget_and_can_still_prove_unsat():
    class Number(rg.Node):
        class State(rg.NodeState):
            x: int = rg.var(init=0)

    number = Number()
    x = rg.V(Number.State.x)
    dead = rg.Phase("A", nodes=(number,), transitions=(rg.If(x < x, "A"), rg.Else(rg.terminate)))
    assert not check([dead], [number], max_depth=1)
    live = rg.Phase("A", nodes=(number,), transitions=(rg.If(x >= 0, "A"), rg.Else(rg.terminate)))
    assert "budget exhausted" in check([live], [number], max_depth=2)[0].message


def test_local_loop_is_not_reported_as_global_nontermination():
    bit = Bit()
    x = rg.V(Bit.State.x)
    phases = [
        rg.Phase(
            "start",
            nodes=(bit,),
            transitions=(rg.If(~x, "loop"), rg.Else(rg.terminate)),
            is_initial=True,
        ),
        rg.Phase("loop", nodes=(), transitions=(rg.If(x, "loop"), rg.Else(rg.terminate))),
    ]
    # The only reachable loop entry has x=false and exits; arbitrary x=true loops.
    issues = check(phases, [bit])
    assert "global nontermination requires a reachable entry" in issues[0].message


def test_c2star_public_budget_counts_scc_transitions():
    system = split_phases()
    assert rg.PhasedReactiveSystem(phases=system.phases, c2star_max_depth=3).compile_report.ok
    with pytest.raises(rg.CompileError):
        rg.PhasedReactiveSystem(phases=system.phases, c2star_max_depth=2)


def explicit_acyclic(table, writes):
    # Independent oracle: enumerate all configurations and update values,
    # then delete zero-indegree vertices. A remainder is a concrete cycle.
    vertices = list(product(range(len(table)), (False, True)))
    edges = {q: set() for q in vertices}
    for phase, value in vertices:
        for post in (False, True) if writes[phase] else (value,):
            target = table[phase][int(post)]
            if target is not None:
                edges[phase, value].add((target, post))
    indegrees = {q: 0 for q in vertices}
    for targets in edges.values():
        for target in targets:
            indegrees[target] += 1
    pending = [q for q in vertices if indegrees[q] == 0]
    count = 0
    while pending:
        q = pending.pop()
        count += 1
        for target in edges[q]:
            indegrees[target] -= 1
            if indegrees[target] == 0:
                pending.append(target)
    return count == len(vertices)


@pytest.mark.parametrize("seed", range(100))
def test_matches_exhaustive_configuration_graph(seed):
    rng = random.Random(seed)
    count = rng.randint(1, 4)
    table = [[rng.choice([None, *range(count)]) for _ in range(2)] for _ in range(count)]
    writes = [rng.choice([False, True]) for _ in range(count)]
    bit = Bit()
    x = rg.V(Bit.State.x)
    phases = [
        rg.Phase(
            str(i),
            nodes=(bit,) if writes[i] else (),
            transitions=(
                rg.If(~x, rg.terminate if targets[0] is None else str(targets[0])),
                rg.Else(rg.terminate if targets[1] is None else str(targets[1])),
            ),
        )
        for i, targets in enumerate(table)
    ]
    assert (not check(phases, [bit])) == explicit_acyclic(table, writes)


def test_finite_integer_domain_sets_bound():
    class Counter(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0, domain=(0, 1, 2))

    counter = Counter()
    value = rg.V(Counter.State.value)
    phase = rg.Phase(
        "A", nodes=(counter,), transitions=(rg.If(value >= 0, "A"), rg.Else(rg.terminate))
    )
    assert "N_S=3" in check([phase], [counter])[0].message


def test_all_components_must_pass():
    bit = Bit()
    x = rg.V(Bit.State.x)
    phases = [
        rg.Phase("dead", nodes=(bit,), transitions=(rg.If(x & ~x, "dead"), rg.Else("live"))),
        rg.Phase("live", nodes=(), transitions=(rg.Goto("live"),)),
    ]
    issues = check(phases, [bit])
    assert len(issues) == 1
    assert issues[0].location == "SCC {live}"

"""Multi-variable SCC certificates, mutations, and independent finite-state oracle.

The oracle evaluates Python predicates over explicit valuations; it does not use
Z3, symbolic guard evaluation, or the production SCC algorithm.
"""

import random
from dataclasses import dataclass
from itertools import product
from time import perf_counter

import pytest

import regelum as rg
from regelum.core._checks import _check_c2star


@dataclass
class Model:
    nodes: tuple
    domains: tuple
    # phase -> (written variable indices, [(symbolic guard, Python guard, target)])
    specs: dict

    def phases(self):
        return tuple(
            rg.Phase(
                name,
                nodes=tuple(self.nodes[i] for i in writes),
                transitions=tuple(
                    (rg.If if j == 0 else rg.ElseIf)(guard, target or rg.terminate)
                    for j, (guard, _, target) in enumerate(edges)
                )
                + (rg.Else(rg.terminate),),
                is_initial=(index == 0),
            )
            for index, (name, (writes, edges)) in enumerate(self.specs.items())
        )

    def check(self, budget):
        return _check_c2star(self.phases(), self.nodes, depth=None, max_depth=budget)


def variables(*sizes):
    nodes = []
    for i, size in enumerate(sizes):

        class Register(rg.Node):
            class State(rg.NodeState):
                value: int = rg.var(init=0, domain=tuple(range(size)))

        nodes.append(Register(name=f"v{i}"))
    return tuple(nodes), tuple(rg.V(f"v{i}.value") for i in range(len(sizes)))


def oracle(model):
    """Return longest path length, or None for a cycle, in the full graph."""
    values = list(product(*model.domains))
    vertices = list(product(model.specs, values))
    edges = {vertex: set() for vertex in vertices}
    for name, state in vertices:
        writes, transitions = model.specs[name]
        for assignment in product(*(model.domains[i] for i in writes)):
            post = list(state)
            for i, value in zip(writes, assignment):
                post[i] = value
            for _, predicate, target in transitions:
                if predicate(post):
                    if target is not None:
                        edges[name, state].add((target, tuple(post)))
                    break
    degree = dict.fromkeys(vertices, 0)
    longest = dict.fromkeys(vertices, 0)
    for targets in edges.values():
        for target in targets:
            degree[target] += 1
    pending = [v for v in vertices if degree[v] == 0]
    visited = 0
    while pending:
        vertex = pending.pop()
        visited += 1
        for target in edges[vertex]:
            longest[target] = max(longest[target], longest[vertex] + 1)
            degree[target] -= 1
            if degree[target] == 0:
                pending.append(target)
    return max(longest.values(), default=0) if visited == len(vertices) else None


def assert_boundary(model, record_property):
    start = perf_counter()
    length = oracle(model)
    record_property("oracle_seconds", perf_counter() - start)
    assert length is not None and length > 1
    start = perf_counter()
    # These models consist of one phase SCC, so full and internal lengths agree.
    assert model.check(length), "SAT at the longest feasible path must not pass"
    assert not model.check(length + 1), "The next depth must certify termination"
    record_property("smt_seconds_two_checks", perf_counter() - start)
    record_property("first_unsat_depth", length + 1)
    record_property("full_configurations", len(model.specs) * len(list(product(*model.domains))))
    return length


def fanout(size, *, mutation=False):
    nodes, (x, y) = variables(size, size)
    # Every return through the hub decreases y. Distinct spokes can be chosen
    # on successive visits, so the proof must cover arbitrary cycle switching.
    hub = [
        (
            (x == i) & ((x <= y) if mutation else (x < y)),
            lambda s, i=i: s[0] == i and (s[0] <= s[1] if mutation else s[0] < s[1]),
            f"b{i}",
        )
        for i in range(size)
    ]
    specs = {"hub": ((0,), hub)}
    specs.update({f"b{i}": ((1,), [(y <= x, lambda s: s[1] <= s[0], "hub")]) for i in range(size)})
    return Model(nodes, (range(size), range(size)), specs)


@pytest.mark.parametrize("size", [4, 8, 16])
def test_many_switchable_cycles_with_decreasing_registers(size, record_property):
    model = fanout(size)
    assert assert_boundary(model, record_property) == 2 * size - 1
    # Exercise the public compiler, including C1/C3, not just the private checker.
    assert rg.PhasedReactiveSystem(
        phases=model.phases(), c2star_max_depth=2 * size
    ).compile_report.ok


def test_one_character_guard_mutation_makes_fanout_infinite():
    model = fanout(4, mutation=True)
    assert oracle(model) is None
    issues = model.check(5 * 4**2)
    assert len(issues) == 1 and "N_S=80" in issues[0].message
    with pytest.raises(rg.CompileError):
        rg.PhasedReactiveSystem(phases=model.phases(), c2star_max_depth=80)


def nested(size, *, mutation=False):
    nodes, (x, y, u, v) = variables(size, size, size, size)
    specs = {
        "outer": ((0,), [(x < y, lambda s: s[0] < s[1], "reset")]),
        "reset": ((2, 3), [(u >= 0, lambda s: s[2] >= 0, "inner")]),
        "inner": (
            (2,),
            [
                (
                    (u <= v) if mutation else (u < v),
                    lambda s: s[2] <= s[3] if mutation else s[2] < s[3],
                    "inner_return",
                ),
                (u >= v, lambda s: s[2] >= s[3], "outer_return"),
            ],
        ),
        "inner_return": ((3,), [(v <= u, lambda s: s[3] <= s[2], "inner")]),
        "outer_return": ((1,), [(y <= x, lambda s: s[1] <= s[0], "outer")]),
    }
    return Model(nodes, (range(size),) * 4, specs)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_nested_loops_with_inner_reset(size, record_property):
    # The inner counters may jump upwards at reset: a single global monotone
    # counter is not assumed. Progress of the outer loop still bounds the run.
    model = nested(size)
    length = assert_boundary(model, record_property)
    assert length >= size**2
    assert rg.PhasedReactiveSystem(
        phases=model.phases(), c2star_max_depth=length + 1
    ).compile_report.ok


def test_inner_livelock_survives_outer_progress():
    model = nested(2, mutation=True)
    assert oracle(model) is None
    issues = model.check(5 * 2**4)
    assert len(issues) == 1 and "N_S=80" in issues[0].message


def test_three_sequential_sccs_require_separate_certificates():
    model = fanout(3)
    specs = {}
    for block in range(3):
        prefix = f"{block}:"
        for name, (writes, transitions) in model.specs.items():
            edges = [(g, p, prefix + t) for g, p, t in transitions]
            if block < 2:
                # Explicit default exit transfers to the next SCC.
                edges.append((rg.V("v0.value") >= 0, lambda s: True, f"{block + 1}:hub"))
            specs[prefix + name] = (writes, edges)
    model.specs = specs
    assert oracle(model) is not None
    assert not model.check(6)
    # Replace only the final hub's strict comparison: earlier SCC certificates
    # must not hide a loop in the last component.
    live = fanout(3, mutation=True)
    writes, transitions = live.specs["hub"]
    model.specs["2:hub"] = (writes, [(g, p, "2:" + t) for g, p, t in transitions])
    assert oracle(model) is None
    issues = model.check(36)
    assert len(issues) == 1 and "2:hub" in issues[0].location


@pytest.mark.parametrize("seed", range(30))
def test_mixed_domains_and_multiple_writers_against_oracle(seed):
    rng = random.Random(seed)
    sizes = (2, 3, 4)
    nodes, symbols = variables(*sizes)
    specs = {}
    phase_count = 5
    valuations = list(product(*(range(n) for n in sizes)))
    # Generate explicit guard partitions, with multi-variable updates and
    # shared ownership across phases. Sparse routing also produces dead SCCs.
    for phase in range(phase_count):
        writes = tuple(i for i in range(3) if rng.random() < 0.5)
        transitions = []
        for valuation in valuations:
            target = str(rng.randrange(phase_count)) if rng.random() < 0.18 else None
            guard = symbols[0] == valuation[0]
            for i in range(1, 3):
                guard = guard & (symbols[i] == valuation[i])
            transitions.append((guard, lambda s, val=valuation: tuple(s) == val, target))
        specs[str(phase)] = (writes, transitions)
    model = Model(nodes, tuple(range(n) for n in sizes), specs)
    expected = oracle(model)
    assert (not model.check(phase_count * 24)) == (expected is not None)

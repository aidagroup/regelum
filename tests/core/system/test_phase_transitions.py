import pytest

import regelum as rg


def test_transition_target_accepts_phase_instance() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

        def update(self) -> State:
            return self.State(ready=True)

    class Done(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    source = Source()
    done_node = Done()
    done = rg.Phase("done", nodes=(done_node,), transitions=(rg.Goto(rg.terminate),))
    start = rg.Phase(
        "start",
        nodes=(source,),
        transitions=(
            rg.If(rg.V(Source.State.ready), done, name="ready"),
            rg.If(~rg.V(Source.State.ready), rg.terminate, name="not-ready"),
        ),
        is_initial=True,
    )
    system = rg.PhasedReactiveSystem(phases=[start, done])

    system.step()

    assert [record.phase for record in system.history] == ["start", "done"]
    assert system.snapshot()["Done.reached"] is True


def test_phase_transition_guards_may_read_state_vars_outside_active_phase() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

        def update(self) -> State:
            return self.State(ready=True)

    class Other(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)

        def update(self) -> State:
            return self.State(value=1)

    source = Source()
    other = Other()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(other, source),
                transitions=(
                    rg.If(rg.V(Source.State.ready), rg.terminate, name="ready"),
                    rg.If(~rg.V(Source.State.ready), rg.terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok
    system.step()
    assert system.snapshot()["Other.value"] == 1


def test_phase_transition_guard_accepts_instance_state_reference() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

    source = Source(name="source_a")
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(source,),
                transitions=(
                    rg.If(rg.V(source.State.ready), rg.terminate, name="ready"),
                    rg.If(~rg.V(source.State.ready), rg.terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_phase_transition_guard_rejects_ambiguous_class_state_reference() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

    source_a = Source(name="source_a")
    source_b = Source(name="source_b")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "check",
                    nodes=(source_a, source_b),
                    transitions=(
                        rg.If(rg.V(Source.State.ready), rg.terminate, name="ready"),
                        rg.If(~rg.V(Source.State.ready), rg.terminate, name="not-ready"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "check.ready"
        and "ambiguous guard variable 'Source.ready'" in issue.message
        and "source_a.ready" in issue.message
        and "source_b.ready" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_phase_predicate_routing() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: float = rg.var(init=1.0)

        def update(self, inputs: rg.NodeInputs) -> State:
            return self.State(value=2.0)

    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            value: float = rg.src(Source.State.value)

        class State(rg.NodeState):
            seen: float = rg.var(init=0.0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "source",
                nodes=(source,),
                transitions=(
                    rg.If(
                        lambda state: state["Source.value"] > 1.0,
                        "sink",
                        name="value-ready",
                    ),
                ),
                is_initial=True,
            ),
            rg.Phase("sink", nodes=(sink,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Sink.seen"] == 2.0


def test_compile_rejects_c3_ambiguous_predicates() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "start",
                    nodes=(),
                    transitions=(
                        rg.If(lambda _: True, rg.terminate, name="first"),
                        rg.If(lambda _: True, rg.terminate, name="second"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "start" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )


def test_if_elseif_else_chain_uses_ordered_fallback_semantics() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=2, domain=(0, 1, 2))

        def update(self) -> State:
            return self.State(value=2)

    class One(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Two(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Fallback(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    mode = Mode()
    one = One()
    two = Two()
    fallback = Fallback()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.value) == 1, "one", name="one"),
                    rg.ElseIf(rg.V(Mode.State.value) == 2, "two", name="two"),
                    rg.Else("fallback"),
                ),
                is_initial=True,
            ),
            rg.Phase("one", nodes=(one,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("two", nodes=(two,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("fallback", nodes=(fallback,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Two.reached"] is True
    assert system.snapshot()["One.reached"] is False
    assert system.snapshot()["Fallback.reached"] is False


def test_if_chain_order_gives_precedence_over_later_elseif() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=2, domain=(0, 1, 2))

        def update(self) -> State:
            return self.State(value=2)

    class First(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Second(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    mode = Mode()
    first = First()
    second = Second()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.value) > 0, "first", name="first"),
                    rg.ElseIf(rg.V(Mode.State.value) > 1, "second", name="second"),
                    rg.Else(rg.terminate),
                ),
                is_initial=True,
            ),
            rg.Phase("first", nodes=(first,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("second", nodes=(second,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["First.reached"] is True
    assert system.snapshot()["Second.reached"] is False


def test_multiple_if_chains_overlap_is_c3_violation() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            a: bool = rg.var(init=True)
            b: bool = rg.var(init=True)

    with pytest.raises(rg.CompileError) as exc_info:
        mode = Mode()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "select",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V(Mode.State.a), rg.terminate, name="a"),
                        rg.If(rg.V(Mode.State.b), rg.terminate, name="b"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "select" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )


def test_multiple_if_chains_can_be_disjoint() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            a: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(a=False)

    mode = Mode()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.a), rg.terminate, name="a"),
                    rg.If(~rg.V(Mode.State.a), rg.terminate, name="not-a"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok
    system.step()


def test_goto_is_unconditional() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)

        def update(self) -> State:
            return self.State(value=1)

    class Sink(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "source",
                nodes=(source,),
                transitions=(rg.Goto("sink"),),
                is_initial=True,
            ),
            rg.Phase("sink", nodes=(sink,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Source.value"] == 1
    assert system.snapshot()["Sink.reached"] is True


def test_goto_cannot_be_mixed_with_if_chains() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=True)

    with pytest.raises(rg.CompileError) as exc_info:
        source = Source()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "mixed",
                    nodes=(source,),
                    transitions=(
                        rg.If(rg.V(Source.State.flag), rg.terminate),
                        rg.Goto(rg.terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.message == "Goto transitions cannot be mixed with If/ElseIf/Else chains"
        for issue in exc_info.value.report.issues
    )


def test_elseif_and_else_require_open_if_chain() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=True)

    with pytest.raises(rg.CompileError) as exc_info:
        source = Source()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "bad",
                    nodes=(source,),
                    transitions=(
                        rg.ElseIf(rg.V(Source.State.flag), rg.terminate),
                        rg.Else(rg.terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    messages = [issue.message for issue in exc_info.value.report.issues]
    assert "ElseIf must follow If or ElseIf" in messages
    assert "Else must follow If or ElseIf" in messages


def test_elseif_after_else_is_compile_error() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=True)

    with pytest.raises(rg.CompileError) as exc_info:
        source = Source()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "bad",
                    nodes=(source,),
                    transitions=(
                        rg.If(rg.V(Source.State.flag), rg.terminate),
                        rg.Else(rg.terminate),
                        rg.ElseIf(~rg.V(Source.State.flag), rg.terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif" and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_second_else_is_compile_error() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=True)

    with pytest.raises(rg.CompileError) as exc_info:
        source = Source()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "bad",
                    nodes=(source,),
                    transitions=(
                        rg.If(rg.V(Source.State.flag), rg.terminate),
                        rg.Else(rg.terminate, name="first-else"),
                        rg.Else(rg.terminate, name="second-else"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.second-else" and issue.message == "Else must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_if_chain_without_total_coverage_is_c3_violation() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=False)

    with pytest.raises(rg.CompileError) as exc_info:
        source = Source()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "select",
                    nodes=(source,),
                    transitions=(rg.If(rg.V(Source.State.flag), rg.terminate),),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "select"
        and issue.message.startswith("C3 violation: no transition is enabled")
        for issue in exc_info.value.report.issues
    )


def test_if_after_else_is_reported_as_warning() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            a: bool = rg.var(init=False)
            b: bool = rg.var(init=True)

    mode = Mode()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.a), rg.terminate, name="a"),
                    rg.Else("next", name="else-a"),
                    rg.If(rg.V(Mode.State.b), rg.terminate, name="b"),
                ),
                is_initial=True,
            ),
            rg.Phase("next", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
        strict=False,
    )

    assert system.compile_report.ok
    assert not system.compile_report.issues
    assert any(
        issue.location == "select.b" and issue.message.startswith("transition follows Else")
        for issue in system.compile_report.warnings
    )


def test_if_after_else_starts_new_chain_that_can_have_else() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            a: bool = rg.var(init=False)
            b: bool = rg.var(init=True)

    mode = Mode()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.a), rg.terminate, name="a"),
                    rg.Else("next", name="else-a"),
                    rg.If(rg.V(Mode.State.b), rg.terminate, name="b"),
                    rg.Else(rg.terminate, name="else-b"),
                ),
                is_initial=True,
            ),
            rg.Phase("next", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
        strict=False,
    )

    assert system.compile_report.ok
    assert any(
        issue.location == "select.b" and issue.message.startswith("transition follows Else")
        for issue in system.compile_report.warnings
    )

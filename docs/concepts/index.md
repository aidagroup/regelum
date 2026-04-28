# Concepts

`regelum` is a framework for modeling **Phased Reactive Systems**.
What that means is best understood through a concrete example, working from
the top down: starting with a continuously running system, breaking it into
something executable, and then descending one level at a time until we reach
the primitives the framework actually asks you to write.

!!! example "The running example: an adaptive-bitrate video player"

    Every adult has watched a video that quietly dropped from 1080p to 480p
    when the network slowed down.
    Let us think about how to model this process.

## The feedback loop

A video player never finishes by itself.
While you watch the current second of video, the player is also quietly
downloading the next few seconds in the background, so that playback does
not have to wait for the network on every frame.
That backlog of already-downloaded but not yet shown video is what we will
call the **buffer** — measured in seconds of viewable content sitting in
memory ahead of the playhead.

If the network is fast, the buffer grows and the player has slack.
If the network slows down, the buffer shrinks; if it runs out completely,
playback stalls and the user sees a spinner.
The player's job is to make sure that does not happen — it watches how much
buffer is left, decides whether the current quality is sustainable, lowers
the quality if not, plays the next chunk, and then *goes back to watching
the buffer*.

This is a **feedback loop**: the result of one pass becomes the input of the
next, and the system runs forever as long as it is alive.

```mermaid
flowchart LR
    measure --> decide{decide}
    decide -->|healthy| play
    decide -->|stalling| drop_quality
    drop_quality --> play
    play -->|next pass| measure
```

## Breaking the cycle into ticks

A diagram with a back-edge tells us what the system *is*, but not how to
*execute* it.
To run it, we cut the loop into a unit of work that has a clear start *and* a
clear end.
We declare an **initial** point — where every pass enters the graph — and a
terminator `⊥` — where every pass leaves it:

```mermaid
flowchart LR
    init([init]) --> measure
    measure --> decide{decide}
    decide -->|healthy| play
    decide -->|stalling| drop_quality
    drop_quality --> play
    play --> done([⊥])
```

One such pass through the graph — from the initial point to `⊥` — is a
**tick**.
A tick is the unit of execution.

`⊥` is **not** the end of the system.
It is the end of one tick.
The original feedback loop is recovered by running tick after tick, with the
state values from the previous tick carried over into the next.
The cycle now lives *outside* the graph — between successive ticks — instead
of being drawn as an explicit edge.

## The high-level graph: phases and transitions

The graph above is still high-level: it tells us the shape of one tick, not
yet what each box actually does.

The boxes (`measure`, `decide`, `drop_quality`, `play`) are called
**phases**.
A phase is a labelled stage of one tick.

Each arrow leaving a phase carries a **predicate** that is evaluated at
runtime to decide whether that arrow fires:

- if a phase has *one* outgoing arrow, its predicate is trivially `true` —
  the arrow always fires, and we draw it as `Goto`.
- if a phase has *several* outgoing arrows, the predicates must be mutually
  exclusive — exactly one fires per tick.

In other words, **every arrow is conditional**; an unconditional `Goto` is
just the special case where the condition is `True`.
This unifies sequential flow and branching under a single rule: at the end of
each phase, evaluate the predicates and follow the one that matches.

In the player, `decide` has two outgoing arrows:

- `If(stalling)` → `drop_quality`
- `Else` → `play`

while `measure`, `drop_quality`, and `play` each have a single arrow that
always fires.

## Phases up close: nodes

So far we have only described the *shape* of one tick: which phases exist,
which arrows connect them, which predicates gate the arrows.
That is enough to draw a diagram, but not enough to actually execute
anything.
To execute the system we need to make the high-level boxes concrete: what
variables exist, who reads them, who writes them, and what computation
happens inside each phase.

A phase is a high-level story.
It can be arbitrarily complex, and it is built out of smaller primitives
called **nodes**.
A node is an atomic unit of computation.

Every node has two kinds of variables:

- **inputs** — variables the node *reads*.
  An input is always the output of *some other* node (or the node's own
  output from a previous tick — that is how persistent state and feedback
  are expressed).
- **outputs** — variables the node *writes*.
  Each output is owned by exactly one node, so there is never any ambiguity
  about who produced a given value.

Inside a phase, several nodes can be active.
They are scheduled in topological order from their input/output dependencies,
so that every read sees a freshly written value when there is one.

For the video player, here is how the high-level phases decompose into
nodes:

| Phase | Nodes | What happens |
|---|---|---|
| `measure` | `Clock`, `Network` | Advance the tick counter; sample the current bandwidth. |
| `decide` | `QualityPolicy` | Compare projected drain rate against the buffer; set `stalling`. |
| `drop_quality` | `BitrateController` | Drop the target bitrate by one rung. |
| `play` | `Decoder`, `MediaSession`, `Logger` | Compute downloaded seconds, integrate the buffer, log. |

This is where the framework's actual work happens: writing node classes,
declaring their inputs and outputs, assigning instances to phases, and
attaching predicates to transitions.

??? example "Full code listing: `examples/video_player.py`"

    ```python
    --8<-- "examples/video_player.py"
    ```

## Where to go next

The remaining pages in this section walk through the model from the bottom
up — same model, opposite direction:

1. **Nodes** — declaring typed inputs, typed outputs, and a `run` method.
2. **Ports** — the input and output declarations that make a node's
   interface explicit.
3. **Connections** — wiring an input to a specific output, including
   instance-level wiring for multi-instance systems.
4. **State and initialization** — which outputs need `initial` values and
   why, and how persistent state closes the feedback loop across ticks.
5. **Phases** — assembling node instances into phases and reading the
   compiled schedule.
6. **Transitions** — `Goto`, `If` / `Elif` / `Else`, predicates, and
   `terminate`.
7. **Compilation** — what is checked statically (C1, C3) and what the
   compile report exposes.
8. **Runtime** — how `step()` and `run()` execute compiled phases.

## Three times to keep separate

There are three distinct moments in the life of a system:

- **declaration time** — Python classes declare ports and behavior;
- **compile time** — node instances, sources, phases, and guards are
  resolved into an executable model;
- **runtime** — phases execute and state values are updated.

Most mistakes should be caught at compile time.
`PhasedReactiveSystem(..., strict=False)` lets a bad system be inspected via
its compile report instead of raising.

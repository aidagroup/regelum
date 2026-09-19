# SCC termination: nontrivial regression results

Measured on 2026-09-19 against the working SCC implementation based on Regelum `204a091`.

## Reproduce

```sh
uv run pytest tests/core/system/test_scc_nontrivial.py -q --durations=12 -o junit_family=legacy --junitxml=/tmp/scc-nontrivial.xml
uv run pytest tests --ignore=tests/core/system/test_scc_nontrivial.py -q
```

The new suite passed all 39 tests in 208.27 seconds. The remaining suite passed all 283 tests in 1.51 seconds (322 tests in total, run in two batches). Ruff and the project type check passed. The original XML run used xunit2 and emitted six metadata-format warnings for `record_property`; the reproduction command selects the compatible format.

## Termination certificates

Each row checks SAT at the longest feasible internal path, UNSAT at the next depth, and acceptance by the public compiler with C1/C3 checks. The oracle explicitly enumerates domain-valid updates and evaluates separate Python predicates. It does not call Z3 or the production SCC traversal.

| Model | Full configurations | First UNSAT depth | Two incremental SMT checks, seconds | Explicit oracle, seconds |
| --- | ---: | ---: | ---: | ---: |
| Switchable cycles, domain size 4, 5 phases | 80 | 8 | 0.078 | 0.0006 |
| Switchable cycles, domain size 8, 9 phases | 576 | 16 | 0.373 | 0.0047 |
| Switchable cycles, domain size 16, 17 phases | 4352 | 32 | 2.839 | 0.0628 |
| Nested reset loops, domain size 2, 5 phases, 4 variables | 80 | 12 | 0.111 | 0.0003 |
| Nested reset loops, domain size 3, 5 phases, 4 variables | 405 | 24 | 0.444 | 0.0023 |
| Nested reset loops, domain size 4, 5 phases, 4 variables | 1280 | 40 | 1.430 | 0.0104 |

The switchable family has a hub and multiple return cycles; guards force progress regardless of the selected return cycle. The nested family resets both inner counters on every outer iteration, so individual counters need not decrease on every transition. Updates remain arbitrary within declared domains, as required by F_max; the tests do not assume a Python decrement implementation.

## Nontermination and component isolation

- Changing a strict `<` guard to `<=` makes the switchable family recurrent; the checker returns SAT at N_S=80 and strict compilation rejects it.
- The same mutation in the inner loop permits livelock despite potential outer-loop progress; SAT at N_S=80 is detected.
- Three sequential SCCs certify separately. Mutating only the final SCC leaves exactly that component uncertified.
- Thirty seeded models have five phases, three variables with domain sizes 2, 3, and 4, and 120 guard alternatives. Phases may write multiple variables and share written variables with other phases. Every checker verdict agrees with the independently enumerated graph.

## Timing scope

The slowest test took 8.88 seconds. Test durations include model construction, formula construction, solver calls, and the explicit oracle; public-compiler tests also include an additional compiler run. The table separately measures the two incremental checker runs used to establish each exact boundary. These are local measurements, not runtime guarantees. On these small finite models, explicit enumeration is faster than SMT. No claim about industrial-model scalability follows from this suite.

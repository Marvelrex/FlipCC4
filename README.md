# CC4 asymmetric-game testbed (reviewer edition)

This repository is the **runnable testbed** behind the submission's experiments, plus a
one-command **offline functional smoke**. It lets a reviewer verify that the mechanism described
in the paper exists and behaves as stated: CAGE Challenge 4 (CybORG 4.0) resource control with
holding payoff and completion-billed action cost, the legal observation/action interface of the
model-driven agents, the configuration adapter, the one-shot B0->B1 whole-team switch with a
shared budget, and the shared-prefix fork (checkpoint, restore, branch continuation).

It is **not** a reproduction package for the paper's results. It contains no model client, no
API key handling, no formal match data and no batch/analysis pipeline, and the smoke runs with
deterministic offline stand-ins for the language models. Re-running the paper's paid experiments
is out of scope.

## Install

Tested with CPython 3.10 on Windows; the code is pure Python and should run on Linux/macOS with the
same pins.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      POSIX: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins only the simulator runtime (numpy, gym, gymnasium, networkx, PyYAML,
prettytable, pygame). After installation the smoke needs **no network and no credentials**.

## Run the smoke

```bash
python smoke.py
```

- Exit code `0` only if every check passed; any failure gives exit code `1`.
- A JSON receipt is written to `receipts/smoke-<UTC>.json` with the per-check results, the
  Python/package versions, the CybORG version and a sha256 of the shipped source tree.
- `--keep-going` runs all checks after a failure; `--config` selects another configuration
  document (default `testbed/configs/smoke.json`); `--out` changes the receipt directory.
- The process re-executes itself with `PYTHONHASHSEED=0`, which the cross-process checkpoint
  check needs (restored set-iteration order must agree between processes).
- The whole run takes well under a minute on a laptop.

Network access is blocked **inside the process** for the whole run (`socket.socket`,
`socket.create_connection`, `socket.getaddrinfo` and `urllib.request.urlopen` are replaced by
functions that raise), and the shipped sources are scanned for network/credential markers. The
acceptance receipts in `receipts/` were produced in a fresh directory with a fresh virtual
environment (see "Acceptance" below).

## What the smoke verifies

| check | what it establishes (all on the real CC4 environment) |
|---|---|
| `network_guard` | runtime block active; no network or credential markers in `flipthem/` or `testbed/`; no real client module shipped |
| `environment_startup` | scenario builds from a seed; the two scored hosts exist; five Blue slots; mission phase stays 0; same seed -> same host set; the wrapper clock reaches the horizon |
| `legal_action_interface` | legal (action, target) pairs are enumerated from the slot's own action space; a listed pair compiles to the native action, unlisted targets and other-team actions compile to `InvalidAction`; a busy slot is not decidable; the context/tool/prompt module sources contain none of the ground-truth access markers (`environment_controller`, `compromise_of`, `get_true_state`, `.state.`); a model answer naming an unlisted target is counted as illegal and billed 0.1 once; legal stub decisions become native actions |
| `holding_payoff` | per-tick value is split by live control of the scored hosts (all Blue at t0, flips when a Red root session exists); in a 30-tick match where Red gains a scored host at tick 10 the control vector flips exactly then, per-tick value is constant-sum, `U_side*T + cost_side = held value` for both sides and `U_R + U_B = sum_v - (cost_R + cost_B)/T` |
| `completed_action_cost` | `Restore` is billed 1.0 exactly once, on the tick it completes, never while in flight; a resubmission to a busy slot is not a second action; an action still in flight at the horizon is not billed; the passive execution ledger has exactly one row for it; attaching the ledger does not change payoff, billing or control trace |
| `switch_b0_to_b1` | under the deadline rule the whole team switches once, at the first legal decision opportunity at/after the deadline, with a Blue action in flight; B0 makes no call after the switch and B1 none before; both components share one ledger, one pending map and one request budget (`spent` = all physical calls); the in-flight action resolves once on the shared ledger and has exactly one executed ledger row before the slot's next submission, and the slot's billed total grows by exactly its ledger rows; the remaining cost cap is inherited (`cap - spent_B0`); at the switch instant B1's wake clock is the carried B0 clock (or the switch tick), never its never-called sentinel; a never-firing selector is indistinguishable from the plain B0 policy (same per-slot actions with parameters, payoff and call counts) |
| `fork_save_restore` | the live match is checkpointed at that same instant; live vs restored agree in content; two restores agree in content **and** iteration order; the copy carries the in-flight action, wake state, budget, environment RNG state and the shared-object aliasing; branches are independent (advancing one leaves the other's fingerprint unchanged; its clients make no call); clients are never serialised (a client without a handle is refused; the checkpoint bytes contain no client); a restore in a **second process** yields the same fingerprint in both modes |
| `fork_continuation` | two B0 continuations of the same checkpoint are identical tick by tick (observations, actions, trace, payoff, billing, prompts, ledger) and identical to the never-forked live continuation; the B branch activates B1 at the fork tick and reaches the horizon; the in-flight action settles exactly once in every branch; the post-fork segment value re-derived from the control trace and the segment cost re-derived from the execution ledger equal the wrapper's accumulators, and with them `U*T = (v-c)_prefix + U_seg*n` holds for both sides; prefix + branch B reproduces an unforked match run under the same deadline rule (same control trace and payoff) |

The counter-example path is also exercised in the acceptance: a configuration with a zero
per-request cost makes `switch_b0_to_b1` fail its inherited-cap assertion and the command exits 1.

## Interfaces

**Configuration adapter** (`testbed/config.py`). One JSON document describes a match:

```json
{"schema": "cc4-testbed-config-v1", "seed": 42,
 "settings": {"steps": 40, "mode": "ft", "max_tool_calls": 4, "max_idle_ticks": 10,
              "max_retries": 0, "budget_limit": 400, "max_cost_usd": 5.0, "framing": "neutral"},
 "client": {"kind": "offline-stub", "usage_cost_usd": 0.001},
 "blue": {"kind": "switch", "start": {"id": "B0"}, "other": {"id": "B1"},
          "arm": "deadline", "event_enabled": false, "deadline_tick": 20},
 "red": {"kind": "llm", "id": "R0"}}
```

A side is `llm` (an assistance profile: `B0` = context-assisted, `B1` = context-tool-assisted,
`R0` = context-tool-assisted, `R1` = campaign-feedback, or any registered `strategy`), `switch`
(two `llm` components sharing one request budget under a frozen `deadline`/`event` selector),
`scripted` (constrained rules on the legal interface: Blue `S0`/`S1`/`S2`, Red `A0`) or `native`
(an upstream CybORG anchor such as Red `fsm`). `build_match(cfg, client_factory)` returns the
policies, a reset environment and a passive execution recorder.

**Model client contract** (`flipthem/llm/backend.py`). A client is any object with
`next(system, user, transcript, force_final) -> dict` returning `{"type": "final", "text": <JSON
decisions>, "usage": {...}}` or a tool-call response. Clients are always injected through
`client_factory(team, token, model)`; `testbed/fake_client.py` ships a deterministic
`OfflineStubClient` (a pure function of the prompt that picks among the listed legal options) and
a `ScriptedClient` (fixed replies). The four read-only tools and the prompt renderer are unchanged
from the research artefact; the rendered prompt surfaces were hash-compared at extraction time.

**Mechanism** (`flipthem/wrapper.py`). `FlipThemCC4(seed, steps, mode="ft").reset()`, then
`step(blue_actions, red_actions)` once per tick with per-slot native actions (`Sleep()` for idle
slots); `team_observation(team)` is the legal per-team observation; `payoff()` returns
`U_B`, `U_R`, `sum_v`, `cost_B`, `cost_R`. `settlement.settle_control` and `billing.Billing`
hold the payoff and cost rules; `execution_log.ExecutionRecorder` is the passive per-action ledger.

**Switch** (`flipthem/switching_policy.py`). `SwitchingTeamPolicy(start, other, Selector(...))`
composes two same-team policies that must share one `RequestBudget`; `switch_log` records the
effective tick, reason, budget and inherited cost cap.

**Fork** (`testbed/fork.py`, `testbed/fork_state.py`). `fork_condition(policy)` is the deadline
selector's own activation predicate evaluated before the policy call; `snapshot_match(...)`
checkpoints the live graph with client handles and verifies live-vs-restored content;
`restore_branch(blob, handles, reference_fp)` returns an independent copy verified in both
fingerprint modes; `branch_policy`/`continue_branch` run branch `A` (continue B0) or `B`
(activate B1); `segment_payoff` evaluates the post-fork segment under the original payoff.

## Layout

```
CybORG/        upstream CAGE Challenge 4 simulator (4.0), unmodified, with its LICENSE.txt
flipthem/      mechanism (resources, settlement, billing, wrapper, execution ledger, trajectory),
               scripted and native anchors, the model-driven policy layer (flipthem/llm/),
               the B0->B1 switching policy and the match runner (flipthem/llm/eval_runner.py)
testbed/       configuration adapter, offline clients, fork helpers, checkpoint engine, configs/
smoke.py       the one-command acceptance
receipts/      JSON receipts of accepted runs
```

## Determinism and scope notes

- Same seed, same configuration and the deterministic stub give the same match; the
  cross-process fork check additionally needs `PYTHONHASHSEED=0` (set automatically).
- The scripted-rule specification texts (`flipthem/_scripted_rule_texts.py`) are a restatement
  written for this edition; their hashes differ from the frozen research values, the rule logic is
  unchanged.
- The smoke is a functional acceptance of the testbed, not a statistical experiment: the payoffs
  it prints come from stub clients and have no empirical meaning.
- Supported surface: `smoke.py`, `flipthem/` and `testbed/`. The upstream `CybORG/Agents/Wrappers`,
  `CybORG/Evaluation` and `CybORG/Tests` additionally need packages that are not pinned here
  (`ray[rllib]`, `torch`, `sb3-contrib`, `matplotlib`, `pettingzoo`, `pytest`, `pytest-mock`); they
  are vendored for completeness, not supported. `run_match(..., conversation_out_dir=...)`
  (conversation journaling) is not included in this edition.

## Acceptance

`receipts/` holds the receipts of the accepted runs. The clean-directory acceptance copied this
tree to an empty directory, created a fresh CPython 3.10 virtual environment, installed
`requirements.txt`, confirmed the original development checkout was not on `sys.path`, and ran
`python smoke.py` with the in-process network block; `receipts/pip-freeze-*.txt` is the exact
package set of that environment.

## License

The testbed code (`flipthem/`, `testbed/`, `smoke.py`, documentation) is released under the MIT
License (see `LICENSE`). The vendored simulator under `CybORG/` is third-party software under its
own MIT license (Commonwealth of Australia / DST Group); see `THIRD_PARTY_NOTICES.md`.

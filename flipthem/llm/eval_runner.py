"""eval_runner: metric + injection protocol.

Unlike `flipthem.matrix._run_episode` (which returns only a payoff -- no telemetry, and only
knows how to build SCRIPTED anchors by name), `run_match` here owns its OWN policy construction
and its OWN per-tick driving loop (the design) so it can pull `policy.telemetry` off any
LLM side after the episode. `blue`/`red` are each either a plain scripted-anchor name (string,
looked up via `flipthem.anchors` exactly as `matrix.py` does) or an `LLMSpec` (constructs an
`LLMPolicy` via `flipthem.llm.agent.make_llm_policy`, offline stub client injected by the caller --
the first revision minimal, testable stand-in for the design's `llm:<model>:<framing>` registration string,
which `MatchRecord.blue`/`.red` still render as).

`BudgetExceeded` (raised by a side's own `RequestBudget`, the design) is caught here, never
propagated to the caller: the match is marked incomplete/excluded with `U_B=U_R=None` -- never a
faked/partial payoff (acceptance #5). Provider failures follow the explicitly recorded
`provider_failure_policy`: the default `exclude` preserves the formal design, while `noop` keeps
the already-existing execution semantics (that side's slots remain Sleep for that tick) and
retains the completed payoff for deployment-level evaluation.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict

from flipthem.wrapper import FlipThemCC4
from flipthem.trajectory import summarise_control
from flipthem.anchors import (
    blue_policy, red_policy, _tools_enabled_for, _context_augmented_for, _campaign_mode_for,
)
from flipthem.llm.agent import make_llm_policy, LLMPolicy, Telemetry
from flipthem.llm.backend import RequestBudget, BudgetExceeded, ReproRecord
from flipthem.scripted_policies import ScriptedTeamPolicy, scripted_policy, get_rule
from flipthem.switching_policy import SwitchingTeamPolicy


PROVIDER_FAILURE_POLICIES = ("exclude", "noop")


@dataclass(frozen=True)
class LLMSpec:
    """the first revision stand-in for a `llm:<model>:<framing>` registration (design the design): enough to
    construct one `LLMPolicy` for one side of one match. `client` is always injected -- offline
    constraint, every test uses a stub; nothing here ever builds a real network client.

    `max_retries` : forwarded to `make_llm_policy` -> `ServiceAdapter`, so the CLI's own
    `--max-retries` flag (the design) actually reaches the adapter underneath an
    LLMSpec-driven match, not just the two-arg `anchors.blue_policy(name, client=...)` registry
    path. Default 0 (exactly one physical attempt per round-trip) matches every pre-existing
    LLMSpec construction, so this is purely additive.

    `strategy` : one of `flipthem.anchors.LLM_STRATEGIES`
    (default `"tool-assisted"`, identical to every pre-existing LLMSpec construction) -- the SAME
    label `anchors.blue_policy`/`red_policy` already accept, so the CLI's `match` sub-command
    (which drives a match through THIS LLMSpec-based path, not the registry) can also select
    `"no-tool"`. Validated the same way anchors.py validates it (raises on an unknown value).

    `price_input`/`price_output` (the design): USD
    per 1,000,000 prompt/completion tokens, forwarded to `make_llm_policy` -> `LLMPolicy` for its
    own per-round cost computation. Default `None` ("not priced", never a
    fabricated 0.0 that would misread as "$0 spent") matches every pre-existing LLMSpec
    construction (none of which ever set these)."""
    client: object
    model: str
    framing: str = "neutral"
    max_tool_calls: int = 4
    max_idle_ticks: int = 10
    budget: RequestBudget | None = None
    value: float = 1.0
    max_retries: int = 0
    max_final_repairs: int = 0
    strategy: str = "tool-assisted"
    price_input: float | None = None
    price_output: float | None = None
    max_cost_usd: float | None = None   # cumulative cost stop-loss for the whole match


@dataclass(frozen=True)
class ScriptedSpec:
    """A constrained scripted side (the scripted-baseline design): the mirror of
    `LLMSpec` for a `flipthem.scripted_policies` rule. `name` is a schedule code (S0/S1/S2/A0) or a
    frozen policy_id. A scripted side has NO client, budget, tokens, price or model telemetry -- it
    reaches `run_match` exactly like an `LLMSpec` does, but builds a `ScriptedTeamPolicy` and
    contributes no cost. Kept as its own type so the runner can branch on it explicitly rather than
    overloading the plain-string anchor path (which builds a DIFFERENT, wider-information native
    policy)."""
    name: str


@dataclass(frozen=True)
class SwitchSpec:
    """A within-episode whole-team configuration switch (configuration-switching design): two
    `LLMSpec` components of the SAME team plus a frozen `Selector` (from `flipthem.switching_policy`).

    The two components MUST share one `RequestBudget` (the caller sets `start.budget is other.budget`),
    so switching claims no second budget. `run_match` builds both as ordinary `LLMPolicy` objects and
    wraps them in a `SwitchingTeamPolicy`; only the active component ever calls a model/tool."""
    start: LLMSpec
    other: LLMSpec
    selector: object   # flipthem.switching_policy.Selector


@dataclass
class MatchRecord:
    blue: str
    red: str
    seed: int
    steps: int
    U_B: float | None
    U_R: float | None
    blue_tel: Telemetry | None
    red_tel: Telemetry | None
    excluded: bool
    # : each LLM side's full ReproRecord
    # (None for a scripted side, exactly like blue_tel/red_tel), plus match-total tokens/$ cost
    # summed across whichever side(s) are LLM. Cost is split into an exact total and a known
    # subtotal: a request that returned no usage may have unknown billing and must not be silently
    # omitted from something labelled "total".
    blue_repro: ReproRecord | None = None
    red_repro: ReproRecord | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    known_cost_subtotal_usd: float | None = None
    cost_complete: bool | None = None
    # The scripted-baseline design: the GAME action-cost split (from `payoff()`), which
    # MatchRecord previously discarded. These are FT action costs (Restore=1, InvalidAction=0.1,
    # ...), NOT API dollars -- the two are reported separately, never conflated. None for an
    # excluded match (no payoff was read).
    cost_B: float | None = None
    cost_R: float | None = None
    # Optional passive execution-ledger sidecar reference {path, sha256, schema_version,
    # summary_path}; None when the recorder was not attached. The ledger itself lives beside the
    # match, never inside the payoff.
    execution_log: dict | None = None
    # Within-episode switch record (config-switching design): per switching side, the frozen
    # selector + active component, event-first-available / switch-effective ticks, reason, and the
    # final shared-budget total. None when neither side switches.
    switch_log: dict | None = None
    # the per-tick per-resource control vectors exactly as `step()` returned
    # them, plus `flipthem.trajectory.summarise_control`'s derived per-resource occupancy /
    # time-to-first-scoring-root (right-censored) / loss / re-entry figures. GROUND TRUTH the
    # runner reads off the mechanism itself -- no policy supplies it and no prompt ever sees it.
    # An excluded or truncated match keeps whatever trajectory it did produce (real data), and the
    # summary is always consistent with the trace that sits beside it.
    control_trace: list | None = None
    control_summary: dict | None = None
    # Item 4: the EPISODE IDENTITY, decomposed. `blue`/`red` render as `llm:<model>:<framing>`,
    # which omits the assistance configuration entirely -- so two arms differing only in
    # tools/context produced byte-identical labels and could not be told apart in a saved run,
    # making keyed pairing impossible from the persisted record alone. These fields are the same
    # facts the runner already had, written down: deployment, assistance and framing PER SIDE
    # (all three None for a scripted side, which has no LLM configuration), plus the repeat and
    # attempt ids. `flipthem.analysis.EPISODE_KEY_FIELDS` reads exactly these; `attempt_id` is
    # recorded but deliberately outside the key (a rerun is not another repeat).
    blue_model: str | None = None
    blue_strategy: str | None = None
    blue_framing: str | None = None
    red_model: str | None = None
    red_strategy: str | None = None
    red_framing: str | None = None
    repeat_index: int | None = None
    attempt_id: int | None = None
    # The FIXED configuration each LLM side ran under (`LLMPolicy.config_fingerprint`), and the
    # match-level conditions. The episode key pins WHICH model and WHICH configuration label was
    # requested; these pin the deployment actually behind that label -- same model id served by a
    # different upstream provider (so a different quantization), a changed output budget or
    # sampling setting, a different horizon or mode. Without them two arms that differ in
    # deployment pair happily and the deployment change is read as a tool effect.
    # `flipthem.analysis` compares these within every block and refuses to pair on a mismatch.
    blue_config_fingerprint: dict | None = None
    red_config_fingerprint: dict | None = None
    mode: str | None = None
    # WHY this match was excluded, when it was: "budget_exceeded", "provider_failure", or the
    # exception type for an unexpected failure. `excluded` alone cannot support the frozen rerun
    # rule (v3 the design), which allows a rerun ONLY after a transport failure -- a budget stop
    # is not one. None on a completed match.
    exclusion_reason: str | None = None
    # Transport handling is a frozen match condition. Under ``noop`` a failed provider round
    # advances with that side's default Sleep actions and the payoff remains usable; telemetry
    # still exposes every failure and ``transport_degraded`` prevents it looking like a clean run.
    provider_failure_policy: str = "exclude"
    transport_degraded: bool = False


def _label(side) -> str:
    if isinstance(side, SwitchSpec):
        return f"switch:{side.selector.arm}:{side.start.model}"
    if isinstance(side, LLMSpec):
        return f"llm:{side.model}:{side.framing}"
    if isinstance(side, ScriptedSpec):
        return f"script:{get_rule(side.name).policy_id}"
    return str(side)


def _full_label(side) -> str:
    """The COMPLETE identity of one side: deployment, framing AND assistance.

    `_label` renders the design's `llm:<model>:<framing>` registration string, which omits the
    assistance configuration entirely. That is fine for `MatchRecord.blue`/`.red` (the record
    carries `blue_strategy`/`red_strategy` beside them), but it is NOT enough to identify the
    OPPONENT in a conversation filename: holding BLUE fixed and moving RED from
    `context-assisted` to `context-tool-assisted` left BLUE's two files with byte-identical
    names, so the second match died on `write_conversation`'s overwrite guard. Both sides' files
    now key off this full identity, so every distinct match has a distinct path."""
    if isinstance(side, SwitchSpec):
        return f"switch:{side.selector.arm}:{side.start.model}:{side.start.framing}"
    if isinstance(side, LLMSpec):
        return f"llm:{side.model}:{side.framing}:{side.strategy}"
    if isinstance(side, ScriptedSpec):
        return f"script:{get_rule(side.name).policy_id}"
    return str(side)


def _build_blue(side):
    if isinstance(side, SwitchSpec):
        # Both components are built as ordinary blue LLMPolicies sharing one RequestBudget; the
        # SwitchingTeamPolicy ctor enforces the shared-budget and same-team invariants.
        return SwitchingTeamPolicy(_build_blue(side.start), _build_blue(side.other), side.selector)
    if isinstance(side, ScriptedSpec):
        return scripted_policy(side.name)
    if isinstance(side, LLMSpec):
        campaign_mode = _campaign_mode_for(side.strategy, "blue")
        return make_llm_policy(team="blue", client=side.client, model=side.model,
                               framing=side.framing, max_tool_calls=side.max_tool_calls,
                               max_idle_ticks=side.max_idle_ticks, budget=side.budget,
                               value=side.value, max_retries=side.max_retries,
                               max_final_repairs=side.max_final_repairs,
                               tools_enabled=_tools_enabled_for(side.strategy),
                               context_augmented=_context_augmented_for(side.strategy),
                               campaign_mode=campaign_mode,
                               price_input=side.price_input, price_output=side.price_output,
                               max_cost_usd=side.max_cost_usd)
    return blue_policy(side)


def _build_red(side):
    """Returns (policy_or_None, red_agent_class_or_None) -- matches `flipthem.anchors.red_policy`'s
    own convention: every native scripted red is self-driving (policy is None, class is set); an
    LLM red is runner-driven (policy is the LLMPolicy, class is None so CC4 defaults red slots to
    SleepAgent internally and our own per-slot dict overrides them every tick, same as `.step()`
    already does for any explicit `red_actions`). A ScriptedSpec red is runner-driven the same
    way (policy is a ScriptedTeamPolicy, class None). A SwitchSpec red wraps two red LLMPolicies."""
    if isinstance(side, SwitchSpec):
        c0 = make_llm_policy(team="red", client=side.start.client, model=side.start.model,
                             framing=side.start.framing, max_tool_calls=side.start.max_tool_calls,
                             max_idle_ticks=side.start.max_idle_ticks, budget=side.start.budget,
                             value=side.start.value, max_retries=side.start.max_retries,
                             max_final_repairs=side.start.max_final_repairs,
                             tools_enabled=_tools_enabled_for(side.start.strategy),
                             context_augmented=_context_augmented_for(side.start.strategy),
                             campaign_mode=_campaign_mode_for(side.start.strategy, "red"),
                             price_input=side.start.price_input, price_output=side.start.price_output,
                             max_cost_usd=side.start.max_cost_usd)
        c1 = make_llm_policy(team="red", client=side.other.client, model=side.other.model,
                             framing=side.other.framing, max_tool_calls=side.other.max_tool_calls,
                             max_idle_ticks=side.other.max_idle_ticks, budget=side.other.budget,
                             value=side.other.value, max_retries=side.other.max_retries,
                             max_final_repairs=side.other.max_final_repairs,
                             tools_enabled=_tools_enabled_for(side.other.strategy),
                             context_augmented=_context_augmented_for(side.other.strategy),
                             campaign_mode=_campaign_mode_for(side.other.strategy, "red"),
                             price_input=side.other.price_input, price_output=side.other.price_output,
                             max_cost_usd=side.other.max_cost_usd)
        return SwitchingTeamPolicy(c0, c1, side.selector), None
    if isinstance(side, ScriptedSpec):
        return scripted_policy(side.name), None
    if isinstance(side, LLMSpec):
        policy = make_llm_policy(team="red", client=side.client, model=side.model,
                                 framing=side.framing, max_tool_calls=side.max_tool_calls,
                                 max_idle_ticks=side.max_idle_ticks, budget=side.budget,
                                 value=side.value, max_retries=side.max_retries,
                                 max_final_repairs=side.max_final_repairs,
                                 tools_enabled=_tools_enabled_for(side.strategy),
                                 context_augmented=_context_augmented_for(side.strategy),
                                 campaign_mode=_campaign_mode_for(side.strategy, "red"),
                                 price_input=side.price_input, price_output=side.price_output,
                                 max_cost_usd=side.max_cost_usd)
        return policy, None
    return red_policy(side)


def _identity(side) -> tuple:
    """(deployment, assistance, framing) for one side. A scripted anchor has no LLM configuration,
    so its assistance/framing are None -- never a stand-in default that would make it look like a
    configured arm to the pairing code."""
    if isinstance(side, SwitchSpec):
        # deployment = start component's model; "strategy" carries the arm (pi_E/pi_T) for pairing.
        return side.start.model, side.selector.arm, side.start.framing
    if isinstance(side, LLMSpec):
        return side.model, side.strategy, side.framing
    if isinstance(side, ScriptedSpec):
        return _label(side), None, None
    return str(side), None, None


def _fingerprint(policy):
    """The fixed configuration of an LLM side (its `config_fingerprint`), the frozen rule identity
    of a scripted side (its `fingerprint`), the switching identity of a switch side, or None for a
    plain native anchor (which has no deployment/rule to change out from under a comparison)."""
    if isinstance(policy, LLMPolicy):
        return policy.config_fingerprint()
    if isinstance(policy, (ScriptedTeamPolicy, SwitchingTeamPolicy)):
        return policy.fingerprint()
    return None


def _side_telemetry(policy):
    """Telemetry for an LLM side or a switch side (merged across its components), else None."""
    if isinstance(policy, (LLMPolicy, SwitchingTeamPolicy)):
        return policy.telemetry
    return None


def _side_repros(policy):
    """The ReproRecord(s) a side contributes: one for an LLM side, one per component for a switch
    side, none for a scripted/native side."""
    if isinstance(policy, LLMPolicy):
        return [policy.repro_record()]
    if isinstance(policy, SwitchingTeamPolicy):
        return [c.repro_record() for c in policy.components]
    return []


def run_match(blue, red, seed: int, steps: int, mode: str = "ft",
             conversation_out_dir: str | None = None,
             repeat_index: int | None = None, attempt_id: int | None = None,
             provider_failure_policy: str = "exclude",
             execution_log_path: str | None = None,
             episode_id: str | None = None) -> MatchRecord:
    """`conversation_out_dir` (addition, the design):
    when given, writes ONE separate conversation JSON (`flipthem.llm.conversation`) per LLM side
    under that directory, using the SAME policy objects this function already built and drove --
    no second drive, no duplicated loop. `None` (the default) writes nothing, so every
    pre-existing call site is unaffected.

    `execution_log_path` (execution-ledger design): when given, a PASSIVE
    `flipthem.execution_log.ExecutionRecorder` is attached as `game.step_observer` and its per-tick
    ledger + Green business summary are written to that JSONL path (a `<path>.summary.json` sits
    beside it). The recorder cannot influence the episode (verified in tests); `None` (the default)
    attaches nothing. `episode_id` labels the ledger rows; it defaults to the two sides' labels."""
    if provider_failure_policy not in PROVIDER_FAILURE_POLICIES:
        raise ValueError(
            f"unknown provider_failure_policy {provider_failure_policy!r}; "
            f"expected one of {PROVIDER_FAILURE_POLICIES}")

    blue_model, blue_strategy, blue_framing = _identity(blue)
    red_model, red_strategy, red_framing = _identity(red)

    pol_b = _build_blue(blue)
    pol_r, red_cls = _build_red(red)

    conversation_sides = ((pol_b, blue, red), (pol_r, red, blue))
    journals = []
    if conversation_out_dir is not None:
        from flipthem.llm.conversation import build_conversation_record, ConversationJournal
        # Create the recovery files before the first paid request. An identity collision or an
        # unwritable directory therefore fails without spending anything.
        for policy, side, opponent in conversation_sides:
            if isinstance(policy, LLMPolicy):
                rec = build_conversation_record(
                    policy, _label(side), getattr(side, "strategy", "tool-assisted"), seed,
                    opponent=_full_label(opponent), repeat_index=repeat_index,
                    attempt_id=attempt_id,
                    provider_failure_policy=provider_failure_policy)
                journal = ConversationJournal(rec, conversation_out_dir)
                # Each physical response/error becomes durable before an in-decision retry or
                # another tool turn. ConversationJournal.capture later sees the advanced request
                # cursor and therefore cannot duplicate it.
                policy.adapter.request_observer = journal.capture_request
                journals.append((policy, side, opponent, journal))

    def _conversation_record(policy, side, opponent):
        from flipthem.llm.conversation import build_conversation_record
        return build_conversation_record(
            policy, _label(side), getattr(side, "strategy", "tool-assisted"), seed,
            opponent=_full_label(opponent), repeat_index=repeat_index, attempt_id=attempt_id,
            provider_failure_policy=provider_failure_policy)

    def _capture_journals():
        for policy, side, opponent, journal in journals:
            journal.capture(_conversation_record(policy, side, opponent))

    g = FlipThemCC4(seed=seed, steps=steps, mode=mode, red_agent_class=red_cls)
    g.reset()

    recorder = None
    if execution_log_path is not None:
        from flipthem.execution_log import ExecutionRecorder
        recorder = ExecutionRecorder(episode_id or f"{_label(blue)}__{_label(red)}__seed{seed}")
        g.step_observer = recorder.on_step   # passive; never steers the episode

    excluded = False
    exclusion_reason = None
    control_trace: list = []   # captured per tick, so a partial episode keeps its own
    try:
        for _ in range(g.T):
            try:
                b = pol_b(g, g.team_observation("blue"))
            finally:
                # Flush BLUE before beginning RED's request. A hard process/machine interruption
                # while RED is in flight must not lose BLUE's already-paid completed response.
                _capture_journals()
            try:
                r = pol_r(g, g.team_observation("red")) if pol_r else None
            finally:
                _capture_journals()
            info = g.step(b, r)
            control_trace.append({k: int(v) for k, v in info["x"].items()})
    except BudgetExceeded:
        excluded = True   # incomplete episode: never read a payoff off a truncated run
        # NOT a transport failure: a budget/cost stop is our own quota deciding to halt, so the
        # frozen rerun rule (v3 the design) must not treat it as grounds for another attempt.
        exclusion_reason = "budget_exceeded"
    except Exception as exc:
        # A diagnostic sidecar, never a partial game payoff. Preserve the original failure.
        if conversation_out_dir is not None:
            from flipthem.llm.conversation import write_conversation
            for policy, side, opponent in ((pol_b, blue, red), (pol_r, red, blue)):
                if isinstance(policy, LLMPolicy):
                    # Same episode key as the success path: a diagnostic sidecar from a failed
                    # match must not collide with (or be blocked by) another repeat's file.
                    rec = _conversation_record(policy, side, opponent)
                    rec.update(incomplete=True, failure={"code": type(exc).__name__},
                               U_B=None, U_R=None)
                    # Ledger includes the last response even when decision parsing/execution failed.
                    requests = policy.adapter.request_log
                    costs = [r["cost_usd"] for r in requests if r.get("cost_usd") is not None]
                    rec["known_cost_usd"] = sum(costs) if costs else None
                    try:
                        write_conversation(rec, out_dir=conversation_out_dir)
                        journal = next(j for p, _s, _o, j in journals if p is policy)
                        journal.finish(rec, "interrupted")
                    except Exception as save_error:
                        # Do not replace the underlying error or print potentially sensitive text.
                        import warnings
                        warnings.warn(f"partial record write failed: {type(save_error).__name__}")
        raise

    # GAP fix (2nd re-review): the LAST submitted action's resolution lands the tick AFTER
    # submission (the observe-before-act loop), so the per-tick loop above alone can never see
    # it for whatever was still in flight when the loop ended. flush() does one extra
    # team_observation() read (no further .step()) per LLM side so it is captured here rather
    # than silently dropped -- flush() makes no model/budget calls, so it is always safe to run
    # even after a BudgetExceeded above. A scripted side has its own flush() (same reason -- resolve
    # its last submitted action's outcome into own_log); a plain native anchor/None has none.
    for policy in (pol_b, pol_r):
        if isinstance(policy, (LLMPolicy, ScriptedTeamPolicy, SwitchingTeamPolicy)):
            policy.flush(g)

    blue_tel = _side_telemetry(pol_b)
    red_tel = _side_telemetry(pol_r)
    transport_degraded = ((blue_tel is not None and blue_tel.flagged)
                          or (red_tel is not None and red_tel.flagged))
    if transport_degraded and provider_failure_policy == "exclude":
        excluded = True   # >=1 PROVIDER_FAILURE anywhere in the episode (design the design/7)
        # Only set when nothing stricter already claimed the exclusion: if a budget stop also
        # happened, that reason stands. Conservative on purpose -- an ambiguous cause must never
        # end up authorising a rerun.
        if exclusion_reason is None:
            exclusion_reason = "provider_failure"

    # : each LLM side's full ReproRecord, plus a match-total tokens/$ cost
    # summed across whichever side(s) returned usage. A switching side has two component repros
    # (summed into the totals); its single MatchRecord repro field stays None (detail in switch_log).
    blue_repro = pol_b.repro_record() if isinstance(pol_b, LLMPolicy) else None
    red_repro = pol_r.repro_record() if isinstance(pol_r, LLMPolicy) else None
    total_tokens, total_cost_usd, known_cost_subtotal_usd, cost_complete =\
        _sum_repro_list(_side_repros(pol_b) + _side_repros(pol_r))

    if conversation_out_dir is not None:
        # Excluded matches ARE still written -- the conversation up to the failure is real
        # reproducibility data, never withheld just because the match itself was incomplete.
        from flipthem.llm.conversation import write_conversation
        for policy, side, opponent in ((pol_b, blue, red), (pol_r, red, blue)):
            if isinstance(policy, LLMPolicy):
                rec = _conversation_record(policy, side, opponent)
                write_conversation(rec, out_dir=conversation_out_dir)
                journal = next(j for p, _s, _o, j in journals if p is policy)
                journal.finish(rec, "excluded" if excluded else "complete")

    if excluded:
        u_b, u_r = None, None   # never a faked/partial payoff for an excluded match
        cost_b = cost_r = None
    else:
        payoff = g.payoff()
        u_b_key, u_r_key = ("U_B_c4", "U_R_c4") if mode == "c4" else ("U_B", "U_R")
        u_b, u_r = payoff[u_b_key], payoff[u_r_key]
        # The GAME action-cost split, kept beside U (never conflated with API dollars).
        cost_b, cost_r = payoff["cost_B"], payoff["cost_R"]

    # A switching side has no single repro field, so each component's ReproRecord (actual returned
    # model/provider, requests, provider failures, API cost) rides along in its switch log.
    switch_log = {}
    if isinstance(pol_b, SwitchingTeamPolicy):
        switch_log["blue"] = {**pol_b.switch_log,
                              "component_repros": [asdict(r) for r in _side_repros(pol_b)]}
    if isinstance(pol_r, SwitchingTeamPolicy):
        switch_log["red"] = {**pol_r.switch_log,
                             "component_repros": [asdict(r) for r in _side_repros(pol_r)]}
    switch_log = switch_log or None

    execution_log = None
    if recorder is not None:
        own_logs = {}
        if isinstance(pol_b, (LLMPolicy, ScriptedTeamPolicy, SwitchingTeamPolicy)):
            own_logs["blue"] = pol_b.own_log
        if isinstance(pol_r, (LLMPolicy, ScriptedTeamPolicy, SwitchingTeamPolicy)):
            own_logs["red"] = pol_r.own_log
        recorder.finalize(own_logs=own_logs, horizon=g.T)
        sha = recorder.write_jsonl(execution_log_path)
        summary_path = str(execution_log_path) + ".summary.json"
        recorder.write_summary(summary_path)
        from flipthem.execution_log import EXECUTION_LOG_SCHEMA_VERSION
        execution_log = {"path": str(execution_log_path), "sha256": sha,
                         "schema_version": EXECUTION_LOG_SCHEMA_VERSION,
                         "summary_path": summary_path}

    return MatchRecord(blue=_label(blue), red=_label(red), seed=seed, steps=steps,
                       U_B=u_b, U_R=u_r, blue_tel=blue_tel, red_tel=red_tel, excluded=excluded,
                       blue_repro=blue_repro, red_repro=red_repro, total_tokens=total_tokens,
                       total_cost_usd=total_cost_usd,
                       known_cost_subtotal_usd=known_cost_subtotal_usd,
                       cost_complete=cost_complete, control_trace=control_trace,
                       control_summary=summarise_control(control_trace, g.values),
                       blue_model=blue_model, blue_strategy=blue_strategy,
                       blue_framing=blue_framing, red_model=red_model,
                       red_strategy=red_strategy, red_framing=red_framing,
                       repeat_index=repeat_index, attempt_id=attempt_id,
                       blue_config_fingerprint=_fingerprint(pol_b),
                       red_config_fingerprint=_fingerprint(pol_r),
                       mode=mode, exclusion_reason=exclusion_reason,
                       provider_failure_policy=provider_failure_policy,
                       transport_degraded=transport_degraded,
                       cost_B=cost_b, cost_R=cost_r, execution_log=execution_log,
                       switch_log=switch_log)


def _sum_repro_totals(blue_repro: ReproRecord | None, red_repro: ReproRecord | None) -> tuple:
    """Two-side convenience wrapper over `_sum_repro_list` (kept for existing call sites/tests)."""
    return _sum_repro_list([blue_repro, red_repro])


def _sum_repro_list(repros_in) -> tuple:
    """Return tokens, exact cost, known cost subtotal, and cost completeness across any number of
    ReproRecords (an LLM side contributes one; a switching side contributes one per component).

    A physical request with no returned usage can have unknown billing. In that case the known
    responses remain available as a subtotal, while the exact total is ``None``."""
    repros = [repro for repro in repros_in if repro is not None]
    any_usage = False
    any_priced = False
    tokens = 0
    known_cost = 0.0
    for repro in repros:
        if repro is not None and repro.total_tokens is not None:
            any_usage = True
            tokens += repro.total_tokens
        subtotal = (repro.known_cost_subtotal_usd
                    if repro.known_cost_subtotal_usd is not None else repro.total_cost_usd)
        if subtotal is not None:
            any_priced = True
            known_cost += subtotal
    complete = all(repro.cost_complete for repro in repros) if repros else None
    exact_cost = known_cost if complete and any_priced else None
    return ((tokens if any_usage else None), exact_cost,
            (known_cost if any_priced else None), complete)


def persist(records, path: str) -> None:
    """Writes JSON with `allow_nan=False` (design the design): an excluded record's U_B/U_R are
    `None` (-> JSON `null`), never NaN, so this never raises on exactly the records it exists to
    make legible.

    ATOMIC write -- serialize to a temp file in the SAME directory, then `os.replace`
    (atomic on POSIX and Windows). A serialization error (e.g. a stray NaN under allow_nan=False)
    therefore leaves any pre-existing file at `path` completely intact and drops no partial file,
    so a paid run's result is never silently corrupted by a late write failure."""
    payload = [asdict(r) for r in records]
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".persist-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, allow_nan=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# Two-sided normal-distribution critical values for the supported confidence levels. Minor fix
# (review round 2): an earlier version silently mapped every confidence to z=1.96 (95%) --
# unsupported levels must raise, never quietly return the wrong interval.
_Z_BY_CONFIDENCE = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def paired_ci(recs_a, recs_b, field: str = "U_B", confidence: float = 0.95):
    """A finite (lower, upper) interval on the paired difference of `field` between two same-
    length record sets (paired by position, e.g. matched seeds) -- a normal-approximation CI.
    the first revision minimal depth: the design's frozen-pool/solver-mixture CI machinery  is a
    later refinement; this only needs to return a finite interval given >=2 samples (design
    the design acceptance).

    Paired_ci fix (fix round 2): with 0 or 1 paired sample there is no sample variance to
    estimate a margin from -- returning (x, x) would silently claim TOTAL confidence from no
    evidence (a single point, or none at all) at all, which is worse than reporting nothing.
    Both cases return the non-finite, explicitly "undefined" interval (-inf, +inf) instead of a
    fabricated zero-width one; a caller that needs a hard failure instead can treat that as its
    cue to raise."""
    if confidence not in _Z_BY_CONFIDENCE:
        raise ValueError(f"unsupported confidence level {confidence!r}; expected one of "
                         f"{sorted(_Z_BY_CONFIDENCE)}")
    diffs = [getattr(a, field) - getattr(b, field) for a, b in zip(recs_a, recs_b)
             if getattr(a, field) is not None and getattr(b, field) is not None]
    n = len(diffs)
    if n < 2:
        return (float("-inf"), float("inf"))
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    z = _Z_BY_CONFIDENCE[confidence]
    margin = z * se
    return (mean - margin, mean + margin)

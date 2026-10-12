"""The LLM-wake gate, the per-tick loop, per-tick history update, Telemetry, LLMPolicy
(the design + the design integration points 1 and 3).

`LLMPolicy.__call__(self, game, team_obs) -> {slot: Action}` is a drop-in team policy with EXACTLY
the mechanism layer's existing calling convention (design the design; matches how `flipthem.anchors`
scripted policies are called: `pol(game, game.team_observation(team))`). It is the ONE place
allowed to touch `game` directly (same as a scripted anchor reading `game.blue_slots`/
`game.team_observation(...)`) -- its whole job is to distil `game` down into a `LegalContext`
before handing off to prompts/tools/backend, none of which ever see `game`/`env`/`state`
(structural isolation, design the design).

Per environment tick (the design): own_log is updated from this side's own legal
observation EVERY call, whether or not the model is invoked that tick -- a slot with a tracked
in-flight action gets an "in_progress" entry every tick it stays busy, "completed"/"session_lost"
the tick its status resolves. A fresh decision is logged "submitted" at the tick it is issued.
Log-entry count is therefore not the tick count (the design): a busy slot logs every
tick even on ticks the model was never called; an idle, never-addressed slot logs nothing.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
from dataclasses import dataclass, field
from typing import Mapping

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.Action import InvalidAction

from flipthem.resources import OPERATIONAL_RESOURCES
from flipthem.llm import compiler
from flipthem.llm.context import (
    LogEntry, build_legal_context, extract_alerts, RESULT_SUBMITTED, RESULT_IN_PROGRESS,
    RESULT_SUCCEEDED, RESULT_FAILED, RESULT_INVALID, RESULT_SESSION_LOST, RESULT_OBSERVED,
)
from flipthem.llm.prompts import build_system, build_user
from flipthem.llm.backend import Backend, RequestBudget, BudgetExceeded, ServiceAdapter,\
    DECISION, MODEL_ERROR, PROVIDER_FAILURE, REFUSAL_CODE
from flipthem.llm.tools import history, action_manual, known_targets, check_plan, TOOL_SCHEMAS
from flipthem.llm.campaign import (
    CAMPAIGN_MODES, CAMPAIGN_VERSION, CAMPAIGN_LIMITS, CAMPAIGN_PROMPT_RECENT,
    CAMPAIGN_PROMPT_MAX_CHARS, CampaignState,
    campaign_tool_schemas, campaign_instructions,
)
from flipthem.llm.cost import cost_for_usage, conversation_totals

DEFAULT_MAX_TOOL_CALLS = 4
DEFAULT_MAX_IDLE_TICKS = 10
# Increment when model-visible history/state rendering or tool-result semantics change.  The
# value rides in the assistance fingerprint so old and new prompt contracts cannot silently pair.
ASSISTANCE_CONTRACT_VERSION = "8"


def _is_ip_text(value) -> bool:
    try:
        ipaddress.ip_address(str(value))
        return True
    except (TypeError, ValueError):
        return False


def should_call(now_tick: int, last_call_tick: int, wake_spec: dict | None, new_event: bool,
                 has_decidable: bool, max_idle_ticks: int) -> bool:
    """Design the design: call iff `new_event`, or the model's own wake condition is met, or the
    idle cap is hit -- whichever first; `has_decidable=False` always means "never call".

    I3 (review round 2): `parse_final` already sanitizes `wake` before it reaches the real loop,
    but this guards defensively too -- a malformed "after" (non-numeric, negative, ...) falls back
    to the cap rather than crashing `int(...)` and killing the match."""
    if not has_decidable:
        return False
    if new_event:
        return True
    idle = now_tick - last_call_tick
    if wake_spec and "after" in wake_spec:
        try:
            after = int(wake_spec["after"])
        except (TypeError, ValueError):
            after = max_idle_ticks
        if after < 0:
            after = max_idle_ticks
        n = min(after, max_idle_ticks)
        return idle >= n
    return idle >= max_idle_ticks   # no wake, or wake={"on":"event"}: event-or-cap


def _tool_names(schemas):
    """The tool menu a client was built with, as a sorted name list -- `None` when it carries no
    menu at all (every no-tool arm, and every stub). Defensive about the schema shape: an entry
    that is not the OpenAI `{"function": {"name": ...}}` form is rendered rather than dropped, so
    a changed menu can never go unnoticed just because one entry looked unfamiliar."""
    if not schemas:
        return None
    names = []
    for schema in schemas:
        fn = schema.get("function") if isinstance(schema, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        names.append(name if isinstance(name, str) else repr(schema))
    return sorted(names)


@dataclass
class Telemetry:
    decisions: int = 0
    tool_calls: int = 0
    model_errors: int = 0
    provider_failures: int = 0
    sleeps: int = 0
    flagged: bool = False
    valid_action_submissions: int = 0
    # --- design the five failure classes, counted APART from one another ----------
    # `model_errors` alone could not tell a whole-round parse failure from a single illegal
    # action, an omitted unit left no trace at all, and there was no denominator for any rate.
    #
    # `decidable_slot_rounds` is that denominator: one per (tick, slot) the model was actually
    # asked to decide. A busy slot or one with no legal action is NEVER counted (the design table
    # row 4) -- it is not an opportunity the model declined, so it cannot be in the denominator
    # of an omission or error rate. Every counted slot-round lands in exactly one of the four
    # classes below, so `omitted + illegal + valid + model_error_slots == decidable_slot_rounds`.
    decidable_slot_rounds: int = 0
    omitted_slots: int = 0        # decidable unit the model simply did not address -> stays Sleep
    illegal_actions: int = 0      # JSON parsed, but THIS slot's action/target compiled to Invalid
    model_error_slots: int = 0    # decidable slots billed Invalid by a WHOLE-ROUND model error
    explicit_sleeps: int = 0      # a deliberate Sleep choice (a SUBSET of valid_action_submissions)
    # A content refusal confirmed from a provider-authored signal (the design: never inferred from
    # an HTTP status code). Classified separately, but still scored by the uniform whole-round
    # "no valid decision" rule, so it also shows up in `model_errors`/`model_error_slots`.
    refusals: int = 0
    # Bounded final-format recovery. These count the original model failure even when a second,
    # forced-final request succeeds, so robustness never erases reliability evidence.
    final_repairs_attempted: int = 0
    final_repairs_recovered: int = 0


class LLMPolicy:
    """`policy(game, team_obs) -> {slot: Action}` -- construct via `make_llm_policy`."""

    def __init__(self, team: str, client, model: str, framing: str = "neutral",
                 max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
                 max_idle_ticks: int = DEFAULT_MAX_IDLE_TICKS,
                 budget: RequestBudget | None = None, value: float = 1.0,
                 wake_alert_hosts: tuple | None = None, wake_on_all_alerts: bool = False,
                 max_retries: int = 0, params: dict | None = None, tools_enabled: bool = True,
                 price_input: float | None = None, price_output: float | None = None,
                 max_cost_usd: float | None = None, context_augmented: bool = False,
                 campaign_mode: str = "off", max_final_repairs: int = 0):
        if team not in ("blue", "red"):
            raise ValueError(f"unknown team {team!r}; expected 'blue' or 'red'")
        if campaign_mode not in CAMPAIGN_MODES:
            raise ValueError(f"unknown campaign mode {campaign_mode!r}; expected {CAMPAIGN_MODES}")
        if campaign_mode != "off" and team != "red":
            raise ValueError("campaign modes are RED-only")
        if campaign_mode != "off" and not tools_enabled:
            raise ValueError("campaign modes require tools_enabled=True")
        self.campaign_mode = campaign_mode
        self.campaign = CampaignState(campaign_mode) if campaign_mode != "off" else None
        self._campaign_tool_names = set(_tool_names(campaign_tool_schemas(campaign_mode)) or ())
        self.team = team
        self.action_names = compiler.BLUE_ACTIONS if team == "blue" else compiler.RED_ACTIONS
        self.client = client
        self.model = model
        self.framing = framing
        # the design: "no-tool" (tools_enabled=False) is a STRUCTURAL guarantee, not merely a prompt-
        # wording change -- forcing the EFFECTIVE tool budget to 0 means Backend's own
        # forced_final gate (`tool_calls_made >= max_tool_calls`) is already True on the very
        # FIRST turn, so `telemetry.tool_calls` stays 0 even if the client/model tries a tool
        # call anyway (Backend then reports MODEL_ERROR, never silently executes it -- see
        # backend.py's run_decision). `build_system(..., tools_enabled=...)` (below, in __call__)
        # separately omits the tool section from the PROMPT TEXT itself -- the two together are
        # the "clean single-prompt control" the design asks for.
        self.tools_enabled = bool(tools_enabled)
        self.max_tool_calls = max_tool_calls if self.tools_enabled else 0
        # context-assisted ("deterministic legal-context augmentation with one model request per
        # decision" -- NOT tool-assisted): tools_enabled is False (same one-request, no-tool-loop
        # wire shape as no-tool), but build_user additionally pre-assembles compact history and
        # current known state into the prompt. Static action details already live in the shared
        # system prompt. Independent of tools_enabled so repro_record() (below)
        # can report the correct one of THREE labels, not a binary tools_enabled guess.
        self.context_augmented = bool(context_augmented)
        self.max_idle_ticks = max_idle_ticks
        self.budget = budget if budget is not None else RequestBudget(limit=None)
        self.max_retries = max_retries
        self.max_final_repairs = max_final_repairs
        self.params = dict(params or {})
        # the design: budget/retry accounting lives ENTIRELY in the adapter -- Backend no longer knows
        # what a budget is. max_retries=0 (the default) means exactly one physical attempt per
        # logical round-trip, identical to the first revision behaviour before the adapter existed.
        self.adapter = ServiceAdapter(client, model, self.params, self.max_retries, self.budget,
                                      price_input=price_input, price_output=price_output,
                                      max_cost_usd=max_cost_usd)
        self.backend = Backend(self.adapter, model, self.max_tool_calls,
                               max_final_repairs=self.max_final_repairs)
        self.value = value
        self.target_hosts = tuple(OPERATIONAL_RESOURCES)
        # Item 6 (fix round 2): the wake trigger's alert filter is EXPLICIT and separate from the
        # observation itself (ctx.alerts always carries the full set from extract_alerts). The
        # the first revision DEFAULT narrows the wake to the 2 operational targets (the same choice C2 made,
        # now correctly scoped to "what wakes the model", never "what the model can see") --
        # override wake_alert_hosts for a deliberately labelled, different experimental condition.
        self.wake_alert_hosts = (tuple(wake_alert_hosts) if wake_alert_hosts is not None
                                 else self.target_hosts)
        # the design: a SEPARATE, coarser labeled opt-in -- when True, ANY alert (not just
        # wake_alert_hosts) may set new_event. Never the default; widens only the alert channel,
        # never the action-resolution channel (the design).
        self.wake_on_all_alerts = bool(wake_on_all_alerts)

        self.telemetry = Telemetry()
        self._own_log: list[LogEntry] = []
        self._pending: dict[str, tuple[str, str | None]] = {}   # slot -> (action_name, target)
        self._last_call_tick = -(10 ** 9)   # forces a call the first decidable tick
        self._wake_spec: dict | None = None
        # Item 2 (fix round 2): cumulative own-side state for computing each tick's real
        # observation delta -- red only (blue's "observed" comes from extract_alerts, which is
        # already a pure function of the CURRENT team_obs and needs no history of its own).
        self._seen_hosts: set = set()
        self._sessioned_hosts: set = set()

        # Review-artifact hooks (read-only from the outside; never consulted by any tool/prompt
        # builder -- purely for the development artefacts).
        self.last_rendered_prompt: tuple[str, str] | None = None   # (system, user)
        self.last_tool_transcript: list = []
        # the design: the LAST DECISION round's own final structured choice, verbatim (each entry is
        # exactly the model's own {"unit","action","target","reason"} JSON) -- nothing hidden or
        # inferred; there is no reasoning-trace field anywhere in the LLM contract to capture.
        # Only ever set on a genuine DECISION outcome (see __call__); stays None/stale on a
        # MODEL_ERROR or PROVIDER_FAILURE round, which never produced a real final choice.
        self.last_decisions: list | None = None

        # : USD per 1,000,000 tokens --
        # model is never hardcoded, so price is not either (a caller/CLI configures these).
        # Review fix item 3: default None means "not priced" -- every computed cost is then
        # None (never a fabricated 0.0 that would misread as "$0 spent"). Only EXPLICITLY
        # setting both to a real float (0.0 included) yields a real computed cost.
        self.price_input = None if price_input is None else float(price_input)
        self.price_output = None if price_output is None else float(price_output)
        # The full per-decision-round conversation log -- public, read-only from the outside,
        # for the separate conversation-JSON writer (flipthem.llm.conversation) only; nothing
        # internal ever reads it back. One entry per tick the backend was ACTUALLY invoked
        # (never one per env tick): the assembled prompt (+its hash), every tool round-trip this
        # round (each with its own usage/cost when the client returned one), the round's outcome,
        # and -- on a genuine DECISION -- the model's own final structured choice, verbatim.
        # Never a hidden/chain-of-thought reasoning field: there is no such field anywhere in the
        # LLM contract for this to capture.
        self.conversation: list = []

    # -- the drop-in team-policy call --------------------------------------------------------

    def __call__(self, game, team_obs: dict) -> dict:
        tick = game._tick
        slots = game.blue_slots if self.team == "blue" else game.red_slots

        new_event = self._update_own_log(tick, slots, team_obs)
        ctx, indices = self._build_context(game, team_obs)
        # Observe every tick, including busy ticks. Feedback remains latched until a real
        # decision receives the notice; an outage or malformed response cannot consume it.
        if self.campaign is not None:
            campaign_event = self.campaign.observe(ctx)
            if self.campaign_mode == "feedback":
                new_event = new_event or campaign_event

        actions = {slot: Sleep() for slot in slots}
        has_decidable = any(pairs for pairs in ctx.decidable.values())
        if not has_decidable:
            return actions

        if self.team == "blue":
            # the design: ctx.alerts is the FULL observation; only the separate, narrower
            # wake_alert_hosts subset (or, with the labeled wake_on_all_alerts opt-in, every
            # alert) may force an immediate re-decision.
            wake_alerts = (ctx.alerts if self.wake_on_all_alerts
                          else tuple(h for h in ctx.alerts if h in self.wake_alert_hosts))
            new_event = new_event or bool(wake_alerts)

        if not should_call(tick, self._last_call_tick, self._wake_spec, new_event,
                           has_decidable, self.max_idle_ticks):
            return actions

        system = build_system(self.team, framing=self.framing,
                               max_tool_calls=self.max_tool_calls, value=self.value,
                               tools_enabled=self.tools_enabled)
        user = build_user(ctx, tools_enabled=self.tools_enabled,
                         context_augmented=self.context_augmented)
        user += (f"\nWake limit: {self.max_idle_ticks} ticks after the last decision; "
                 "action completion or a relevant event may wake you earlier. Busy units "
                 "continue their current actions and are omitted from the options.")
        campaign_before = None
        if self.campaign is not None:
            campaign_before = self.campaign.snapshot()
            system += "\n\n" + campaign_instructions(self.campaign_mode)
            user += "\n\nRED CAMPAIGN MEMORY (compact; call campaign_state for the full ledger):\n" + json.dumps(
                self.campaign.prompt_snapshot(), sort_keys=True, ensure_ascii=False,
                allow_nan=False)
        self.last_rendered_prompt = (system, user)

        transcript_log: list = []
        tool_attempts: list = []
        request_start = len(self.adapter.request_log)

        def tool_dispatch(name, args):
            from flipthem.llm.wire import tool_json
            # Backend preserves every call in an oversized final batch. Enabled campaigns
            # return an error for excess calls before dispatch (including baseline reads), so
            # a batched remember_task cannot mutate past the shared budget. Error responses
            # still count in telemetry and transcripts, exactly like other tool errors.
            if self.campaign is not None and len(transcript_log) >= self.max_tool_calls:
                value = {"error": "campaign tool budget exhausted; choose a final decision"}
            elif self.campaign is not None and name in self._campaign_tool_names:
                value = self.campaign.dispatch(ctx, name, args or {})
            else:
                value = _dispatch_tool(ctx, name, args or {})
            payload = tool_json(value)
            transcript_log.append({"tool": name, "args": args or {}, "result": payload})
            tool_attempts.append(self.budget.spent)
            self.telemetry.tool_calls += 1
            return payload

        try:
            result = self.backend.run_decision(system, user, tool_dispatch)
        except Exception as exc:
            # Campaign tools can already have changed memory before the next request fails.
            # Preserve that audit trail, then propagate the SAME exception. Baseline behavior
            # is unchanged. Exception text may contain request details, so record only its type.
            if self.campaign is not None:
                self.last_tool_transcript = transcript_log
                round_trips, final_usage = self._campaign_round_accounting(
                    transcript_log, tool_attempts, request_start)
                self.conversation.append({
                    "tick": tick, "system": system, "system_hash": _hash_text(system),
                    "user": user, "tool_round_trips": round_trips, "outcome": "INTERRUPTED",
                    "diagnostic": {"code": "decision_interrupted",
                                   "exception_type": type(exc).__name__},
                    "final_choice": None, "final_usage": final_usage,
                    "final_cost_usd": cost_for_usage(final_usage, self.price_input,
                                                    self.price_output),
                    "campaign": {"before": campaign_before, "after": self.campaign.snapshot()},
                })
            raise
        self.last_tool_transcript = transcript_log
        self._last_call_tick = tick

        # : one conversation-log entry for THIS round, regardless of
        # outcome (a MODEL_ERROR/PROVIDER_FAILURE round is still real reproducibility data).
        # result.usages has ONE entry per PHYSICAL response (backend.py), i.e. one per TURN --
        # NOT one per tool call. A batched turn is one response (one usage) covering several calls,
        # so usage is attributed to the TURN: recorded on that turn's FIRST round-trip and None on
        # the rest (totals never double-count, per cost.py's conversation_totals), never spread
        # positionally across the individual calls. result.tool_turn_sizes gives the calls-per-turn
        # grouping; result.usages[:n_tool_turns] are the tool turns' usages and all remaining
        # entries are terminal attempts (normally one; two when a final-format repair runs).
        sizes = result.tool_turn_sizes
        n_tool_turns = len(sizes)
        round_trips = []
        ti = 0
        for t, size in enumerate(sizes):
            turn_usage = result.usages[t] if t < min(n_tool_turns, len(result.usages)) else None
            for j in range(size):
                if ti >= len(transcript_log):
                    break
                entry = transcript_log[ti]
                ti += 1
                usage = turn_usage if j == 0 else None   # the turn's usage, once, on its first call
                round_trips.append({
                    "tool": entry["tool"], "args": entry["args"], "result": entry["result"],
                    "usage": usage,
                    "cost_usd": cost_for_usage(usage, self.price_input, self.price_output),
                })
        terminal_usages = list(result.usages[n_tool_turns:])
        final_usage = terminal_usages[-1] if terminal_usages else None
        if self.campaign is not None:
            # Some backend error exits do not return tool-turn grouping. The dispatch audit
            # and physical response log still identify every executed campaign mutation.
            round_trips, _campaign_final_usage = self._campaign_round_accounting(
                transcript_log, tool_attempts, request_start)
        final_cost = cost_for_usage(final_usage, self.price_input, self.price_output)
        final_attempts = [
            {"usage": usage,
             "cost_usd": cost_for_usage(usage, self.price_input, self.price_output)}
            for usage in terminal_usages
        ]
        self.conversation.append({
            "tick": tick,
            "system": system,
            "system_hash": _hash_text(system),
            "user": user,
            "tool_round_trips": round_trips,
            "outcome": result.kind,
            "diagnostic": result.diagnostic,
            "final_choice": result.decisions if result.kind == DECISION else None,
            "final_usage": final_usage,
            "final_cost_usd": final_cost,
            # Usually one element. A semantic format repair keeps both the rejected response and
            # the recovered/terminal response here so totals and post-hoc reliability audits see
            # every billed completion. Legacy final_usage remains the last attempt for readers.
            "final_attempts": final_attempts,
        })
        repair = (result.diagnostic or {}).get("semantic_repair")
        if repair:
            self.telemetry.final_repairs_attempted += int(repair.get("attempted") or 0)
            if repair.get("recovered"):
                self.telemetry.final_repairs_recovered += 1
        if self.campaign is not None:
            if result.kind == DECISION:
                self.campaign.acknowledge()
            self.conversation[-1]["campaign"] = {
                "before": campaign_before, "after": self.campaign.snapshot(),
            }

        if result.kind == DECISION:
            self.telemetry.decisions += 1
            self._wake_spec = result.wake
            self.last_decisions = result.decisions
            by_slot = _index_decisions(result.decisions)
            for slot in slots:
                pairs = ctx.decidable.get(slot, ())
                if not pairs:
                    continue   # busy / nothing legal: stays Sleep, never resubmitted -- and NOT
                               # in the decidable denominator (v3 the design table row 4)
                self.telemetry.decidable_slot_rounds += 1
                chosen = by_slot.get(slot)
                if chosen is None:
                    # partial decisions are explicitly ALLOWED, so this is not an
                    # error -- but it is counted, and counted apart from a deliberate Sleep below.
                    self.telemetry.omitted_slots += 1
                    continue   # model didn't address this decidable unit: stays Sleep
                action_name, target = chosen
                native = compiler.compile_action(indices[slot], action_name, target)
                actions[slot] = native
                if isinstance(native, InvalidAction):
                    # THIS slot's choice was illegal; the round's other slots still executed
                    # normally. Deliberately not `model_errors` (v3 the design row 2).
                    self.telemetry.illegal_actions += 1
                else:
                    self.telemetry.valid_action_submissions += 1
                    if isinstance(native, Sleep):
                        self.telemetry.explicit_sleeps += 1   # a considered wait, not an omission
                self._record_submission(tick, slot, native)
        elif result.kind == MODEL_ERROR:
            self.telemetry.model_errors += 1
            # Re-ask on the NEXT tick rather than clearing the wake spec. A malformed response is not
            # a considered "wait": clearing it dropped the model back to `idle >= max_idle_ticks`
            # (default 10), so in a 10-step match one bad response on tick 0 meant the model was
            # never called again -> zero decisions for the whole game (seen live with two
            # production models, each a single parallel-`tool_calls` response). The output itself is
            # still REJECTED as strictly as before (InvalidAction billed below, unchanged);
            # only the re-ask cadence changes, so one stochastic protocol slip no longer decides the
            # match. PROVIDER_FAILURE deliberately keeps clearing the spec (a real outage, not the
            # model's output -- its slots stay Sleep and are never billed).
            self._wake_spec = {"after": 1}
            if (result.diagnostic or {}).get("code") == REFUSAL_CODE:
                # a CONFIRMED content refusal (provider-authored signal only, never
                # inferred from an HTTP code -- that classification happens in backend.py) is
                # counted in its own class, then scored by the SAME whole-round "no valid
                # decision" rule as any other model error. The episode is kept: it is a fact about
                # the model under this prompt, not a transport fault to be retried until it passes.
                self.telemetry.refusals += 1
            for slot in slots:
                if ctx.decidable.get(slot, ()):
                    self.telemetry.decidable_slot_rounds += 1
                    self.telemetry.model_error_slots += 1
                    native = InvalidAction()
                    actions[slot] = native
                    self._record_submission(tick, slot, native)
        elif result.kind == PROVIDER_FAILURE:
            self.telemetry.provider_failures += 1
            self.telemetry.flagged = True
            self._wake_spec = None
            # slots stay Sleep (already defaulted above); never billed as InvalidAction.

        self.telemetry.sleeps += sum(1 for a in actions.values() if isinstance(a, Sleep))
        return actions

    # -- internals ----------------------------------------------------------------------------

    def _campaign_round_accounting(self, transcript_log, tool_attempts, request_start):
        """Attribute actual response usage once per physical turn, even after interruption."""
        requests = self.adapter.request_log[request_start:]
        by_attempt = {request["attempt"]: request for request in requests}
        billed_attempts = set()
        round_trips = []
        for entry, attempt in zip(transcript_log, tool_attempts):
            usage = (by_attempt[attempt].get("usage") if attempt not in billed_attempts else None)
            billed_attempts.add(attempt)
            round_trips.append({
                **entry, "usage": usage,
                "cost_usd": cost_for_usage(usage, self.price_input, self.price_output),
            })
        # Return the latest terminal response for the legacy final_usage view. Semantic-repair
        # accounting uses Result.usages to retain every terminal attempt; retries and failed
        # transport requests have an error and carry no returned response usage.
        final_request = next((request for request in reversed(requests)
                              if request["attempt"] not in billed_attempts
                              and request.get("error") is None), None)
        return round_trips, final_request.get("usage") if final_request is not None else None

    def _build_context(self, game, team_obs: dict):
        """One legal observation boundary for live decisions and final campaign flushing."""
        slots = game.blue_slots if self.team == "blue" else game.red_slots
        indices = {}
        for slot in slots:
            # target_hosts gates BLUE's owner-specific block pairs; own_obs supplies RED's
            # held-session Withdraw pairs. Neither adds actions to the other side's menu.
            indices[slot] = compiler.build_slot_action_index(
                slot, game.env.get_action_space(slot), self.action_names,
                target_hosts=self.target_hosts, own_obs=(team_obs or {}).get(slot),
            )
        ctx = build_legal_context(
            self.team, game._tick, game.T, team_obs,
            {slot: index.legal_pairs for slot, index in indices.items()},
            self._own_log, self.target_hosts,
            include_red_sessions_by_unit=self.campaign is not None,
        )
        return ctx, indices

    def _record_submission(self, tick: int, slot: str, native) -> None:
        name = type(native).__name__
        target = self._pending_target_for(native)
        if isinstance(native, Sleep):
            return   # Sleep is the "nothing to report" default: no log entry, nothing pending
        if isinstance(native, InvalidAction):
            self._own_log.append(LogEntry(round=tick, unit=slot, action=name, target=None,
                                          result=RESULT_INVALID))
            return
        self._own_log.append(LogEntry(round=tick, unit=slot, action=name, target=target,
                                      result=RESULT_SUBMITTED))
        self._pending[slot] = (name, target)

    @staticmethod
    def _pending_target_for(native) -> str | None:
        # Block/Allow expose two native subnet parameters, while the model-facing legal pair uses
        # the protected operational hostname.  Recover that same logical target so history does
        # not show a valid firewall decision as ``target: null``.  The fixed mapping is the one
        # used by compiler.build_slot_action_index to create the native action.
        if hasattr(native, "from_subnet") and hasattr(native, "to_subnet"):
            from_subnet = str(native.from_subnet)
            to_subnet = str(native.to_subnet)
            for host, restricted in compiler.RESOURCE_RESTRICTED_SUBNET.items():
                if (str(restricted) == from_subnet
                        and str(compiler.RESOURCE_OPERATIONAL_SUBNET.get(host)) == to_subnet):
                    return host
            # Preserve an unfamiliar pair for diagnosis instead of erasing it as null.
            return f"{from_subnet}|{to_subnet}"
        if hasattr(native, "hostname") and hasattr(native, "ip_address"):
            return f"{native.ip_address}|{native.hostname}"
        if hasattr(native, "hostname"):
            return native.hostname
        if hasattr(native, "subnet"):
            return str(native.subnet)
        if hasattr(native, "ip_address"):
            return str(native.ip_address)
        return None

    def _update_own_log(self, tick: int, slots, team_obs: dict) -> bool:
        """Runs every tick regardless of whether the model is called (the design).
        Returns True iff something resolved this tick (a "new event" the wake gate should honour).

        the design: when NO slot has a pending action this tick (so the per-slot loop below creates no
        lifecycle entry at all), this tick's real observation delta (team_wide) would otherwise be
        silently lost forever -- a non-target alert that never wakes anyone, with nothing else
        pending either, previously vanished the instant the tick passed. A dedicated
        result="observed" entry (unit=action=target=None) persists it instead, appended ONLY when
        no lifecycle entry this tick already carries the same team_wide value (no double-count)."""
        new_event = False
        team_wide = self._team_wide_observed(tick, team_obs)
        observed_host_ips = self._red_observed_host_ips(team_obs) if self.team == "red" else ()
        any_lifecycle_entry = False
        for slot in slots:
            pending = self._pending.get(slot)
            if pending is None:
                continue
            any_lifecycle_entry = True
            action_name, target = pending
            obs = team_obs.get(slot, {}) if team_obs else {}
            if obs.get("success") == TernaryEnum.IN_PROGRESS:
                self._own_log.append(LogEntry(round=tick, unit=slot, action=action_name,
                                              target=target, result=RESULT_IN_PROGRESS,
                                              observed=team_wide,
                                              observed_host_ips=observed_host_ips))
                continue
            observed_action = obs.get("action")
            observed_name = type(observed_action).__name__ if observed_action is not None else None
            success = obs.get("success")
            if observed_name != action_name:
                # the tracked action never actually executed (filtered/session vanished) -- own-
                # side-legal-observation-detectable via the SAME 'action' key wrapper.py's own
                # billing reads, never touching env/state directly.
                result = RESULT_SESSION_LOST
            elif success == TernaryEnum.TRUE:
                result = RESULT_SUCCEEDED
            else:
                # FALSE ("ended, but did not succeed") or any other non-TRUE outcome: item 1 (fix
                # round 2) -- "ended" != "succeeded", never label this succeeded/"completed".
                result = RESULT_FAILED
            evidence = None
            if self.team == "blue" and action_name == "Analyse" and result == RESULT_SUCCEEDED:
                evidence = self._analyse_evidence(team_obs, slot, target)
            self._own_log.append(LogEntry(round=tick, unit=slot, action=action_name,
                                          target=target, result=result, observed=team_wide,
                                          observed_host_ips=observed_host_ips,
                                          analyse_evidence=evidence))
            new_event = True
            del self._pending[slot]

        if not any_lifecycle_entry and team_wide:
            self._own_log.append(LogEntry(round=tick, unit=None, action=None, target=None,
                                          result=RESULT_OBSERVED, observed=team_wide,
                                          observed_host_ips=observed_host_ips))
        return new_event

    @staticmethod
    def _red_observed_host_ips(team_obs: dict) -> tuple:
        """Extract exact hostname/IP pairs from RED's own structured observation.

        A pair is retained only when the observation itself establishes the relationship: an
        IP-keyed entry names ``System info.Hostname``; a hostname-keyed entry exposes an
        interface IP; or IP and hostname views carry the same RED session id.  No topology,
        environment state, action-space guess, or cross-side information is consulted.
        """
        direct = set()
        sessions = {}
        for slot, obs in (team_obs or {}).items():
            if not isinstance(obs, Mapping):
                continue
            for key, value in obs.items():
                if key in ("success", "action") or not isinstance(value, Mapping):
                    continue
                key = str(key)
                key_is_ip = _is_ip_text(key)
                system = value.get("System info") or {}
                hostname = system.get("Hostname") if isinstance(system, Mapping) else None
                if hostname is not None:
                    hostname = str(hostname)
                interface_ips = set()
                for interface in value.get("Interface") or ():
                    if isinstance(interface, Mapping) and interface.get("ip_address") is not None:
                        candidate = str(interface["ip_address"])
                        if _is_ip_text(candidate):
                            interface_ips.add(candidate)

                if key_is_ip and hostname:
                    direct.add((hostname, key))
                if not key_is_ip:
                    for ip in interface_ips:
                        direct.add((key, ip))
                if hostname:
                    for ip in interface_ips:
                        direct.add((hostname, ip))

                for session in value.get("Sessions") or ():
                    if not isinstance(session, Mapping) or session.get("agent") != slot:
                        continue
                    sid = (slot, session.get("session_id"))
                    item = sessions.setdefault(sid, {"hosts": set(), "ips": set()})
                    if key_is_ip:
                        item["ips"].add(key)
                    else:
                        item["hosts"].add(key)
                    if hostname:
                        item["hosts"].add(hostname)
                    item["ips"].update(interface_ips)

        for item in sessions.values():
            direct.update((host, ip) for host in item["hosts"] for ip in item["ips"])
        return tuple(sorted(direct))

    def _team_wide_observed(self, tick: int, team_obs: dict) -> tuple:
        """Item 2 (fix round 2): the tick's real legal observation delta -- blue: the FULL set of
        Monitor alerts this side legally received (never narrowed here; see item 6 for the
        SEPARATE, narrower subset that drives the wake trigger); red: host names newly visible
        (a successful Discover) or newly/no-longer carrying one of red's OWN sessions (a
        successful Exploit/PrivilegeEscalate, or an eviction) -- read live from team_obs, exactly
        like `flipthem.llm.context._red_holdings`, and gated the same way to sessions whose
        'agent' is one of red's own slots."""
        if self.team == "blue":
            return extract_alerts(team_obs, tick)

        own_slots = set(team_obs or {})
        current_hosts, current_sessioned = set(), set()
        for _slot, obs in (team_obs or {}).items():
            if not isinstance(obs, Mapping):
                continue
            for host, val in obs.items():
                if host in ("success", "action") or not isinstance(val, Mapping):
                    continue
                current_hosts.add(host)
                for sess in (val.get("Sessions") or ()):
                    if isinstance(sess, Mapping) and sess.get("agent") in own_slots:
                        current_sessioned.add(host)
                        break
        delta = ((current_hosts - self._seen_hosts) | (current_sessioned - self._sessioned_hosts)
                | (self._sessioned_hosts - current_sessioned))
        self._seen_hosts |= current_hosts
        self._sessioned_hosts = current_sessioned
        return tuple(sorted(delta))

    @staticmethod
    def _analyse_evidence(team_obs: dict, slot: str, target_host: str | None) -> dict:
        """Preserve scanner-returned file fields, never Sessions or hidden compromise state."""
        obs = (team_obs or {}).get(slot, {})
        host_obs = obs.get(target_host) if isinstance(obs, Mapping) else None
        files = []
        for item in (host_obs.get("Files") or ()) if isinstance(host_obs, Mapping) else ():
            if isinstance(item, Mapping):
                row = {}
                for native, key in (("File Name", "name"), ("Path", "path"),
                                    ("Density", "density"), ("Signed", "signed")):
                    value = item.get(native, item.get(key))
                    if isinstance(value, (str, int, float, bool)):
                        if not isinstance(value, float) or math.isfinite(value):
                            row[key] = value
                if row and row not in files:
                    files.append(row)
        return {"files": files}

    @property
    def own_log(self) -> tuple:
        """Read-only snapshot of this policy's own per-tick log (the design: for external reporting --
        e.g. the manifest writer -- only; nothing internal ever reads this property, only
        the private `_own_log` list it copies)."""
        return tuple(self._own_log)

    def flush(self, game) -> None:
        """Call once after the driving loop ends (episode over) so the LAST submitted action's
        resolution -- which lands the tick AFTER submission, per the observe-before-act loop
        (design the design discussion) -- is not silently dropped . A no-op
        when nothing is pending in baseline mode. Enabled campaigns also observe final own-side
        session changes when idle, without making another model request."""
        if not self._pending and self.campaign is None:
            return
        team_obs = game.team_observation(self.team)
        self._update_own_log(game._tick, list(self._pending.keys()), team_obs)
        if self.campaign is not None:
            ctx, _ = self._build_context(game, team_obs)
            self.campaign.observe(ctx)
            if self.conversation:
                self.conversation[-1]["campaign"]["final_observation"] = self.campaign.snapshot()

    def _strategy_label(self) -> str:
        """The registrable strategy label for this policy's own (tools_enabled,
        context_augmented) pair, LOOKED UP in `flipthem.anchors.LLM_STRATEGIES`.

        The label is not a binary function of `tools_enabled`: `no-tool` and `context-assisted`
        share `tools_enabled=False`, and `tool-assisted` and `context-tool-assisted` share
        `tools_enabled=True` -- each pair is separated only by `context_augmented`. This used to
        be a hand-written three-way branch, which silently reported a real
        `context-tool-assisted` run as `tool-assisted` in its ReproRecord (MatchRecord was
        correct, so the two artifacts disagreed about the same match). Reading the registry means
        a new strategy can never be mislabelled as an old one again: an unregistered combination
        raises rather than guessing the nearest label.

        Imported inside the function: `anchors` imports this module to build policies, so a
        module-level import would be circular."""
        if self.campaign is not None:
            return "campaign-" + self.campaign_mode
        from flipthem.anchors import LLM_STRATEGIES
        want = (bool(self.tools_enabled), bool(self.context_augmented))
        for name, cfg in LLM_STRATEGIES.items():
            if cfg.get("campaign_mode", "off") != "off":
                continue
            if (bool(cfg["tools_enabled"]), bool(cfg["context_augmented"])) == want:
                return name
        raise ValueError(
            f"no registered strategy for tools_enabled={want[0]}, context_augmented={want[1]}; "
            f"known: {sorted(LLM_STRATEGIES)}. Register it in anchors.LLM_STRATEGIES rather than "
            f"reporting it under another strategy's name.")

    def config_fingerprint(self) -> dict:
        """The FIXED configuration this policy ran under -- everything the dual-LLM design  requires held constant across a side's arms: "for one model in one role, freeze the
        endpoint, the output budget, the sampling settings, the history construction and the wake
        rules across all four configurations; open the pre-defined tool budget only in the
        tool-enabled conditions."

        TWO PARTS, because the two have different scopes:

        `fixed` -- must be identical across EVERY arm of a model in a role: the deployment
        (endpoint, pinned upstream, reasoning effort), the sampling and output budget, the wake
        rules, the retry rule, the declared value. The model id alone is not enough here: one
        model is served by several upstream providers at different quantizations, so a
        "tool effect" could really be a quantization change.

        `assistance` -- the tool contract: which strategy, the tool budget, the tool menu, and
        the tool-call rules. These are ALLOWED to differ between assistance conditions (a no-tool
        arm has no tool budget; that IS the manipulation) but must be identical among arms that
        SHARE an assistance condition. Excluding them outright, as an earlier version did, let a
        real confound through: two `context-tool-assisted` arms with tool caps of 1 and 4 paired
        happily and their difference was reported as a framing effect. `flipthem.analysis`
        applies each part at its own scope.

        `framing` is in neither: it is a manipulated factor in its own right and is already an
        episode-key field, so the block structure handles it.

        Plain JSON values only, so it survives `persist()` and compares by equality."""
        client = self.client
        fingerprint = {
            "fixed": {
                "model": self.model,
                "provider": getattr(client, "provider", None),
                "upstream_provider": getattr(client, "upstream_provider", None),
                "endpoint": getattr(client, "endpoint", None),
                "reasoning_effort": getattr(client, "reasoning_effort", None),
                "client_params": dict(getattr(client, "params", None) or {}),
                "adapter_params": dict(self.params),
                "max_idle_ticks": self.max_idle_ticks,
                "max_retries": self.max_retries,
                "max_final_repairs": self.max_final_repairs,
                "wake_on_all_alerts": bool(self.wake_on_all_alerts),
                "wake_alert_hosts": sorted(self.wake_alert_hosts or ()),
                "value": self.value,
            },
            "assistance": {
                "contract_version": ASSISTANCE_CONTRACT_VERSION,
                "strategy": self._strategy_label(),
                "tools_enabled": bool(self.tools_enabled),
                "context_augmented": bool(self.context_augmented),
                # The EFFECTIVE cap (0 whenever tools are off), i.e. what the loop actually ran
                # under -- not the requested number, which a no-tool arm may still carry.
                "max_tool_calls": self.max_tool_calls,
                "tool_names": _tool_names(getattr(client, "tools", None)),
                "final_tool_choice": getattr(client, "final_tool_choice", None),
                "force_first_tool": getattr(client, "force_first_tool", None),
            },
        }
        if self.campaign is not None:
            fingerprint["assistance"]["campaign"] = self._campaign_config()
        return fingerprint

    def _campaign_config(self) -> dict:
        """Enabled-only assistance contract; baseline fingerprint shape stays unchanged."""
        return {
            "mode": self.campaign_mode,
            "version": CAMPAIGN_VERSION,
            "limits": dict(CAMPAIGN_LIMITS),
            "prompt_recent_events_and_outcomes": CAMPAIGN_PROMPT_RECENT,
            "prompt_max_chars": CAMPAIGN_PROMPT_MAX_CHARS,
            "tool_budget_rule": "shared_cap_excess_batch_calls_return_error_without_dispatch",
            "effective_tool_names": sorted(
                set(_tool_names(TOOL_SCHEMAS) or ()) | self._campaign_tool_names),
        }

    def repro_record(self):
        """the design: this policy's `ReproRecord` as of the LATEST decision round -- model id,
        sampling params, the last assembled prompt's stable hash, tool config, request count
        (physical attempts, incl. retries), token usage (if the client ever reported any), and
        provider-failure count. Delegates the adapter-tracked fields entirely to the adapter.

        : also threads in the fields ONLY this policy can know --
        `provider` (read from the injected client if it has one, e.g. a real `RealClient`; `None`
        for every stub, which simply has no such attribute), `strategy` (one of the THREE
        registrable labels, via `_strategy_label()` -- see there), `tool_call_count`/
        `model_errors` (from this policy's own `Telemetry`, cumulative over the whole match so
        far), and `total_tokens`/`total_cost_usd` (summed over `self.conversation`, i.e. every
        round decided so far). Unknown request billing is represented by ``cost_complete=False``;
        known responses remain in ``known_cost_subtotal_usd`` and are never mislabeled as the
        exact total."""
        total_tokens, known_cost = conversation_totals(self.conversation)
        cost_complete = all(request.get("cost_usd") is not None
                            for request in self.adapter.request_log)
        total_cost = known_cost if cost_complete else None
        tool_config = {
            "contract_version": ASSISTANCE_CONTRACT_VERSION,
            "max_tool_calls": self.max_tool_calls, "max_idle_ticks": self.max_idle_ticks,
            "tools_enabled": self.tools_enabled, "context_augmented": self.context_augmented,
            "max_final_repairs": self.max_final_repairs,
        }
        if self.campaign is not None:
            tool_config["campaign"] = self._campaign_config()
        return self.adapter.repro_record(
            tool_config=tool_config,
            provider=getattr(self.client, "provider", None),
            upstream_provider_requested=getattr(self.client, "upstream_provider", None),
            reasoning_effort=getattr(self.client, "reasoning_effort", None),
            strategy=self._strategy_label(),
            tool_call_count=self.telemetry.tool_calls,
            model_errors=self.telemetry.model_errors,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            known_cost_subtotal_usd=known_cost,
            cost_complete=cost_complete,
        )


def _hash_text(text: str) -> str:
    """A stable sha256 digest of ONE string -- used for the conversation log's per-round system-
    prompt hash (distinct from backend.py's `_hash_prompt`, which hashes the (system, user) PAIR
    for `ReproRecord`; this one is deliberately just the system prompt alone, matching design
    doc rev.6 the design's "system prompt (text + hash)")."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_decisions(decisions) -> dict:
    out = {}
    for d in decisions or ():
        if not isinstance(d, dict):
            continue
        unit = d.get("unit")
        action = d.get("action")
        if unit is None or action is None:
            continue
        out[unit] = (action, d.get("target"))
    return out


def _dispatch_tool(ctx, name: str, args: dict) -> dict:
    # Model-generated arguments are untrusted data, not Python call-site guarantees.
    # Report a tool error so the remaining bounded loop can continue; never catch
    # implementation exceptions from inside the tool itself.
    if not isinstance(args, dict):
        return {"error": "tool arguments must be an object"}
    if name == "history":
        return history(ctx, last_n=args.get("last_n"))
    if name == "action_manual":
        if args.get("action") is not None and not isinstance(args["action"], str):
            return {"error": "action must be a string or omitted"}
        return action_manual(args.get("action"))
    if name == "known_targets":
        return known_targets(ctx)
    if name == "check_plan":
        slot = args.get("unit")
        if slot is not None and not isinstance(slot, str):
            return {"error": "unit must be a string"}
        if slot is None:
            slots = [s for s, pairs in ctx.decidable.items() if pairs]
            if len(slots) > 1:
                return {"error": "unit is required when multiple units are decidable"}
            slot = slots[0] if slots else None
        raw_pairs = args.get("pairs", [])
        if not isinstance(raw_pairs, (list, tuple)) or any(
            not isinstance(p, (list, tuple)) or len(p) != 2
            or not isinstance(p[0], str)
            or (p[1] is not None and not isinstance(p[1], str)) for p in raw_pairs
        ):
            return {"error": "pairs must contain [action string, target string or null]"}
        pairs = [tuple(p) for p in raw_pairs]
        if slot is None:
            return {"results": [], "error": "no decidable unit to check against"}
        return check_plan(ctx, slot, pairs)
    return {"error": f"unknown tool {name!r}"}


def make_llm_policy(team: str, client, model: str, framing: str = "neutral",
                     max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
                     max_idle_ticks: int = DEFAULT_MAX_IDLE_TICKS,
                     budget: RequestBudget | None = None, value: float = 1.0,
                     wake_alert_hosts: tuple | None = None,
                     wake_on_all_alerts: bool = False,
                     max_retries: int = 0, params: dict | None = None,
                     tools_enabled: bool = True, price_input: float | None = None,
                     price_output: float | None = None, max_cost_usd: float | None = None,
                     context_augmented: bool = False, campaign_mode: str = "off",
                     max_final_repairs: int = 0) -> LLMPolicy:
    return LLMPolicy(team=team, client=client, model=model, framing=framing,
                      max_tool_calls=max_tool_calls, max_idle_ticks=max_idle_ticks,
                      budget=budget, value=value, wake_alert_hosts=wake_alert_hosts,
                      wake_on_all_alerts=wake_on_all_alerts, max_retries=max_retries,
                      params=params, tools_enabled=tools_enabled, price_input=price_input,
                      price_output=price_output, max_cost_usd=max_cost_usd,
                      context_augmented=context_augmented, campaign_mode=campaign_mode,
                      max_final_repairs=max_final_repairs)

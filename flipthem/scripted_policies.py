"""Constrained scripted BLUE/RED baselines on the LLM's own legal interface
(scripted-baseline design, the design).

Two layers, deliberately separated:

1. A RULE is a pure decision function `decide(ctx: LegalContext) -> {slot: (action_name, target)}`.
   It receives nothing but the `LegalContext` an LLM side of the SAME team would receive -- never
   `game`/`env`/`state`/controller objects -- and may only pick pairs already present in
   `ctx.decidable[slot]`. The scripted BLUE rules (S0/S1/S2) are stateless (the whole decision is a
   function of the current context). A0 (RED) keeps NO hidden state either: its scan/exploration/
   cooldown ledger is reconstructed every call from `ctx.own_log` (unit/action/target/result/round)
   and current holdings from `ctx.holdings`, so identical contexts always yield identical choices
   and there is no RNG anywhere.

2. `ScriptedTeamPolicy` is the outer adapter with the ordinary team-policy calling convention
   `policy(game, team_obs) -> {slot: Action}`. It builds the context through the SAME code path an
   `LLMPolicy` uses -- own_log upkeep, `compiler.build_slot_action_index`, `build_legal_context` --
   by delegating to an inert, network-free `LLMPolicy` instance (its backend/adapter are never
   invoked; there is no client to call), then compiles the rule's choices with the unchanged
   `compiler.compile_action`. Execution timing, action filtering, failure and FT billing therefore
   stay entirely with the native executor and `FlipThemCC4`; the script cannot bypass any of them.

This module is intentionally separate from the prompt-frozen sources (`anchors.py`, `compiler.py`,
`context.py`, `agent.py`, ...): nothing here edits them, so the prompt freeze is unaffected. A
change to any rule's behaviour must bump that rule's `version` (the scripted-rule specification) so old and new
samples never pool.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Callable

from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.Action import InvalidAction

from flipthem._scripted_rule_texts import RULE_TEXTS
from flipthem.llm import compiler
from flipthem.llm.agent import LLMPolicy
from flipthem.llm.context import (
    LegalContext, RESULT_SUCCEEDED, RESULT_FAILED,
)

# Version of the adapter/observation contract every rule in this module runs behind. Bump when the
# context construction, menu validation or compile step changes -- it rides in every rule's
# fingerprint so an executor change can never masquerade as the same scripted sample.
EXECUTOR_CONTRACT_VERSION = "scripted-legal-adapter-v1"

# Owner of each scored target, as fixed by the compiler (zone A -> blue_agent_1, zone B ->
# blue_agent_3). Read from the compiler rather than duplicated so the two cannot drift.
TARGET_OWNER = dict(compiler.RESOURCE_OWNER)

# Action names (must match the compiler's own spelling).
_RESTORE = "Restore"
_BLOCK = "BlockTrafficZone"
_PRIVESC = "PrivilegeEscalate"
_EXPLOIT = "ExploitRemoteService"
_AGGRESSIVE = "AggressiveServiceDiscovery"
_DISCOVER = "DiscoverRemoteSystems"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ScriptedRuleViolation(RuntimeError):
    """A rule proposed a pair outside its slot's current legal menu, or for a busy/undecidable
    slot, or a legal-looking pair that did not compile. This is an implementation defect, never a
    game outcome, so it stops the episode rather than silently degrading to Sleep."""


# --------------------------------------------------------------------------------------------
# Rule dataclass
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScriptedRule:
    team: str                       # "blue" | "red"
    key: str                        # short schedule code: S0/S1/S2/A0
    policy_id: str                  # frozen id, e.g. "alert-restore-legal-v1"
    version: str
    params: dict
    tie_break: str
    info_sources: tuple
    decide: Callable[[LegalContext], dict] = field(repr=False, compare=False)

    @property
    def label(self) -> str:
        return f"script:{self.policy_id}"

    def rule_text_sha256(self) -> str:
        """sha256 of the frozen specification prose (the scripted-rule specification), the value pinned in the
        design's frozen-inputs.json. NOT the implementation source -- the two are frozen and
        checked separately (the scripted-rule specification 'rule full-text hash' vs 'implementation source hash')."""
        return _sha256_text(RULE_TEXTS[self.key])

    def fingerprint(self) -> dict:
        """The frozen identity of one scripted policy (the scripted-rule specification 'freeze fields'): policy
        kind/id/version, rule full-text hash, all parameters, the selection/tie-break rule, the
        information sources it reads, and the executor/observation contract version. Plain JSON
        values only, so it survives persistence and compares by equality."""
        return {
            "kind": "scripted",
            "team": self.team,
            "key": self.key,
            "policy_id": self.policy_id,
            "version": self.version,
            "rule_text_sha256": self.rule_text_sha256(),
            "params": dict(self.params),
            "tie_break": self.tie_break,
            "info_sources": list(self.info_sources),
            "executor_contract_version": EXECUTOR_CONTRACT_VERSION,
        }


# --------------------------------------------------------------------------------------------
# Legal-menu helpers (shared by the rules)
# --------------------------------------------------------------------------------------------

def _listed(ctx: LegalContext, slot: str, action_name: str, target) -> bool:
    """Is (action_name, target) currently a legal, decidable pair for `slot`?"""
    return (action_name, target) in ctx.decidable.get(slot, ())


def _targets_for(ctx: LegalContext, slot: str, action_name: str):
    """Every currently-legal target for one action on one slot, in the menu's own order."""
    return [target for (name, target) in ctx.decidable.get(slot, ()) if name == action_name]


# --------------------------------------------------------------------------------------------
# BLUE rules
# --------------------------------------------------------------------------------------------

def _decide_s0(ctx: LegalContext) -> dict:
    """alert-restore-legal-v1: skip the initial snapshot (tick 0); from tick 1, Restore a scored
    target only when THIS round's own legal Monitor alert names it, its owner is decidable and
    (Restore, target) is listed. No queueing of old alerts, no cooldown, no execution retry."""
    out: dict = {}
    if ctx.round < 1:
        return out
    alerts = set(ctx.alerts)
    for target, owner in TARGET_OWNER.items():
        if target in alerts and _listed(ctx, owner, _RESTORE, target):
            out[owner] = (_RESTORE, target)
    return out


def _decide_s1(ctx: LegalContext, *, k: int = 10, phase: int = 0) -> dict:
    """periodic10-legal-v1: at decision ticks phase, phase+k, ... each owner submits a legal
    Restore to its own scored target. Only decidability + the menu are checked (never infection or
    alerts). A busy or unlisted opportunity is skipped with no catch-up and no drift of the next
    scheduled tick."""
    out: dict = {}
    if ctx.round < phase or (ctx.round - phase) % k != 0:
        return out
    for target, owner in TARGET_OWNER.items():
        if _listed(ctx, owner, _RESTORE, target):
            out[owner] = (_RESTORE, target)
    return out


def _decide_s2(ctx: LegalContext, *, submission_tick: int = 0) -> dict:
    """block-once-legal-v1: only at `submission_tick`, each owner selects the BlockTrafficZone that
    protects its scored target (compiler target = the operational server). One attempt: no retry,
    no delayed re-issue, no Restore, no Allow. Every later tick is Sleep."""
    out: dict = {}
    if ctx.round != submission_tick:
        return out
    for target, owner in TARGET_OWNER.items():
        if _listed(ctx, owner, _BLOCK, target):
            out[owner] = (_BLOCK, target)
    return out


# --------------------------------------------------------------------------------------------
# RED rule A0 -- legal-rule-red-v1
# --------------------------------------------------------------------------------------------

def _a0_ledger(ctx: LegalContext) -> dict:
    """Reconstruct A0's per-unit knowledge purely from `ctx.own_log` (the scripted-rule specification): the only
    memory A0 keeps is what its own completed legal log can prove. Nothing here reads holdings,
    campaign facts, or any hidden state -- holdings are consulted live at decision time.

    Returns, keyed by (unit):
      scanned_ok[unit]      -> set of IPs this unit successfully AggressiveServiceDiscovery'd
      explored_ok[unit]     -> {subnet: last_success_round}
    and, keyed by (unit, action, target):
      failed_at             -> the LAST failed-completion round (drives the 10-tick cooldown)
      submit_count          -> number of times submitted (tie-break)
      last_submit           -> last submission round (tie-break)
    """
    scanned_ok: dict = {}
    explored_ok: dict = {}
    failed_at: dict = {}
    submit_count: dict = {}
    last_submit: dict = {}
    for e in ctx.own_log:
        if e.unit is None or e.action is None:
            continue
        triple = (e.unit, e.action, e.target)
        if e.result == "submitted":
            submit_count[triple] = submit_count.get(triple, 0) + 1
            last_submit[triple] = e.round
        elif e.result == RESULT_SUCCEEDED:
            if e.action == _AGGRESSIVE and e.target is not None:
                scanned_ok.setdefault(e.unit, set()).add(e.target)
            elif e.action == _DISCOVER and e.target is not None:
                prev = explored_ok.setdefault(e.unit, {})
                if e.round >= prev.get(e.target, -1):
                    prev[e.target] = e.round
        elif e.result == RESULT_FAILED:
            # "execution failure": ended without success. Cools THIS exact (unit,action,target).
            failed_at[triple] = max(failed_at.get(triple, -1), e.round)
    return {"scanned_ok": scanned_ok, "explored_ok": explored_ok,
            "failed_at": failed_at, "submit_count": submit_count, "last_submit": last_submit}


def _a0_held_ips(ctx: LegalContext) -> set:
    """The team's KNOWN currently-held IP set: only IPs that a currently-held session host maps to
    through the legally-observed host->ips association (the scripted-rule specification). A missing association
    means 'not known held', never 'proven not held'."""
    holdings = ctx.holdings or {}
    sessions = set(holdings.get("sessions") or ())
    host_to_ips = ((holdings.get("discovered") or {}).get("host_to_ips") or {})
    held: set = set()
    for host in sessions:
        for ip in host_to_ips.get(host, ()):
            held.add(ip)
    return held


def _a0_pick(ctx: LegalContext, ledger: dict, slot: str, action_name: str, candidates) -> "tuple | None":
    """From `candidates` (targets), drop any (slot, action, target) still inside its 10-tick
    cooldown, then choose the winner by the frozen tie-break: submission_count ascending,
    last_submission_tick ascending (never-submitted sorts first), target string lexical."""
    cooldown = int(A0.params["failure_cooldown_ticks"])
    eligible = []
    for target in candidates:
        triple = (slot, action_name, target)
        failed_round = ledger["failed_at"].get(triple)
        if failed_round is not None and ctx.round - failed_round < cooldown:
            continue
        eligible.append(target)
    if not eligible:
        return None

    def sort_key(target):
        triple = (slot, action_name, target)
        count = ledger["submit_count"].get(triple, 0)
        last = ledger["last_submit"].get(triple, float("-inf"))
        return (count, last, str(target))

    return (action_name, min(eligible, key=sort_key))


def _decide_a0(ctx: LegalContext) -> dict:
    """legal-rule-red-v1: a deterministic, observation-limited scan -> exploit -> escalate state
    machine over the ORDINARY red LegalContext (the scripted-rule specification). Units are processed in name
    order; each unit selects at most one action by the fixed priority below. Never uses the native
    `fsm`'s wider menu or campaign facts."""
    if ctx.team != "red":
        raise ScriptedRuleViolation("A0 is a RED-only rule")
    ledger = _a0_ledger(ctx)
    root = set((ctx.holdings or {}).get("root") or ())
    held_ips = _a0_held_ips(ctx)
    refresh = int(A0.params["subnet_refresh_ticks"])

    out: dict = {}
    for slot in sorted(ctx.decidable):
        if not ctx.decidable[slot]:
            continue  # busy / nothing legal: no replacement action, stays Sleep

        scanned = ledger["scanned_ok"].get(slot, set())
        explored = ledger["explored_ok"].get(slot, {})

        # 1. PrivilegeEscalate a listed host not already team-rooted (compiler already checked the
        #    session precondition for this slot).
        cand = [h for h in _targets_for(ctx, slot, _PRIVESC) if h not in root]
        choice = _a0_pick(ctx, ledger, slot, _PRIVESC, cand)

        # 2. Exploit an IP this slot successfully scanned and not in the known-held set.
        if choice is None:
            cand = [ip for ip in _targets_for(ctx, slot, _EXPLOIT)
                    if ip in scanned and ip not in held_ips]
            choice = _a0_pick(ctx, ledger, slot, _EXPLOIT, cand)

        # 3. Aggressively scan a visible IP this slot has not yet successfully scanned.
        if choice is None:
            cand = [ip for ip in _targets_for(ctx, slot, _AGGRESSIVE) if ip not in scanned]
            choice = _a0_pick(ctx, ledger, slot, _AGGRESSIVE, cand)

        # 4. Discover an unexplored visible subnet.
        if choice is None:
            cand = [s for s in _targets_for(ctx, slot, _DISCOVER) if s not in explored]
            choice = _a0_pick(ctx, ledger, slot, _DISCOVER, cand)

        # 5. Refresh a subnet not successfully explored in the last `refresh` ticks; else Sleep.
        if choice is None:
            cand = [s for s in _targets_for(ctx, slot, _DISCOVER)
                    if s in explored and ctx.round - explored[s] >= refresh]
            choice = _a0_pick(ctx, ledger, slot, _DISCOVER, cand)

        if choice is not None:
            out[slot] = choice
    return out


# --------------------------------------------------------------------------------------------
# Frozen rule registry
# --------------------------------------------------------------------------------------------

S0 = ScriptedRule(
    team="blue", key="S0", policy_id="alert-restore-legal-v1", version="1",
    params={"initial_skip_ticks": 1, "queue_old_alerts": False, "execution_retries": 0},
    tie_break="none", info_sources=("round", "decidable", "alerts", "target_hosts"),
    decide=_decide_s0)

S1 = ScriptedRule(
    team="blue", key="S1", policy_id="periodic10-legal-v1", version="1",
    params={"k": 10, "phase": 0, "catch_up": False},
    tie_break="none", info_sources=("round", "decidable", "target_hosts"),
    decide=_decide_s1)

S2 = ScriptedRule(
    team="blue", key="S2", policy_id="block-once-legal-v1", version="1",
    params={"submission_tick": 0, "execution_retries": 0, "restore": False, "allow": False},
    tie_break="none", info_sources=("round", "decidable", "target_hosts"),
    decide=_decide_s2)

A0 = ScriptedRule(
    team="red", key="A0", policy_id="legal-rule-red-v1", version="1",
    params={"failure_cooldown_ticks": 10, "subnet_refresh_ticks": 20,
            "service_scan": _AGGRESSIVE,
            "tie_break": ["submission_count_ascending", "last_submission_tick_ascending",
                          "target_lexical"],
            "rng": None, "information_profile": "ordinary_red_legal_context",
            "include_red_sessions_by_unit": False, "root_filter": "team_holdings_root",
            "held_ip_mapping": "observed_host_to_ips_join_team_current_sessions",
            "unmapped_ip_semantics": "not_known_held_not_proven_absent"},
    tie_break="submission_count_ascending,last_submission_tick_ascending,target_lexical",
    info_sources=("round", "decidable", "own_log", "holdings"),
    decide=_decide_a0)

_RULES = {r.key: r for r in (S0, S1, S2, A0)}
_RULES_BY_ID = {r.policy_id: r for r in (S0, S1, S2, A0)}
SCRIPTED_BLUE = ("S0", "S1", "S2")
SCRIPTED_RED = ("A0",)


def get_rule(name: str) -> ScriptedRule:
    """Resolve a schedule code (S0/S1/S2/A0) or a frozen policy_id to its ScriptedRule."""
    if name in _RULES:
        return _RULES[name]
    if name in _RULES_BY_ID:
        return _RULES_BY_ID[name]
    raise ValueError(f"unknown scripted rule {name!r}; expected one of "
                     f"{sorted(_RULES) + sorted(_RULES_BY_ID)}")


# --------------------------------------------------------------------------------------------
# Team-policy adapter
# --------------------------------------------------------------------------------------------

class ScriptedTeamPolicy:
    """`policy(game, team_obs) -> {slot: Action}` for one scripted rule on one team.

    Reuses `LLMPolicy`'s own-log upkeep and context construction verbatim (an inert delegate whose
    backend/adapter are never invoked -- there is no client), so a scripted side sees byte-for-byte
    the SAME `LegalContext` an LLM of that team would, and the isolation guarantees carry over
    unchanged. The delegate is NOT an `LLMPolicy` to the outside: `run_match` keys telemetry/repro/
    fingerprint collection off `isinstance(..., LLMPolicy)`, so a `ScriptedTeamPolicy` contributes
    no model telemetry and needs no client/budget."""

    def __init__(self, rule: ScriptedRule):
        self.rule = rule
        self.team = rule.team
        # An inert delegate: client is None and the backend is never called. Constructing it makes
        # no network access (ServiceAdapter merely stores the client); tools are disabled so the
        # effective tool budget is 0.
        self._delegate = LLMPolicy(team=rule.team, client=None, model=f"scripted:{rule.policy_id}",
                                   tools_enabled=False, context_augmented=False)

    def __call__(self, game, team_obs: dict) -> dict:
        tick = game._tick
        slots = game.blue_slots if self.team == "blue" else game.red_slots
        # Own-log upkeep every tick (busy/idle alike), identical to the LLM path.
        self._delegate._update_own_log(tick, slots, team_obs)
        ctx, indices = self._delegate._build_context(game, team_obs)

        actions = {slot: Sleep() for slot in slots}
        choices = self.rule.decide(ctx)
        for slot, choice in choices.items():
            legal = ctx.decidable.get(slot, ())
            if not legal:
                raise ScriptedRuleViolation(
                    f"{self.rule.label}: chose {choice!r} for busy/undecidable slot {slot!r}")
            if choice not in legal:
                raise ScriptedRuleViolation(
                    f"{self.rule.label}: {choice!r} not in slot {slot!r} legal menu")
            native = compiler.compile_action(indices[slot], choice[0], choice[1])
            if isinstance(native, InvalidAction):
                raise ScriptedRuleViolation(
                    f"{self.rule.label}: legal pair {choice!r} failed to compile for {slot!r}")
            actions[slot] = native
            self._delegate._record_submission(tick, slot, native)
        return actions

    def flush(self, game) -> None:
        """Resolve the last submitted action's outcome after the driving loop ends (same reason as
        LLMPolicy.flush). Safe to call always; makes no model/budget calls."""
        self._delegate.flush(game)

    @property
    def own_log(self) -> tuple:
        return self._delegate.own_log

    def fingerprint(self) -> dict:
        return self.rule.fingerprint()


def scripted_policy(name: str) -> ScriptedTeamPolicy:
    """Fresh `ScriptedTeamPolicy` for schedule code or policy_id `name`."""
    return ScriptedTeamPolicy(get_rule(name))


def implementation_sha256() -> str:
    """sha256 of THIS module's source with canonical newlines -- the 'implementation source hash'
    the design freezes alongside each rule's text hash. New code is allowed a new value (README
    the design); it is recorded, not asserted equal to a pre-implementation null."""
    src = inspect.getsource(inspect.getmodule(implementation_sha256))
    return _sha256_text(src.replace("\r\n", "\n").replace("\r", "\n"))

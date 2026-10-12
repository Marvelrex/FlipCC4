"""Within-episode whole-team configuration switch (configuration-switching design).

A `SwitchingTeamPolicy` composes two already-built `LLMPolicy` components of the SAME team (e.g.
Blue B0 and B1) into ONE controller that switches the *whole team* from the start component
to the other component **at most once**, at a legal decision opportunity, under a frozen selector:

* **pi_E** -- switch at the first legal decision opportunity where the own-legal event E is true,
  otherwise at the first legal opportunity at or after the frozen deadline.
* **pi_T** -- identical components / direction / deadline / handover / budget, but the selector's event
  input is always False, so it switches only by the deadline rule.

Invariants enforced here (design the design):
- **Whole-team selection.** Exactly one complete component decides every tick; slots are never mixed.
- **Only the active component runs.** The inactive component's `__call__` is never invoked -- no LLM
  request, no tool call, no counterfactual plan, no free warm-up.
- **One true ledger + in-flight continuity.** Both components share the single `_own_log` / `_pending`
  (and observed-host sets); a started native action keeps running; nothing is reset at the switch.
- **Global clock & wake continuity.** The team tick, the last-actual-call tick and the wake spec are
  carried across the switch; the new component does not reset them.
- **One shared budget.** Both components are built with the SAME `RequestBudget`; switching claims no
  second budget and no extra decision opportunity.
- **No fabricated history.** The new component inherits the real ledger as-is; it is never pretended
  to have been running all along.

The components' frozen configs (prompts, tools, client, wake rules, budget) are untouched -- this
module only chooses which one is active. Zero-switch behaviour must equal the plain component
(verified in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from flipthem.llm.agent import LLMPolicy, Telemetry
from flipthem.llm.context import extract_alerts

# ---- Frozen selector parameters (configuration-switching design) ----------------
# All drawn from BLUE's own legally available inputs; none reads evaluator seed, hidden Red
# privileges, true control state, or opponent private history.
FROZEN_EVENT = "blue_monitor_alert_on_scored_target_round_ge_1"
FROZEN_DIRECTION = "B0->B1"      # start = incumbent B0, switch to B1
FROZEN_DEADLINE_TICK = 100       # half of the nominal horizon T=200
FROZEN_HANDOVER = "shared_true_ledger_and_pending__no_private_memory__wake_and_budget_continued"
SELECTOR_CONTRACT_VERSION = "switching-selector-v1"


@dataclass(frozen=True)
class Selector:
    """A frozen selector. `event_enabled=True` is pi_E; `False` is the pi_T timing control."""
    arm: str                                  # "pi_E" | "pi_T"
    event_enabled: bool
    deadline_tick: int = FROZEN_DEADLINE_TICK
    event: str = FROZEN_EVENT
    direction: str = FROZEN_DIRECTION
    handover: str = FROZEN_HANDOVER

    def fingerprint(self) -> dict:
        return {"contract_version": SELECTOR_CONTRACT_VERSION, **asdict(self)}


def pi_E_selector() -> Selector:
    return Selector(arm="pi_E", event_enabled=True)


def pi_T_selector() -> Selector:
    return Selector(arm="pi_T", event_enabled=False)


def _merge_telemetry(a: Telemetry, b: Telemetry) -> Telemetry:
    """Sum the two components' telemetry for match-level reporting (only the active one ever acts, so
    this is the real per-match total across the switch). `flagged` is OR."""
    out = Telemetry()
    for f in out.__dataclass_fields__:
        va, vb = getattr(a, f), getattr(b, f)
        setattr(out, f, (bool(va) or bool(vb)) if isinstance(va, bool) else (va + vb))
    return out


class SwitchingTeamPolicy:
    """`policy(game, team_obs) -> {slot: Action}` -- a one-shot whole-team component switch."""

    def __init__(self, start_component: LLMPolicy, other_component: LLMPolicy, selector: Selector):
        if start_component.team != other_component.team:
            raise ValueError("switch components must be the same team")
        if start_component.budget is not other_component.budget:
            raise ValueError("switch components must share ONE RequestBudget (design sect. 3)")
        if start_component.adapter.max_cost_usd != other_component.adapter.max_cost_usd:
            raise ValueError("switch components must carry the same per-side cost stop-loss")
        self.team = start_component.team
        self.comp = [start_component, other_component]
        self.selector = selector
        self._active = 0
        self._switched = False

        # Handover: the OTHER component inherits the single true ledger + in-flight actions + the
        # observed-host sets (same objects, mutated in place by whichever component is active). No
        # private memory is transferred because B0/B1 carry none (campaign is off); nothing is
        # fabricated -- the new component simply reads the real history when it becomes active.
        s, o = start_component, other_component
        o._own_log = s._own_log
        o._pending = s._pending
        o._seen_hosts = s._seen_hosts
        o._sessioned_hosts = s._sessioned_hosts

        # Authoritative wake state, carried across the switch so the new component never resets it.
        self._last_call_tick = s._last_call_tick
        self._wake_spec = s._wake_spec

        self.switch_log = {
            "arm": selector.arm,
            "selector": selector.fingerprint(),
            "start_component": s._strategy_label(),
            "other_component": o._strategy_label(),
            "event_first_available_tick": None,   # first tick E was legally true (even if not switched)
            "switch_effective_tick": None,        # tick the active component actually changed
            "switch_reason": None,                # "event" | "deadline" | None
            "switched": False,
            "budget_spent_final": None,
        }

    # -- the drop-in team-policy call --------------------------------------------------------

    def _event_true(self, team_obs: dict, tick: int) -> bool:
        """Own-legal event E: a BLUE Monitor alert on a scored target at round >= 1. `extract_alerts`
        returns () at tick 0 (the initial snapshot), so this is a within-episode feedback change,
        never initial-scenario identification, and never ground truth."""
        if tick < 1:
            return False
        targets = set(self.comp[0].target_hosts)
        return bool(set(extract_alerts(team_obs, tick)) & targets)

    def _has_decidable(self, game, team_obs: dict) -> bool:
        """A legal decision opportunity: the team has at least one decidable slot this tick. Built
        read-only via the active component's own context path (no model/tool call, no ledger
        mutation); decidability is a function of the game/observation, identical for both components."""
        ctx, _ = self.comp[self._active]._build_context(game, team_obs)
        return any(pairs for pairs in ctx.decidable.values())

    def __call__(self, game, team_obs: dict) -> dict:
        tick = game._tick

        event_true = self._event_true(team_obs, tick)
        if event_true and self.switch_log["event_first_available_tick"] is None:
            self.switch_log["event_first_available_tick"] = tick

        if not self._switched:
            by_event = self.selector.event_enabled and event_true
            by_deadline = tick >= self.selector.deadline_tick
            if (by_event or by_deadline) and self._has_decidable(game, team_obs):
                self._active = 1
                self._switched = True
                self._carry_cost_cap()
                self.switch_log["switched"] = True
                self.switch_log["switch_effective_tick"] = tick
                self.switch_log["switch_reason"] = "event" if by_event else "deadline"

        active = self.comp[self._active]
        # Carry the single wake state into the active component, and read it back -- continuous
        # across a switch, never reset by the new component's initialisation.
        active._last_call_tick = self._last_call_tick
        active._wake_spec = self._wake_spec
        actions = active(game, team_obs)
        self._last_call_tick = active._last_call_tick
        self._wake_spec = active._wake_spec
        return actions

    def _carry_cost_cap(self) -> None:
        """The per-side cost stop-loss is ONE cap for the whole side, like the request budget: the
        incoming component may spend only what the outgoing one left (never a fresh full cap)."""
        start, other = self.comp[0].adapter, self.comp[1].adapter
        if start.max_cost_usd is not None:
            other.max_cost_usd = max(0.0, start.max_cost_usd - start._cost_spent)
        self.switch_log["cost_spent_before_switch"] = start._cost_spent
        self.switch_log["cost_cap_remaining_after_switch"] = other.max_cost_usd

    def flush(self, game) -> None:
        """Resolve the last action's outcome on the single shared ledger (via the active component),
        and record the final shared-budget total."""
        self.comp[self._active].flush(game)
        self.switch_log["budget_spent_final"] = self.comp[0].budget.spent

    # -- read-only views for the runner ------------------------------------------------------

    @property
    def own_log(self) -> tuple:
        return self.comp[self._active].own_log   # the single shared ledger

    @property
    def telemetry(self) -> Telemetry:
        return _merge_telemetry(self.comp[0].telemetry, self.comp[1].telemetry)

    @property
    def components(self):
        return tuple(self.comp)

    def fingerprint(self) -> dict:
        return {
            "kind": "switching",
            "arm": self.selector.arm,
            "selector": self.selector.fingerprint(),
            "start_component": self.comp[0].config_fingerprint(),
            "other_component": self.comp[1].config_fingerprint(),
        }

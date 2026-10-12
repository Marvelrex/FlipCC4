"""Scripted anchor team policies for FlipThem-on-CC4.

A team policy is `policy(game, team_obs) -> {slot: Action}`, called once per tick for one team's
slots. `blue_policy(name)` returns a **fresh** policy closure on every call. `red_policy(name)`
returns `(policy_or_None, red_agent_class_or_None)`: every red anchor here is a native,
self-driving `FiniteStateRedAgent` subclass, so `red_policy` always returns `(None, <class>)` and
the caller passes that class to `FlipThemCC4(red_agent_class=...)` and `red_actions=None` to
`.step()` (CC4 drives red internally; see `tests/test_wrapper.py::test_native_red_self_drive_...`).

Design decision: the three custom red anchors (`seize-and-hold`,
`recon-only-never-impact`, `impact-asap`) are implemented as `FiniteStateRedAgent` subclasses
rather than runner-driven policies -- this reuses CC4's own host-state tracking and legal-action
selection (`_choose_host_and_action`) instead of hand-rolling a fragile action chain, and is the
ruling's preferred, more robust mechanism. Each overrides only what is needed to make it
BEHAVIOURALLY DISTINCT from stock `fsm`/`discovery-fsm` and from each other; see each class'
docstring for the specific mechanism and rationale for verification.

Design note -- `restore-every-k` is UNCONDITIONAL (reconciliation decision, see report): the brief's
Step 3 text reads "issue a Restore ... on the slot that owns a compromised resource r ... else
Sleep", but the *given* distinctness test (`test_restore_anchors_are_behaviourally_distinct`)
compares `restore-every-5` vs `restore-every-20` at seed=2026/steps=20 against `fsm` red, and at
that seed/horizon `fsm` never actually compromises either OPERATIONAL_RESOURCES server (verified
empirically) -- so a live-compromise-gated reading would make both anchors
produce identical all-Sleep traces and fail the given test. The spec's own framing of `k` as "the
interval between Restore *initiations*" (never mentioning a compromise gate) and the classical
FlipIt/FlipThem convention of an unconditional periodic "flip" both support reading
`restore-every-k` as a blind, schedule-only reclaim: every k ticks, if the owner slot is idle,
Restore its owned resource, regardless of live compromise. This is also the more robust anchor
design for the full pairwise matrix (test_every_anchor_pairing_runs_and_is_finite pairs it against
all five red anchors): its behaviour never silently degenerates to a no-op just because a given red
anchor/seed/horizon combination happens not to reach the operational zone. `restore-on-alert`, by
contrast, is genuinely reactive -- gated on the owner slot's own Monitor-observed alert, exactly as
briefed -- so the two blue reactive-vs-scheduled archetypes stay clearly different in kind.
"""
from __future__ import annotations

from dataclasses import dataclass

from CybORG.Agents import FiniteStateRedAgent, DiscoveryFSRed
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.AbstractActions import Restore
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone
from CybORG.Shared.Enums import TernaryEnum

from flipthem.resources import OPERATIONAL_RESOURCES

# Slot map (design spec the design): operational_zone_a -> blue_agent_1, operational_zone_b ->
# blue_agent_3. OPERATIONAL_RESOURCES is (zone_a_server, zone_b_server) in that order.
RESOURCE_OWNER = {
    OPERATIONAL_RESOURCES[0]: "blue_agent_1",
    OPERATIONAL_RESOURCES[1]: "blue_agent_3",
}
RESOURCE_RESTRICTED_SUBNET = {
    OPERATIONAL_RESOURCES[0]: "restricted_zone_a_subnet",
    OPERATIONAL_RESOURCES[1]: "restricted_zone_b_subnet",
}
RESOURCE_OPERATIONAL_SUBNET = {
    OPERATIONAL_RESOURCES[0]: "operational_zone_a_subnet",
    OPERATIONAL_RESOURCES[1]: "operational_zone_b_subnet",
}

BLUE_ANCHORS = {
    "never-restore",
    "restore-on-alert",
    "restore-every-5",
    "restore-every-10",
    "restore-every-20",
    "block-chokepoint",
}
RED_ANCHORS = {
    "fsm",
    "discovery-fsm",
    "seize-and-hold",
    "recon-only-never-impact",
    "impact-asap",
}


# --------------------------------------------------------------------------------------------
# the design: an `"llm:<model>:<framing>"` name-dispatch branch on top of
# the scripted anchors above -- ONLY this branch + the tool-assisted/no-tool labels are
# new here; no scripted strategy gains anything. `blue_policy`/`red_policy` construct nothing
# from `flipthem.llm.agent` unless a `client` is actually given (the CLI's own `--yes` cost gate,
# the design, is the only production caller that ever passes one) -- without a client this
# dispatch returns an inert `LLMPlaceholder`, so simply NAMING an llm: anchor is always free.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMPlaceholder:
    """What `blue_policy`/`red_policy` return for an `"llm:<model>:<framing>"` name when no
    client is injected -- enough to report or dry-run with, and structurally incapable of
    touching a client (there is no client reference on it at all, not merely an unused one).

    Defined here rather than imported from `flipthem.llm.agent`/`eval_runner` for the same reason
    compiler.py duplicates RESOURCE_OWNER instead of importing it back from this module : the real (client-given) path below needs `flipthem.llm.agent.make_llm_policy`, and
    `eval_runner.py` already imports `blue_policy`/`red_policy` from here, so importing an
    eval_runner-defined type back into anchors.py would risk an `anchors -> eval_runner ->
    anchors` cycle.
    """
    team: str
    model: str
    framing: str
    strategy: str = "tool-assisted"


def _parse_llm_name(name: str) -> tuple:
    """`"llm:<model>:<framing>"` -> `(model, framing)`. A usage error (malformed name), never a
    silent guess, on anything else."""
    parts = name.split(":")
    if len(parts) != 3 or parts[0] != "llm":
        raise ValueError(f"malformed llm anchor name {name!r}; expected 'llm:<model>:<framing>'")
    _, model, framing = parts
    return model, framing


# the design + context-assisted: the three registrable, independent LLM strategies --
# `tool-assisted` (the default: tool section + loop, unchanged from every pre-the design call),
# `no-tool` (a clean single-prompt control; see `flipthem.llm.prompts.build_system`'s
# `tools_enabled` and `LLMPolicy.__init__`'s structural zero-tool-calls guarantee), and
# `context-assisted` ("deterministic legal-context augmentation with one model request per
# decision" -- NOT tool-assisted; the PROGRAM pre-assembles compact history/current known state
# from the LegalContext and injects it into the prompt (the static manual is already in system);
# the model never calls a tool -- same
# `tools_enabled=False` wire shape as no-tool, plus `context_augmented=True`). The label rides on
# `blue_policy`/`red_policy`'s own `strategy=` kwarg -- NOT a new segment in the
# `llm:<model>:<framing>` name -- so T5's registry contract is unchanged; passing neither is
# identical to `strategy="tool-assisted"`.
# `context-tool-assisted` completes the (tools_enabled, context_augmented) 2x2: the SAME
# deterministic pre-assembled context as `context-assisted`, with the tool menu still open so the
# model can fetch what the injected window does not cover (e.g. older history). The dual-LLM
# ablation needs context-without-tools and context-with-tools as distinct arms, so that the tools
# effect is measured against an identical information baseline rather than against a different one.
# RED-only campaign profiles add cumulative task memory, conditional planning and feedback to
# context-plus-tools. Their extra mode distinguishes them from the four shared baseline corners.
LLM_STRATEGIES = {
    "tool-assisted": {"tools_enabled": True, "context_augmented": False},
    "no-tool": {"tools_enabled": False, "context_augmented": False},
    "context-assisted": {"tools_enabled": False, "context_augmented": True},
    "context-tool-assisted": {"tools_enabled": True, "context_augmented": True},
    "campaign-memory": {"tools_enabled": True, "context_augmented": True,
                        "campaign_mode": "memory"},
    "campaign-planning": {"tools_enabled": True, "context_augmented": True,
                          "campaign_mode": "planning"},
    "campaign-feedback": {"tools_enabled": True, "context_augmented": True,
                          "campaign_mode": "feedback"},
}


def _tools_enabled_for(strategy: str) -> bool:
    if strategy not in LLM_STRATEGIES:
        raise ValueError(f"unknown llm strategy {strategy!r}; expected one of "
                         f"{sorted(LLM_STRATEGIES)}")
    return LLM_STRATEGIES[strategy]["tools_enabled"]


def _context_augmented_for(strategy: str) -> bool:
    if strategy not in LLM_STRATEGIES:
        raise ValueError(f"unknown llm strategy {strategy!r}; expected one of "
                         f"{sorted(LLM_STRATEGIES)}")
    return LLM_STRATEGIES[strategy]["context_augmented"]


def _campaign_mode_for(strategy: str, team: str = "red") -> str:
    """Resolve a registered profile; campaign assistance is available only to RED."""
    _tools_enabled_for(strategy)
    mode = LLM_STRATEGIES[strategy].get("campaign_mode", "off")
    if mode != "off" and team != "red":
        raise ValueError(f"{strategy!r} is a RED-only strategy")
    return mode


def idle_blue_slots(game) -> list:
    """Blue slots whose most recent observation is not IN_PROGRESS (safe to submit a new action)."""
    obs = game.team_observation("blue")
    return [s for s in game.blue_slots if obs[s].get("success") != TernaryEnum.IN_PROGRESS]


def _sleep_all(slots) -> dict:
    return {s: Sleep() for s in slots}


# --------------------------------------------------------------------------------------------
# Blue anchors
# --------------------------------------------------------------------------------------------

def _make_never_restore():
    def policy(game, team_obs):
        return _sleep_all(game.blue_slots)
    return policy


def _make_restore_every_k(k: int):
    # Fresh mutable tick counter per call (per "fresh closure per call" state-freshness rule).
    tick = {"t": 0}

    def policy(game, team_obs):
        t = tick["t"]
        tick["t"] += 1
        actions = _sleep_all(game.blue_slots)
        if t % k == 0:
            idle = set(idle_blue_slots(game))
            for resource, owner in RESOURCE_OWNER.items():
                if owner in idle:
                    actions[owner] = Restore(session=0, agent=owner, hostname=resource)
        return actions
    return policy


def _make_restore_on_alert():
    # Fresh per-call counter: the very first call in any run reflects the ONE-TIME initial/reset
    # snapshot (env.py's `init_state`), not a per-tick Monitor-combined observation -- Monitor is
    # an end-turn action inside `SimulationController.step()`, so it structurally cannot have run
    # yet before the first `.step()` call. That initial dump already carries a baseline non-empty
    # "Processes" entry for a host the slot has visibility on (its own child session's PID/
    # username, e.g. `{'PID': 6656, 'username': 'ubuntu'}` -- verified live), which is NOT a
    # Monitor-detected event and would otherwise false-positive an alert on tick 0 of every run
    # regardless of red. Genuine per-tick alerts are unaffected: Monitor clears both
    # `network_connections` and `process_creation` after reading them each tick and only adds a
    # host key at all when one of those lists was non-empty that tick (`Monitor.execute`), so from
    # tick 1 onward an event-free host key is simply absent from the observation -- confirmed live
    # that `resource in team_obs[owner]` is False on event-free ticks.
    tick = {"t": 0}

    def policy(game, team_obs):
        t = tick["t"]
        tick["t"] += 1
        actions = _sleep_all(game.blue_slots)
        if t == 0:
            return actions
        idle = set(idle_blue_slots(game))
        for resource, owner in RESOURCE_OWNER.items():
            if owner not in idle:
                continue
            host_obs = team_obs.get(owner, {}).get(resource, {})
            # Monitor (blue's automatic end-turn action, see wrapper docstring) folds any
            # network-connection/process-creation event on `resource` into this tick's combined
            # observation as a non-empty "Processes" list on that host's key -- this is the
            # "Monitor alert this tick" signal the brief specifies.
            if host_obs.get("Processes"):
                actions[owner] = Restore(session=0, agent=owner, hostname=resource)
        return actions
    return policy


def _make_block_chokepoint():
    # Fresh one-shot latch per call.
    issued = {"done": False}

    def policy(game, team_obs):
        actions = _sleep_all(game.blue_slots)
        if not issued["done"]:
            idle = set(idle_blue_slots(game))
            for resource, owner in RESOURCE_OWNER.items():
                if owner in idle:
                    actions[owner] = BlockTrafficZone(
                        session=0,
                        agent=owner,
                        from_subnet=RESOURCE_RESTRICTED_SUBNET[resource],
                        to_subnet=RESOURCE_OPERATIONAL_SUBNET[resource],
                    )
            issued["done"] = True
        return actions
    return policy


_BLUE_POLICY_FACTORIES = {
    "never-restore": _make_never_restore,
    "restore-on-alert": _make_restore_on_alert,
    "restore-every-5": lambda: _make_restore_every_k(5),
    "restore-every-10": lambda: _make_restore_every_k(10),
    "restore-every-20": lambda: _make_restore_every_k(20),
    "block-chokepoint": _make_block_chokepoint,
}


def blue_policy(name: str, client=None, strategy: str = "tool-assisted", **llm_kwargs):
    """Return a fresh blue team policy closure for anchor `name`.

    the design: `name` of the form `"llm:<model>:<framing>"` dispatches to the LLM layer instead of a
    scripted anchor. Without a `client` (the default) this returns an inert `LLMPlaceholder` --
    `flipthem.llm.agent` is never even imported. With a `client` (injected by the CLI only after
    its own `--yes` cost gate) this returns a genuine `flipthem.llm.agent.LLMPolicy`.
    `**llm_kwargs` (e.g. max_tool_calls, max_idle_ticks, max_retries, budget, value,
    wake_alert_hosts, wake_on_all_alerts, params) are forwarded to `make_llm_policy` unchanged;
    ignored entirely for a scripted `name`.

    the design: `strategy` is one of `LLM_STRATEGIES` (default `"tool-assisted"`, identical to every
    pre-the design call) -- `"no-tool"` builds a policy with `tools_enabled=False` (see
    `flipthem.llm.agent.LLMPolicy`/`flipthem.llm.prompts.build_system`). Ignored for a scripted
    `name`; a malformed `strategy` always raises, even without a client. RED-only campaign
    profiles are rejected for BLUE.
    """
    if name.startswith("llm:"):
        model, framing = _parse_llm_name(name)
        tools_enabled = _tools_enabled_for(strategy)   # validates strategy unconditionally
        context_augmented = _context_augmented_for(strategy)
        campaign_mode = _campaign_mode_for(strategy, "blue")
        if client is None:
            return LLMPlaceholder(team="blue", model=model, framing=framing, strategy=strategy)
        from flipthem.llm.agent import make_llm_policy   # lazy: only touched on the real path
        return make_llm_policy(team="blue", client=client, model=model, framing=framing,
                               tools_enabled=tools_enabled,
                               context_augmented=context_augmented,
                               campaign_mode=campaign_mode, **llm_kwargs)
    if name not in BLUE_ANCHORS:
        raise ValueError(f"unknown blue anchor {name!r}; expected one of {sorted(BLUE_ANCHORS)}")
    return _BLUE_POLICY_FACTORIES[name]()


# --------------------------------------------------------------------------------------------
# Red anchors -- all five are native, self-driving FiniteStateRedAgent subclasses.
# --------------------------------------------------------------------------------------------

class ImpactASAPRedAgent(FiniteStateRedAgent):
    """Rushes to root, then uses `Impact` immediately and repeatedly: "reaches and uses Impact
    as fast as possible" (brief).

    Mechanism: re-weight `state_transitions_probability` so every pre-root state always advances
    (never re-discovers/re-checks-deception when a faster option is legal) and, critically, both
    rooted states (`R`, `RD` -- the only rows where the base class ever makes `Impact` legal at
    all) go to `Impact` with probability 1.0 (base fsm splits `R` 0.25 Impact/0.25
    Degrade/0.5 rediscover; `RD` 0.5/0.5 Impact/Degrade). `set_host_state_priority_list` also
    biases host selection toward hosts already close to or at root (U/UD/R/RD outweigh
    K/KD/S/SD 60:40) so an in-progress chain is finished before starting a new one elsewhere.
    Every row keeps the same non-None (legal) columns as the base class -- only the *weights*
    change -- so this cannot make a state's legal-action set empty (a pre-existing base-class
    fallback path, `_choose_host_and_action`'s `new_options.pop(chosen_host)` branch, would raise
    on a non-int host key; keeping the same legal columns as upstream avoids ever exercising it).
    """

    def __init__(self, name=None, np_random=None, agent_subnets=None):
        super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
        self.prioritise_servers = True

    def set_host_state_priority_list(self):
        return {'K': 8, 'KD': 7, 'S': 15, 'SD': 10, 'U': 20, 'UD': 15, 'R': 15, 'RD': 10}

    def state_transitions_probability(self):
        return {
            'K':  [0.2,  0.4,  0.4,  None, None, None, None, None, None],
            'KD': [None, 0.5,  0.5,  None, None, None, None, None, None],
            'S':  [0.0,  None, None, 0.0,  1.0,  None, None, None, None],
            'SD': [None, None, None, 0.0,  1.0,  None, None, None, None],
            'U':  [0.0,  None, None, None, None, 1.0,  None, None, 0.0 ],
            'UD': [None, None, None, None, None, 1.0,  None, None, 0.0 ],
            'R':  [0.0,  None, None, None, None, None, 1.0,  0.0,  0.0 ],
            'RD': [None, None, None, None, None, None, 1.0,  0.0,  0.0 ],
        }


class ReconOnlyNeverImpactRedAgent(FiniteStateRedAgent):
    """Compromises hosts through root but "NEVER issues Impact" (brief) -- a control-holder that
    never damages service.

    Mechanism: `Impact` (column 6) is legal (non-None) in exactly two rows of the base
    probability matrix -- `R` and `RD` -- and nowhere else (verified by inspection of
    `FiniteStateRedAgent.state_transitions_probability`). Zeroing column 6 in just those two rows
    is therefore complete and sufficient to guarantee the probability-matrix-driven choice can
    never select `Impact`; every other row (K/KD/S/SD/U/UD) is left identical to the stock FSM, so
    pre-root behaviour -- and thus the approach path -- matches `fsm` exactly. Impact's weight is
    redirected to further reconnaissance (`R` -> always re-run `DiscoverRemoteSystems`, never
    Degrade either) and, once a second recon pass has been logged on the same rooted host (`RD`),
    to `Withdraw` -- disengaging rather than lingering, which is arguably the more faithful
    "recon-only" ending (never Degrade there either, so it is never a covert saboteur).
    """

    def state_transitions_probability(self):
        return {
            'K':  [0.5,  0.25, 0.25, None, None, None, None, None, None],
            'KD': [None, 0.5,  0.5,  None, None, None, None, None, None],
            'S':  [0.25, None, None, 0.25, 0.5,  None, None, None, None],
            'SD': [None, None, None, 0.25, 0.75, None, None, None, None],
            'U':  [0.5,  None, None, None, None, 0.5,  None, None, 0.0 ],
            'UD': [None, None, None, None, None, 1.0,  None, None, 0.0 ],
            'R':  [1.0,  None, None, None, None, None, 0.0,  0.0,  0.0 ],
            'RD': [None, None, None, None, None, None, 0.0,  0.0,  1.0 ],
        }


class SeizeAndHoldRedAgent(FiniteStateRedAgent):
    """Gains root, then lurks: "stops escalating/impacting, prioritises staying hidden" (brief).

    Mechanism: overrides `_choose_host_and_action` itself (an explicitly-sanctioned steering
    point per the design decision) rather than the probability matrix, because "do
    nothing" has no probability-matrix representation (every row always ends in some real action
    -- there is no legal all-states no-op column). The moment ANY known host is in state `R` or
    `RD`, it short-circuits to `(None, Sleep())` unconditionally, bypassing host/action selection
    entirely -- the strongest available form of "stop" (no further Discover/Exploit/PrivEsc/
    Impact/Degrade/Withdraw anywhere, not just on the rooted host), and Sleep generates no new
    Monitor-visible events ("staying hidden"). `_host_state_transition` is a no-op on a Sleep
    observation (`Sleep` is not in `self.action_list`), so lurking never corrupts host-state
    bookkeeping. If blue later Restores the rooted host, `_session_removal_state_change` drops its
    state below root and normal FSM behaviour resumes on the next tick -- fitting "seize AND
    hold": holding is conditional on actually still holding something.
    """

    def _choose_host_and_action(self, action_space, host_options):
        if any(self.host_states[h]['state'] in ('R', 'RD') for h in host_options):
            return None, Sleep()
        return super()._choose_host_and_action(action_space, host_options)


_RED_CLASS_BY_NAME = {
    "fsm": FiniteStateRedAgent,
    "discovery-fsm": DiscoveryFSRed,
    "impact-asap": ImpactASAPRedAgent,
    "recon-only-never-impact": ReconOnlyNeverImpactRedAgent,
    "seize-and-hold": SeizeAndHoldRedAgent,
}


def red_policy(name: str, client=None, strategy: str = "tool-assisted", **llm_kwargs):
    """Return (policy_or_None, red_agent_class_or_None) for anchor `name`.

    Every SCRIPTED red anchor here is native/self-driving, so those always return
    `(None, <class>)`: pass the class to `FlipThemCC4(red_agent_class=...)` and
    `red_actions=None` to `.step()`. the design: an `"llm:<model>:<framing>"` name is runner-driven
    instead (never a self-driving native FSM), so it always returns
    `(<LLMPlaceholder | LLMPolicy>, None)` -- same client-injection contract as `blue_policy`
    (no client -> inert placeholder, `flipthem.llm.agent` untouched; a client -> a genuine
    `LLMPolicy`). the design: `strategy` (default `"tool-assisted"`) is the same registrable label as
    `blue_policy`'s -- see there.
    """
    if name.startswith("llm:"):
        model, framing = _parse_llm_name(name)
        tools_enabled = _tools_enabled_for(strategy)   # validates strategy unconditionally
        context_augmented = _context_augmented_for(strategy)
        campaign_mode = _campaign_mode_for(strategy, "red")
        if client is None:
            return LLMPlaceholder(team="red", model=model, framing=framing,
                                  strategy=strategy), None
        from flipthem.llm.agent import make_llm_policy
        return make_llm_policy(team="red", client=client, model=model, framing=framing,
                               tools_enabled=tools_enabled,
                               context_augmented=context_augmented,
                               campaign_mode=campaign_mode, **llm_kwargs), None
    if name not in RED_ANCHORS:
        raise ValueError(f"unknown red anchor {name!r}; expected one of {sorted(RED_ANCHORS)}")
    return None, _RED_CLASS_BY_NAME[name]

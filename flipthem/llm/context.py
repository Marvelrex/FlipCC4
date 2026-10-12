"""`LegalContext` -- the data contract (the design; the design).

The isolation boundary is structural: `build_legal_context` assembles one `LegalContext` per
decision from (a) the wrapper's own legal per-slot observation (`team_obs`, i.e. exactly what
`FlipThemCC4.team_observation(team)` returns), (b) the CALLER's already-filtered, already-stringified
per-slot legal (action, target) pairs (`legal_pairs` -- built by `flipthem.llm.compiler` from
`env.get_action_space(slot)`, per the reviewer's the design: the action space is fetched
separately by the caller, never assumed to live in `team_obs`), and (c) this side's own cumulative
log (`own_log`). `LegalContext` is the ONLY thing the four tools (`flipthem/llm/tools.py`) and the
prompt builders (`flipthem/llm/prompts.py`) ever receive -- never a `CybORG`/`FlipThemCC4`/`state`,
so they structurally cannot reach ground truth.

`defense_state` (blue) is derived from `own_log` alone -- never from raw `team_obs` host content
-- so blue's own log is the single source of truth for what it "knows"; this is what makes the
the design controlled isolation tests hold structurally, not by convention: a blue `LegalContext` has
no dataclass field a compromise value could ever be written into. `holdings` (red) is different
(fix round 2, item 1): red's `sessions`/`root` are read live from `team_obs`'s own per-host
`Sessions` entries each tick (own_log would only ever repeat what red itself SUBMITTED, which is
exactly the bug this fixes -- a failed action must grant nothing, and an evicted session must
drop, neither of which own_log's action names alone can tell you). This is safe ONLY because it is
gated to `team == "red"`: blue never takes this path, and `_red_holdings` only ever reads Sessions
entries whose `agent` is one of RED's own slots.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from CybORG.Shared.Enums import TernaryEnum

RESULT_SUBMITTED = "submitted"
RESULT_IN_PROGRESS = "in_progress"
RESULT_SUCCEEDED = "succeeded"      # obs["success"] == TernaryEnum.TRUE (fix round 2, item 1)
RESULT_FAILED = "failed"            # obs["success"] == TernaryEnum.FALSE -- ended, but NOT succeeded
RESULT_INVALID = "invalid"
RESULT_SESSION_LOST = "session_lost"
RESULT_OBSERVED = "observed"        # the design: unit=action=target=None; carries a tick's legal delta
                                    # ONLY when no lifecycle entry that tick already carried it


@dataclass(frozen=True)
class LogEntry:
    round: int
    unit: str | None                # None only for the the design result="observed" variant
    action: str | None              # None only for the the design result="observed" variant
    target: str | None
    result: str            # "submitted"|"in_progress"|"succeeded"|"failed"|"invalid"|"session_lost"|"observed"
    observed: tuple[str, ...] = ()
    # Exact hostname/IP associations present in RED's own legal observation.  Keeping the pair
    # preserves information that was previously destroyed when a structured observation was
    # flattened into two unrelated strings.  BLUE never populates this field.
    observed_host_ips: tuple[tuple[str, str], ...] = ()
    # BLUE's own Analyse result, separate from team-wide Monitor alert hostnames.
    analyse_evidence: dict | None = None


@dataclass(frozen=True)
class LegalContext:
    team: str                                      # "blue" | "red"
    round: int
    horizon: int
    decidable: dict                                # slot -> tuple[(action, target|None), ...]
    alerts: tuple[str, ...]                         # blue: new Monitor-alert hosts this round; red: ()
    own_log: tuple[LogEntry, ...]
    target_hosts: tuple[str, str]
    holdings: dict                                  # red only; {} for blue
    defense_state: dict                             # blue only; {} for red
    # Optional RED campaign facts, gated at construction. Neither baseline prompts nor baseline
    # holdings consume these; session ownership comes solely from RED's own observation.
    red_sessions_by_unit: dict = field(default_factory=dict)


def build_legal_context(team: str, tick: int, horizon: int, team_obs: Mapping,
                         legal_pairs: Mapping, own_log: Sequence[LogEntry],
                         target_hosts: tuple, *,
                         include_red_sessions_by_unit: bool = False) -> LegalContext:
    """Assemble one `LegalContext`.

    `legal_pairs` is the caller's already-filtered, already-stringified per-slot action space
    (the design) -- `dict[slot, tuple[(action_name, target_str | None), ...]]`, built by
    `flipthem.llm.compiler.build_slot_action_index` from `env.get_action_space(slot)` and NEVER
    computed in here (this module never imports the compiler or sees a raw CybORG action space).
    `team_obs` is used for: whether a slot is currently busy (`success == IN_PROGRESS`, mirroring
    `FlipThemCC4._in_progress`); blue's FULL set of Monitor-alerted hosts this round (fix round 2,
    item 6 -- restored to the complete legal signal; a separate, narrower subset drives the wake
    trigger in agent.py, not this function); and, for red ONLY (fix round 2, item 1), the live
    per-host Sessions used to derive `holdings.sessions`/`holdings.root`.
    """
    own_log = tuple(own_log)
    decidable = {}
    for slot, pairs in legal_pairs.items():
        obs = team_obs.get(slot, {}) if team_obs else {}
        busy = obs.get("success") == TernaryEnum.IN_PROGRESS
        pairs = tuple(pairs)
        has_non_sleep = any(action != "Sleep" for action, _ in pairs)
        decidable[slot] = pairs if (not busy and has_non_sleep) else ()

    alerts = extract_alerts(team_obs, tick) if team == "blue" else ()
    holdings = _red_holdings(team_obs, own_log) if team == "red" else {}
    defense_state = _blue_defense_state(own_log, target_hosts) if team == "blue" else {}

    return LegalContext(
        team=team, round=tick, horizon=horizon, decidable=decidable, alerts=alerts,
        own_log=own_log, target_hosts=tuple(target_hosts), holdings=holdings,
        defense_state=defense_state,
        red_sessions_by_unit=(_red_sessions_by_unit(team_obs)
                              if team == "red" and include_red_sessions_by_unit else {}),
    )


def extract_alerts(team_obs: Mapping, tick: int) -> tuple:
    """The FULL set of new Monitor-alert hosts this round  -- every host,
    not narrowed to any subset. This is the OBSERVATION: what blue legally received this round
    must reach ctx.alerts/history/the prompt complete, never silently deleted. An earlier round
    (C2) narrowed this function itself to just the 2 target hosts to fix a real problem (blue was
    re-woken almost every tick on routine green activity unrelated to either target) -- but that
    conflated two different concerns: WHAT blue observes vs. WHAT triggers an immediate
    re-decision. The fix keeps this function complete and moves the narrowing to a separate,
    explicit step (`agent.py`'s `LLMPolicy.wake_alert_hosts` filter) that drives ONLY the wake
    gate's `new_event`, never `ctx.alerts` itself. A real restriction on what blue SEES would have
    to be a deliberately labelled experimental condition, not a silent default baked in here.

    Tick 0 is skipped, mirroring `anchors.py::_make_restore_on_alert`'s documented reasoning
    exactly: the very first call in any run reflects the ONE-TIME initial/reset snapshot (Monitor
    is an end-turn action inside `SimulationController.step()`, so it structurally cannot have run
    yet before the first `.step()` call), which already carries a baseline non-empty 'Processes'
    entry for a host the slot has visibility on (its own child session's PID/username) -- NOT a
    Monitor-detected event. Genuine per-tick alerts from tick 1 onward are unaffected (Monitor
    clears both lists it reads every tick, per that same docstring)."""
    if tick == 0:
        return ()
    alerted = []
    for _slot, obs in (team_obs or {}).items():
        if not isinstance(obs, Mapping):
            continue
        for host, val in obs.items():
            if host in ("success", "action"):
                continue
            if isinstance(val, Mapping) and val.get("Processes") and host not in alerted:
                alerted.append(host)
    return tuple(alerted)


def _blue_defense_state(own_log: tuple, target_hosts: tuple) -> dict:
    state = {h: {"analysed": False, "evidence": None, "blocked": False} for h in target_hosts}
    for e in own_log:
        if e.result != RESULT_SUCCEEDED or e.target not in state:
            continue
        if e.action == "Analyse":
            state[e.target]["analysed"] = True
            state[e.target]["evidence"] = e.analyse_evidence
            state[e.target]["evidence_round"] = e.round
        elif e.action == "Restore":
            # a fresh reimage: prior evidence no longer describes the current host state
            state[e.target]["analysed"] = False
            state[e.target]["evidence"] = None
            state[e.target].pop("evidence_round", None)
        elif e.action == "BlockTrafficZone":
            state[e.target]["blocked"] = True
        elif e.action == "AllowTrafficZone":
            state[e.target]["blocked"] = False
    return state


_PRIVILEGED_USERNAMES = ("root", "system")


def _red_sessions_by_unit(team_obs: Mapping) -> dict:
    """Own-team sessions, attributed to their observed owner rather than the reporting slot."""
    own_slots = set(team_obs or {})
    by_unit = {slot: {"sessions": set(), "root": set()} for slot in own_slots}
    for obs in (team_obs or {}).values():
        if not isinstance(obs, Mapping):
            continue
        for host, val in obs.items():
            if host in ("success", "action") or not isinstance(val, Mapping):
                continue
            for sess in val.get("Sessions") or ():
                if not isinstance(sess, Mapping) or sess.get("agent") not in own_slots:
                    continue
                facts = by_unit[sess["agent"]]
                facts["sessions"].add(host)
                if str(sess.get("username", "")).lower() in _PRIVILEGED_USERNAMES:
                    facts["root"].add(host)
    return {slot: {kind: sorted(hosts) for kind, hosts in facts.items()}
            for slot, facts in sorted(by_unit.items())}


def _red_holdings(team_obs: Mapping, own_log: tuple) -> dict:
    """Fix round 2, item 1: `sessions`/`root` are read from the CURRENT tick's live obs -- never
    replayed from own_log action names -- so a FAILED action grants nothing, an evicted session
    (blue Restored it) drops with no log bookkeeping needed, and the initial foothold (present in
    the very first obs, before any action was ever submitted) is included from tick 0.

    Isolation invariant: this reads ONLY team_obs's per-host 'Sessions' list, gated to entries
    whose 'agent' is one of red's OWN slots (`agent in team_obs`) -- never blue's or green's. This
    function is called only when `team == "red"` (see `build_legal_context`); it is never reached
    for blue, and blue's own holdings stays the empty-dict default."""
    own_slots = set(team_obs or {})
    sessions, root = set(), set()
    for _slot, obs in (team_obs or {}).items():
        if not isinstance(obs, Mapping):
            continue
        for host, val in obs.items():
            if host in ("success", "action") or not isinstance(val, Mapping):
                continue
            for sess in (val.get("Sessions") or ()):
                if not isinstance(sess, Mapping) or sess.get("agent") not in own_slots:
                    continue
                sessions.add(host)
                if str(sess.get("username", "")).lower() in _PRIVILEGED_USERNAMES:
                    root.add(host)

    return {
        "sessions": sorted(sessions),
        "root": sorted(root),
        "discovered": _red_discovered(own_log),
    }


def _looks_like_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except (ValueError, TypeError):
        return False


def _red_discovered(own_log: tuple) -> dict:
    """the design: reconstructed from the PERSISTED own_log `.observed` stream (replayable from the
    FULL log), never a single tick's `legal_pairs` (frozen per-tick, lost the instant that tick
    passes) and never `sorted(root)` (that conflates "known" with "currently rooted" -- a
    discovered-but-not-yet-rooted host must show up here without being in `root`).

    `subnets` still comes from `DiscoverRemoteSystems`'s own succeeded `.target` -- a subnet
    string is never itself an obs dict key, so it can never appear in `.observed`. `ips`/`hosts`
    are the cumulative UNION of every log entry's `.observed` tuple (this naturally covers BOTH
    ordinary lifecycle entries carrying a delta AND the design's dedicated `result="observed"` rows,
    with no double-counting risk: exactly one entry ever carries a given tick's delta), split by
    shape -- a real IPv4 address is an ip, anything else (a hostname key) is a host. RED-side only
    (`build_legal_context` never calls this for blue)."""
    subnets = set()
    for e in own_log:
        if e.result == RESULT_SUCCEEDED and e.action == "DiscoverRemoteSystems" and e.target:
            subnets.add(e.target)

    ips, hosts = set(), set()
    host_to_ips = {}
    for e in own_log:
        for item in e.observed:
            (ips if _looks_like_ip(item) else hosts).add(item)
        for host, ip in e.observed_host_ips:
            if not isinstance(host, str) or not isinstance(ip, str):
                continue
            try:
                ipaddress.ip_address(ip)
            except ValueError:
                continue
            hosts.add(host)
            ips.add(ip)
            host_to_ips.setdefault(host, set()).add(ip)

    return {
        "subnets": sorted(subnets),
        "ips": sorted(ips),
        "hosts": sorted(hosts),
        "host_to_ips": {host: sorted(values) for host, values in sorted(host_to_ips.items())},
    }

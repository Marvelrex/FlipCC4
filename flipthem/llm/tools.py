"""The four read-only, own-side-only, non-simulating tools.

Each takes a `LegalContext` (or, for `action_manual`, nothing at all); none ever takes an env.
`check_plan` is a pure membership test against `ctx.decidable[slot]` -- so "no env clone / no
rollout" is structural: there is no env in scope to clone, and nothing here calls `.step`.

``history`` deliberately exposes a compact model-facing view rather than the raw audit rows.
The raw ``own_log`` retains every submitted/in-progress/terminal lifecycle row for replay and
diagnosis.  The compact view folds those rows into one action attempt and unions the same tick's
repeated team-wide observation into one observation event.  A failure remains visible until an
identical same-tick retry supersedes it in the model view; the raw audit retains both attempts.

RED's current-state view also reports active consecutive-failure counts keyed by the exact
``(unit, action, target)`` tuple.  These are observations only: they carry no recommendation,
probability, or prediction about what the agent should do next.
"""
from __future__ import annotations

from typing import Sequence

from flipthem.llm.context import (
    LegalContext, RESULT_SUBMITTED, RESULT_IN_PROGRESS, RESULT_SUCCEEDED, RESULT_FAILED,
    RESULT_SESSION_LOST, RESULT_OBSERVED,
)

HISTORY_DEFAULT_RECENT = 20

# Pure, static per-action rules (design the design's action_manual). Durations/costs mirror the
# frozen mechanism layer (flipthem/billing.py: Restore k=1/duration=5, InvalidAction k=0.1, else
# free) -- these numbers are declared game rules, never a live/native lookup, and carry no FT
# reward/U_B/U_R numerics (offline reward boundary, design the design).
# Durations verified live this session against each native class's own .duration (fix round 2,
# item 3 -- a parameterised test in tests/test_llm_tools.py compares every one of these against a
# freshly-built real instance, so this table can never silently drift from CybORG's own values
# again). Cost mirrors the frozen mechanism layer (flipthem/billing.py: Restore k=1,
# InvalidAction k=0.1, else free) -- these are declared game rules, never a live/native lookup,
# and carry no FT reward/U_B/U_R numerics (offline reward boundary, design the design).
_ACTION_MANUAL = {
    "Sleep": {"duration": 1, "cost": 0.0, "target": None,
              "description": "No-op; costs nothing; never billed."},
    "Restore": {"duration": 5, "cost": 1.0, "target": "hostname",
                "description": "Re-image a host; the only way to evict a rooted attacker."},
    "Analyse": {"duration": 2, "cost": 0.0, "target": "hostname",
                "description": "Investigate a host; returns evidence, not proof of compromise."},
    "Remove": {"duration": 3, "cost": 0.0, "target": "hostname",
               "description": "Kill a non-root foothold on a host; does not remove root."},
    "BlockTrafficZone": {"duration": 1, "cost": 0.0, "target": "hostname",
                          "description": "Cut a target's approach path (reversible)."},
    "AllowTrafficZone": {"duration": 1, "cost": 0.0, "target": "hostname",
                          "description": "Re-open a target's approach path (reversible)."},
    "DiscoverRemoteSystems": {"duration": 1, "cost": 0.0, "target": "subnet",
                              "description": "Discover hosts on an already-discovered subnet."},
    "AggressiveServiceDiscovery": {"duration": 1, "cost": 0.0, "target": "ip_address",
                                   "description": "Fast, noisy service scan of a known IP."},
    "StealthServiceDiscovery": {"duration": 3, "cost": 0.0, "target": "ip_address",
                                "description": "Slow, quieter service scan of a known IP."},
    "ExploitRemoteService": {"duration": 4, "cost": 0.0, "target": "ip_address",
                             "description": "Exploit a discovered service for a foothold."},
    "PrivilegeEscalate": {"duration": 2, "cost": 0.0, "target": "hostname",
                          "description": "Escalate to root on a host you hold a foothold on."},
    "Withdraw": {"duration": 1, "cost": 0.0, "target": "ip_address+hostname",
                "description": "Abandon a foothold/root on a host."},
    "InvalidAction": {"duration": 0, "cost": 0.1, "target": None,
                      "description": "An illegal/invalid choice; always costs 0.1."},
}


def action_manual(action: str | None = None) -> dict:
    """Pure static rules for one action, or every action when `action is None`. Takes no ctx."""
    if action is None:
        return {"actions": {k: dict(v) for k, v in _ACTION_MANUAL.items()}}
    info = _ACTION_MANUAL.get(action)
    if info is None:
        return {"error": f"unknown action {action!r}"}
    return dict(info)


def _history_limit(last_n):
    if last_n == "all":
        return None
    if last_n is None:
        return HISTORY_DEFAULT_RECENT
    try:
        n = int(last_n)
    except (TypeError, ValueError):
        return {"error": f"last_n must be a non-negative int, None, or 'all'; got {last_n!r}"}
    if n < 0:
        return {"error": f"last_n must be non-negative; got {n}"}
    return n


def _compact_history_events(log) -> list[dict]:
    """Fold raw lifecycle rows into action attempts plus once-per-round observations.

    A unit can have only one pending native action, so its submitted row opens an attempt and its
    later in-progress/terminal rows update that attempt.  The private sort fields preserve the
    order of the most recent update: a Restore that completes now is recent even if it started
    five ticks ago.  Orphan lifecycle rows remain visible instead of being silently discarded.
    """
    events = []
    active = {}
    observations = {}

    def new_action(entry, sequence):
        item = {
            "type": "action",
            "unit": entry.unit,
            "action": entry.action,
            "target": entry.target,
            "start": entry.round,
            "result": entry.result,
            "_round": entry.round,
            "_sequence": sequence,
        }
        events.append(item)
        return item

    for sequence, entry in enumerate(log):
        if entry.observed:
            observations.setdefault(entry.round, {"values": set(), "sequence": sequence})
            observations[entry.round]["values"].update(entry.observed)
            observations[entry.round]["sequence"] = sequence

        if entry.result == RESULT_OBSERVED or entry.action is None:
            continue
        key = entry.unit
        current = active.get(key)
        matches = (current is not None and current["action"] == entry.action
                   and current["target"] == entry.target)

        if entry.result == RESULT_SUBMITTED:
            current = new_action(entry, sequence)
            active[key] = current
            continue

        if not matches:
            current = new_action(entry, sequence)
            if entry.result == RESULT_IN_PROGRESS:
                active[key] = current

        current["result"] = entry.result
        if entry.analyse_evidence is not None:
            current["evidence"] = entry.analyse_evidence
        current["_round"] = entry.round
        current["_sequence"] = sequence
        if entry.result == RESULT_IN_PROGRESS:
            current["last"] = entry.round
        else:
            current["end"] = entry.round
            current.pop("last", None)
            if active.get(key) is current:
                del active[key]

    # Once the policy has retried the exact same choice in the same tick that the prior attempt
    # failed, the superseded failure is no longer decision-relevant. Keep only the new attempt in
    # the model view. A failure with no retry remains visible, and the raw audit remains complete.
    superseded = set()
    latest_by_choice = {}
    for item in events:
        choice = (item["unit"], item["action"], item["target"])
        previous = latest_by_choice.get(choice)
        if (previous is not None
                and previous["result"] in (RESULT_FAILED, RESULT_SESSION_LOST)
                and previous.get("end") == item.get("start")):
            superseded.add(id(previous))
        latest_by_choice[choice] = item
    events = [item for item in events if id(item) not in superseded]

    for round_, data in observations.items():
        events.append({
            "type": "observation",
            "round": round_,
            "observed": sorted(data["values"]),
            "_round": round_,
            "_sequence": data["sequence"],
        })

    events.sort(key=lambda item: (item["_round"], item["_sequence"],
                                  1 if item["type"] == "observation" else 0))
    for item in events:
        item.pop("_round", None)
        item.pop("_sequence", None)
    return events


def history(ctx: LegalContext, last_n: int | None = None) -> dict:
    """Return a cumulative summary and compact recent episode events.

    ``last_n`` counts compact events, not raw lifecycle rows.  Omitting it returns the most recent
    20; explicit ``"all"`` returns every compact event.  Invalid values report a tool error.

    Item 4 (fix round 2): summary reports SEPARATE counters instead of conflating a multi-round
    action's own lifecycle rows with "how many actions/rounds happened" -- one Restore's real
    6-row lifecycle (submitted -> in_progress x4 -> succeeded) is ONE action over 6 real ticks,
    never a 6-way tally:
      - env_ticks: actual environment ticks elapsed (from ctx.round, the current tick -- not the
        own-log's row count, which can exceed or undercount ticks depending on how many slots/
        rows landed that tick).
      - submissions: non-Sleep decisions actually issued (RESULT_SUBMITTED rows -- Sleep is never
        logged as a submission at all, see agent.py's _record_submission).
      - completions: actions whose lifecycle ENDED this log (succeeded, failed, or session_lost --
        never in_progress/submitted, which have not ended yet).
      - successes: the subset of completions whose obs["success"] was TRUE.
    """
    log = ctx.own_log
    submissions = sum(1 for e in log if e.result == RESULT_SUBMITTED)
    completions = sum(1 for e in log if e.result in (RESULT_SUCCEEDED, RESULT_FAILED, RESULT_SESSION_LOST))
    successes = sum(1 for e in log if e.result == RESULT_SUCCEEDED)
    targets_discovered = set()
    restores_done = 0
    observed_rounds = set()
    for e in log:
        if e.observed:
            targets_discovered.update(e.observed)
            observed_rounds.add(e.round)
        if e.action == "Restore" and e.result == RESULT_SUCCEEDED:
            restores_done += 1
    summary = {
        "env_ticks": ctx.round + 1,
        "submissions": submissions,
        "completions": completions,
        "successes": successes,
        "targets_discovered": sorted(targets_discovered),
        "restores_done": restores_done,
        # Historically named alerts_seen.  Count each observation-bearing round once: the raw
        # audit log can attach the same team-wide observation to several concurrent lifecycles.
        "alerts_seen": len(observed_rounds),
        "sessions_held": len(ctx.holdings.get("sessions", ())) if ctx.holdings else 0,
        "roots_held": len(ctx.holdings.get("root", ())) if ctx.holdings else 0,
    }
    limit = _history_limit(last_n)
    if isinstance(limit, dict):
        return limit
    compact = _compact_history_events(log)
    recent = compact if limit is None else (compact[-limit:] if limit else [])
    return {"summary": summary, "recent": recent}


def _consecutive_failure_streaks(log) -> list[dict]:
    """Active failure streaks for exact ``(unit, action, target)`` choices.

    Lifecycle rows that have not finished do not change a streak. A success resets only the
    matching tuple; failures of other choices neither extend nor reset it. InvalidAction rows
    cannot be attributed to the model's original action/target after compilation, so they are
    deliberately excluded rather than assigned to a false key.
    """
    streaks = {}
    # Reuse the model-facing lifecycle compaction so an identical retry submitted in the same
    # tick suppresses the just-failed attempt here exactly as it does in history(). This prevents
    # the new summary from reintroducing retry noise that the recent-history view removed.
    for entry in _compact_history_events(log):
        if entry.get("type") != "action":
            continue
        key = (entry["unit"], entry["action"], entry["target"])
        if entry["result"] == RESULT_SUCCEEDED:
            streaks.pop(key, None)
        elif entry["result"] in (RESULT_FAILED, RESULT_SESSION_LOST):
            prior = streaks.get(key)
            streaks[key] = {
                "unit": entry["unit"],
                "action": entry["action"],
                "target": entry["target"],
                "count": (prior["count"] + 1) if prior else 1,
                "last_failure_round": entry.get("end", entry["start"]),
            }
    return sorted(streaks.values(), key=lambda item: (
        -item["count"], -item["last_failure_round"], item["unit"], item["action"],
        "" if item["target"] is None else item["target"],
    ))


def known_targets(ctx: LegalContext) -> dict:
    """Public objective names plus own-side observations; never opponent control or true state."""
    if ctx.team == "red":
        holdings = dict(ctx.holdings)
        sessions = set(holdings.get("sessions") or ())
        root = set(holdings.get("root") or ())
        discovered = holdings.get("discovered") or {}
        discovered_hosts = set(discovered.get("hosts") or ())
        host_to_ips = discovered.get("host_to_ips") or {}
        progress = {}
        for target in ctx.target_hosts:
            observed_ips = list(host_to_ips.get(target) or ())
            progress[target] = {
                "observed_by_reconnaissance": target in discovered_hosts,
                "observed_ips": observed_ips,
                "session_on_exact_target": target in sessions,
                "root_on_exact_target": target in root,
            }
        return {"targets": list(ctx.target_hosts), "target_progress": progress,
                "holdings": holdings,
                "consecutive_failures": _consecutive_failure_streaks(ctx.own_log)}
    return {"targets": list(ctx.target_hosts),
            "defense_state": {h: dict(v) for h, v in ctx.defense_state.items()}}


def check_plan(ctx: LegalContext, slot: str, pairs: Sequence[tuple]) -> dict:
    """Membership test of each (action, target) pair against `ctx.decidable[slot]` ONLY --
    never "legal in any slot" (the design). No env exists in scope; returns
    only current-menu membership, never outcome/next_state/reward."""
    listed_set = set(ctx.decidable.get(slot, ()))
    results = []
    for action, target in pairs:
        listed_now = (action, target) in listed_set
        reason = ("listed for this unit in the current compiler menu" if listed_now
                  else "not listed for this unit in the current compiler menu")
        results.append({"action": action, "target": target, "listed_now": listed_now,
                        "outcome_unknown": True, "reason": reason})
    return {"results": results}


# a later revision final preflight (wire-protocol item 1): the OpenAI function-calling JSON schema for
# each of the four tools above, defined HERE (not in real_client.py) so it can never drift from
# _dispatch_tool's own arg names (agent.py) -- "last_n"/"action"/no-args/"unit"+"pairs" match
# exactly what _dispatch_tool reads off `args`. RealClient sends this list verbatim as the
# request's "tools" field for the tool-assisted strategy (None/omitted for no-tool).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "history",
            "description": ("Your own past actions/results, the alerts/events you've legally "
                            "observed, and a cumulative summary of your campaign so far."),
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {
                        "description": ("How many recent compact events to return. One action "
                                        "attempt spans submitted/in-progress/final lifecycle rows; "
                                        "one observation event represents a round. Use a non-negative "
                                        "integer or 'all'; omit for the latest 20."),
                        "anyOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "string", "enum": ["all"]},
                        ],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_manual",
            "description": ("The exact rules (duration, cost, preconditions) for one action, "
                            "or every action if none is given."),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action name, e.g. 'Restore'. Omit for every action.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "known_targets",
            "description": ("The public scoring-target hostnames plus your own discovered "
                             "subnets/IPs/hosts and footholds/root (red), or your Analyse "
                             "evidence/active blocks (blue). Takes no arguments."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_plan",
            "description": ("Reports whether each proposed (action, target) pair is listed for "
                             "one unit in the current compiler menu. outcome_unknown is true for "
                             "every result."),
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "description": ("The decidable unit id to check against; omit to use "
                                        "your only decidable unit."),
                    },
                    "pairs": {
                        "type": "array",
                        "description": "List of [action, target] pairs to check.",
                        "items": {
                            "type": "array",
                            "items": {"type": ["string", "null"]},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "required": ["pairs"],
            },
        },
    },
]

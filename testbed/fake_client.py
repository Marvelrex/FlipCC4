"""Deterministic offline model clients.

A model client is any object with ``next(system, user, transcript, force_final) -> dict``; see
``flipthem/llm/backend.py``. The clients here never open a connection and need no credential.
"""
from __future__ import annotations

import hashlib
import json
import re

# One line per decidable unit in the context-augmented user prompt, e.g.
#   - blue_agent_1: Restore|Analyse->[host_a,host_b]; Sleep->[null]
_UNIT_LINE = re.compile(r"^- ((?:blue|red)_agent_\d+): (.*)$", re.M)
_PREF = {"blue": ("Restore", "Remove", "Analyse", "BlockTrafficZone"),
         "red": ("PrivilegeEscalate", "ExploitRemoteService", "AggressiveServiceDiscovery",
                 "DiscoverRemoteSystems", "Withdraw")}
TEAMS = ("blue", "red")


def parse_legal_options(user: str) -> dict:
    """{unit: {action_name: [targets]}} parsed from the rendered legal-options block."""
    out = {}
    for unit, opts in _UNIT_LINE.findall(user):
        legal = {}
        for part in opts.split("; "):
            if "->" not in part:
                continue
            names, targets = part.split("->", 1)
            ts = [t for t in targets.strip("[]").split(",") if t and t != "null"]
            for n in names.split("|"):
                legal[n] = ts
        out[unit] = legal
    return out


def final(decisions, wake=None, usage_cost=0.0) -> dict:
    """A well-formed 'final' response carrying `decisions`."""
    return {"type": "final", "text": json.dumps({"decisions": decisions, "wake": wake}),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "cost": usage_cost},
            "returned_provider": "offline-stub", "returned_model": "offline-stub"}


class OfflineStubClient:
    """A pure function of the prompt text: identical states give identical answers.

    Chooses among the LEGAL options listed in the prompt (a third of the units Sleep, chosen by
    hash), preferring the team's recovery/attack actions. Requires a context-augmented profile,
    because only those prompts carry the legal-options block."""

    def __init__(self, team: str, *, usage_cost: float = 0.0, label: str = "offline-stub"):
        if team not in TEAMS:
            raise ValueError(f"team must be one of {TEAMS}, got {team!r}")
        self.team = team
        self.usage_cost = float(usage_cost)
        self.label = label
        self.provider = "stub"
        self.tools = None
        self.upstream_provider = None
        self.calls = 0

    def next(self, system, user, transcript, force_final):
        self.calls += 1
        decisions = []
        for unit, legal in parse_legal_options(user).items():
            h = int(hashlib.sha256((user + unit).encode()).hexdigest(), 16)
            if h % 3 == 0:
                decisions.append({"unit": unit, "action": "Sleep", "target": None})
                continue
            for n in _PREF[self.team]:
                if legal.get(n):
                    ts = legal[n]
                    if self.team == "blue":
                        ts = [t for t in ts if "operational_zone" in t and "server_host_0" in t] or ts
                    decisions.append({"unit": unit, "action": n, "target": ts[h % len(ts)]})
                    break
        return final(decisions, usage_cost=self.usage_cost)


class ScriptedClient:
    """Replays a fixed list of responses; the last one repeats when `repeat_last` is set,
    otherwise an exhausted script raises."""

    def __init__(self, responses, *, repeat_last: bool = True):
        self.responses = list(responses)
        self.repeat_last = repeat_last
        self.provider = "stub"
        self.tools = None
        self.calls = 0

    def next(self, system, user, transcript, force_final):
        self.calls += 1
        if self.calls <= len(self.responses):
            return self.responses[self.calls - 1]
        if self.repeat_last and self.responses:
            return self.responses[-1]
        raise AssertionError(f"scripted client exhausted after {len(self.responses)} responses")


def stub_factory(client_cfg: dict | None = None):
    """A client factory for the configuration adapter: factory(team, token, model) -> client."""
    client_cfg = client_cfg or {}
    kind = client_cfg.get("kind", "offline-stub")
    if kind != "offline-stub":
        raise ValueError(f"unknown client kind {kind!r}; this testbed ships 'offline-stub' only")
    usage_cost = float(client_cfg.get("usage_cost_usd", 0.0))

    def factory(team: str, token: str, model: str):
        return OfflineStubClient(team, usage_cost=usage_cost, label=f"{token}:{model}")
    return factory

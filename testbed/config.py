"""Configuration adapter: one small JSON document -> the policies and environment of one match.

Shape (see testbed/configs/smoke.json)::

    {"schema": "cc4-testbed-config-v1", "seed": 42,
     "settings": {"steps": 40, "mode": "ft", "max_tool_calls": 4, "max_idle_ticks": 10,
                  "max_retries": 0, "budget_limit": 400, "max_cost_usd": 5.0, "framing": "neutral"},
     "client": {"kind": "offline-stub", "usage_cost_usd": 0.001},
     "blue": {"kind": "switch", "start": {"id": "B0"}, "other": {"id": "B1"},
              "arm": "deadline", "event_enabled": false, "deadline_tick": 20},
     "red": {"kind": "llm", "id": "R0"}}

A side is one of
  * ``llm``      -- a model-driven policy; ``id`` selects a registered assistance profile
                    (B0/B1/R0/R1, see CONFIGURATION_IDS) or ``strategy`` names one directly;
  * ``switch``   -- two ``llm`` components of the same team sharing ONE request budget, switched
                    once by a frozen selector (``arm`` ``deadline`` or ``event``);
  * ``scripted`` -- a constrained scripted rule on the legal interface (S0/S1/S2 blue, A0 red);
  * ``native``   -- an upstream CybORG anchor (e.g. red ``fsm``), self-driving inside the env.

Model clients are always injected through ``client_factory(team, token, model)``; nothing here
builds a network client.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from flipthem.anchors import LLM_STRATEGIES, BLUE_ANCHORS, RED_ANCHORS
from flipthem.execution_log import ExecutionRecorder
from flipthem.llm.backend import RequestBudget
from flipthem.llm.eval_runner import LLMSpec, ScriptedSpec, SwitchSpec, _build_blue, _build_red
from flipthem.scripted_policies import SCRIPTED_BLUE, SCRIPTED_RED
from flipthem.switching_policy import Selector
from flipthem.wrapper import FlipThemCC4

CONFIG_SCHEMA = "cc4-testbed-config-v1"
# Configuration ids used in the write-up -> registered assistance profiles (flipthem.anchors).
CONFIGURATION_IDS = {
    "B0": "context-assisted",          # Blue without tools
    "B1": "context-tool-assisted",     # Blue with the four read-only tools
    "R0": "context-tool-assisted",     # Red with tools
    "R1": "campaign-feedback",         # Red with tools and bounded campaign memory/feedback
}
DEFAULT_SETTINGS = {
    "steps": 40, "mode": "ft", "max_tool_calls": 4, "max_idle_ticks": 10, "max_retries": 0,
    "max_final_repairs": 0, "budget_limit": None, "price_input": None, "price_output": None,
    "max_cost_usd": None, "framing": "neutral", "value": 1.0,
}


def load_config(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA!r}")
    if not isinstance(cfg.get("seed"), int):
        raise ValueError("config.seed must be an integer")
    for team in ("blue", "red"):
        if not isinstance(cfg.get(team), dict) or "kind" not in cfg[team]:
            raise ValueError(f"config.{team} must be an object with a 'kind'")
    st = settings_of(cfg)
    if st["mode"] != "ft":
        raise ValueError("this testbed accepts mode 'ft' only")
    if int(st["steps"]) < 2:
        raise ValueError("settings.steps must be at least 2")


def settings_of(cfg: dict) -> dict:
    return {**DEFAULT_SETTINGS, **cfg.get("settings", {})}


def strategy_of(side_cfg: dict) -> str:
    if "strategy" in side_cfg:
        strategy = side_cfg["strategy"]
    elif side_cfg.get("id") in CONFIGURATION_IDS:
        strategy = CONFIGURATION_IDS[side_cfg["id"]]
    else:
        raise ValueError(f"llm side needs an 'id' in {sorted(CONFIGURATION_IDS)} or a 'strategy'")
    if strategy not in LLM_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    return strategy


def _llm_spec(team, side_cfg, st, budget, client_factory, token, clients) -> LLMSpec:
    model = side_cfg.get("model", f"offline-stub-{team}")
    client = client_factory(team, token, model)
    clients[token] = client
    return LLMSpec(client=client, model=model, framing=st["framing"],
                   max_tool_calls=int(st["max_tool_calls"]), max_idle_ticks=int(st["max_idle_ticks"]),
                   budget=budget, value=float(st["value"]), max_retries=int(st["max_retries"]),
                   max_final_repairs=int(st["max_final_repairs"]), strategy=strategy_of(side_cfg),
                   price_input=st["price_input"], price_output=st["price_output"],
                   max_cost_usd=st["max_cost_usd"])


def build_side(team: str, side_cfg: dict, st: dict, client_factory, clients: dict, *,
               deadline_override=None):
    kind = side_cfg["kind"]
    # Client handle tokens match testbed.fork.HANDLE_TOKENS: the fork never serialises a client and
    # rebinds each token to a fresh client per branch.
    if kind == "llm":
        budget = RequestBudget(limit=st["budget_limit"])
        return _llm_spec(team, side_cfg, st, budget, client_factory,
                         "red" if team == "red" else "blue:B0", clients)
    if kind == "switch":
        budget = RequestBudget(limit=st["budget_limit"])   # ONE budget for both components
        start = _llm_spec(team, side_cfg["start"], st, budget, client_factory,
                          "red" if team == "red" else "blue:B0", clients)
        other = _llm_spec(team, side_cfg["other"], st, budget, client_factory,
                          "red:other" if team == "red" else "blue:B1", clients)
        arm = side_cfg.get("arm", "deadline")
        deadline = side_cfg.get("deadline_tick", 100) if deadline_override is None else deadline_override
        selector = Selector(arm=arm, event_enabled=bool(side_cfg.get("event_enabled", arm == "event")),
                            deadline_tick=int(deadline))
        return SwitchSpec(start=start, other=other, selector=selector)
    if kind == "scripted":
        allowed = SCRIPTED_BLUE if team == "blue" else SCRIPTED_RED
        if side_cfg["name"] not in allowed:
            raise ValueError(f"scripted {team} rule must be one of {allowed}")
        return ScriptedSpec(side_cfg["name"])
    if kind == "native":
        allowed = BLUE_ANCHORS if team == "blue" else RED_ANCHORS
        if side_cfg["name"] not in allowed:
            raise ValueError(f"native {team} anchor must be one of {sorted(allowed)}")
        return side_cfg["name"]
    raise ValueError(f"unknown side kind {kind!r}")


def build_specs(cfg: dict, client_factory, *, deadline_override=None) -> SimpleNamespace:
    validate_config(cfg)
    st = settings_of(cfg)
    clients = {}
    blue = build_side("blue", cfg["blue"], st, client_factory, clients, deadline_override=deadline_override)
    red = build_side("red", cfg["red"], st, client_factory, clients, deadline_override=deadline_override)
    return SimpleNamespace(blue=blue, red=red, clients=clients, settings=st)


def build_match(cfg: dict, client_factory, *, deadline_override=None, episode_id=None) -> SimpleNamespace:
    """Policies + a reset CC4 environment + a passive execution recorder, ready to drive."""
    specs = build_specs(cfg, client_factory, deadline_override=deadline_override)
    st = specs.settings
    pol_b = _build_blue(specs.blue)
    pol_r, red_cls = _build_red(specs.red)
    game = FlipThemCC4(seed=cfg["seed"], steps=int(st["steps"]), mode=st["mode"], red_agent_class=red_cls)
    game.reset()
    recorder = ExecutionRecorder(episode_id or f"seed{cfg['seed']}")
    game.step_observer = recorder.on_step
    return SimpleNamespace(game=game, pol_b=pol_b, pol_r=pol_r, red_cls=red_cls, recorder=recorder,
                           trace=[], clients=specs.clients, specs=specs, settings=st, seed=cfg["seed"])


def variant(cfg: dict, **changes) -> dict:
    """A deep copy of `cfg` with top-level keys replaced (settings are merged)."""
    out = copy.deepcopy(cfg)
    for k, v in changes.items():
        if k == "settings":
            out.setdefault("settings", {}).update(v)
        else:
            out[k] = v
    return out

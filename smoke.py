#!/usr/bin/env python
"""One-command offline functional acceptance of the CC4 testbed.

    python smoke.py [--config testbed/configs/smoke.json] [--out receipts] [--keep-going]

Runs the real CC4 (CybORG 4.0) environment with deterministic offline model clients. No API key,
no network: `socket.socket` and `urllib.request.urlopen` are replaced by functions that raise for
the whole run, and the shipped sources are scanned for network imports. Every check is recorded
in a JSON receipt; the exit code is 0 only if every check passed.

The process re-executes itself with PYTHONHASHSEED=0 so that a checkpoint restored in a second
process reproduces the same set-iteration order (the fork check restores one in a child process).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")   # pygame is an upstream import; no display is used

RECEIPT_SCHEMA = "cc4-testbed-smoke-receipt-v1"
REQUIRED_HASH_SEED = "0"
PINNED = ("numpy", "gym", "gymnasium", "networkx", "PyYAML", "prettytable", "pygame")
NETWORK_MARKERS = ("urllib.request", "http.client", "import requests", "import httpx",
                   "import aiohttp", "OPENROUTER", "api_key")
GROUND_TRUTH_MARKERS = ("environment_controller", "compromise_of", "get_true_state", ".state.")


class Fail(AssertionError):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


# ------------------------------------------------------------------------------------------------
# process hygiene
# ------------------------------------------------------------------------------------------------

def _reexec_with_hash_seed(argv):
    env = {**os.environ, "PYTHONHASHSEED": REQUIRED_HASH_SEED}
    return subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv], env=env)


def block_network():
    import socket
    import urllib.request

    def boom(*a, **k):
        raise RuntimeError("network access attempted during the offline smoke")
    socket.socket = boom
    socket.create_connection = boom
    socket.getaddrinfo = boom
    urllib.request.urlopen = boom
    urllib.request.OpenerDirector.open = boom


def source_tree_sha256() -> str:
    """sha256 over the relative path + LF-normalised contents of every shipped .py file."""
    digest = hashlib.sha256()
    files = sorted(p for d in ("CybORG", "flipthem", "testbed") for p in (ROOT / d).rglob("*.py")
                   if "__pycache__" not in p.parts) + [ROOT / "smoke.py"]
    for p in files:
        digest.update(str(p.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0" + p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------------

def check_network_guard(ctx):
    import socket
    import urllib.request
    for fn, args in ((socket.socket, ()), (urllib.request.urlopen, ("http://127.0.0.1:9/",))):
        try:
            fn(*args)
        except RuntimeError:
            pass
        else:
            raise Fail("network primitive was not blocked")
    scanned, hits = 0, []
    for d in ("flipthem", "testbed"):
        for p in sorted((ROOT / d).rglob("*.py")):
            scanned += 1
            text = p.read_text(encoding="utf-8")
            for marker in NETWORK_MARKERS:
                if marker in text:
                    hits.append(f"{p.relative_to(ROOT).as_posix()}: {marker}")
    need(not hits, f"network/credential markers in shipped sources: {hits}")
    real_client_shipped = (ROOT / "flipthem" / "llm" / "real_client.py").exists()
    real_client_importable = importlib.util.find_spec("flipthem.llm.real_client") is not None
    need(not real_client_shipped and not real_client_importable, "a real client module is shipped")
    return {"blocked_primitives": ["socket.socket", "socket.create_connection", "socket.getaddrinfo",
                                   "urllib.request.urlopen", "urllib.request.OpenerDirector.open"],
            "files_scanned": scanned, "marker_hits": hits, "markers": list(NETWORK_MARKERS),
            "real_client_shipped": real_client_shipped, "real_client_importable": real_client_importable}


def check_environment_startup(ctx):
    from CybORG.Simulator.Actions import Sleep
    from flipthem.resources import OPERATIONAL_RESOURCES, build_cc4_env, discover_resources
    from flipthem.wrapper import FlipThemCC4
    cfg, st = ctx["cfg"], ctx["settings"]
    env = build_cc4_env(cfg["seed"], steps=int(st["steps"]))
    res = discover_resources(env)
    need(sorted(res) == sorted(OPERATIONAL_RESOURCES), f"scored resources not found: {res}")
    slots = list(env.environment_controller.agent_interfaces)
    blue = sorted(s for s in slots if s.startswith("blue_agent_"))
    red = sorted(s for s in slots if s.startswith("red_agent_"))
    need(blue == [f"blue_agent_{i}" for i in range(5)], f"unexpected blue slots {blue}")
    need(len(red) >= 1, "no red slots")
    hosts = sorted(env.environment_controller.state.hosts)
    env2 = build_cc4_env(cfg["seed"], steps=int(st["steps"]))
    need(sorted(env2.environment_controller.state.hosts) == hosts, "same seed produced a different host set")
    game = FlipThemCC4(seed=cfg["seed"], steps=5)
    game.reset()
    info, phases = None, []
    for _ in range(5):
        info = game.step({s: Sleep() for s in game.blue_slots}, {s: Sleep() for s in game.red_slots})
        phases.append(game.env.environment_controller.state.mission_phase)
    need(all(ph == 0 for ph in phases), f"mission phase advanced: {phases}")
    need(game._tick == 5 and info["done"], "wrapper clock did not reach the horizon")
    return {"hosts": len(hosts), "blue_slots": blue, "red_slots": len(red),
            "scored_resources": list(OPERATIONAL_RESOURCES), "mission_phase_per_tick": phases,
            "final_tick": game._tick, "done_flag": info["done"]}


def check_legal_action_interface(ctx):
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Actions.AbstractActions import Restore
    from CybORG.Simulator.Actions.Action import InvalidAction
    from flipthem.llm import context, prompts, tools
    from flipthem.llm.agent import make_llm_policy
    from flipthem.llm.backend import RequestBudget
    from flipthem.llm.compiler import BLUE_ACTIONS, build_slot_action_index, compile_action
    from flipthem.llm.context import RESULT_INVALID, build_legal_context
    from flipthem.resources import OPERATIONAL_RESOURCES
    from flipthem.wrapper import FlipThemCC4
    from testbed.fake_client import OfflineStubClient, ScriptedClient, final
    cfg = ctx["cfg"]
    zone_a = OPERATIONAL_RESOURCES[0]
    slot = "blue_agent_1"
    game = FlipThemCC4(seed=cfg["seed"], steps=12)
    game.reset()

    def indices(obs):
        return {s: build_slot_action_index(s, game.env.get_action_space(s), BLUE_ACTIONS,
                                           OPERATIONAL_RESOURCES, obs[s]) for s in game.blue_slots}
    obs = game.team_observation("blue")
    idx = indices(obs)
    need(("Restore", zone_a) in idx[slot].legal_pairs, "Restore on the owned scored host is not listed")
    native = compile_action(idx[slot], "Restore", zone_a)
    need(isinstance(native, Restore), "listed pair did not compile to the native action")
    need(isinstance(compile_action(idx[slot], "Restore", "no-such-host"), InvalidAction), "unlisted target accepted")
    need(isinstance(compile_action(idx[slot], "ExploitRemoteService", zone_a), InvalidAction), "red action accepted for blue")
    ctx0 = build_legal_context("blue", game._tick, game.T, obs, {s: i.legal_pairs for s, i in idx.items()},
                               (), OPERATIONAL_RESOURCES)
    need(("Restore", zone_a) in ctx0.decidable[slot], "legal context omits a listed pair")
    game.step({**{s: Sleep() for s in game.blue_slots}, slot: native}, None)
    need(game._in_progress(slot), "submitted Restore is not in progress")
    obs1 = game.team_observation("blue")
    idx1 = indices(obs1)
    ctx1 = build_legal_context("blue", game._tick, game.T, obs1, {s: i.legal_pairs for s, i in idx1.items()},
                               (), OPERATIONAL_RESOURCES)
    need(ctx1.decidable[slot] == (), "busy slot still decidable")
    marker_hits = [(mod.__name__, marker) for mod in (context, tools, prompts)
                   for marker in GROUND_TRUTH_MARKERS if marker in inspect.getsource(mod)]
    need(not marker_hits, f"ground-truth access markers in the model-facing modules: {marker_hits}")
    # A model answer that names an unlisted target compiles to InvalidAction and is billed 0.1 once.
    g2 = FlipThemCC4(seed=cfg["seed"], steps=4)
    g2.reset()
    bad = ScriptedClient([final([{"unit": slot, "action": "Restore", "target": "no-such-host"}])])
    pol = make_llm_policy("blue", bad, "offline-scripted", tools_enabled=False, context_augmented=True,
                          budget=RequestBudget(limit=None))
    acts = pol(g2, g2.team_observation("blue"))
    need(isinstance(acts[slot], InvalidAction), "illegal model decision did not compile to InvalidAction")
    g2.step(acts, None)
    need(pol.telemetry.illegal_actions == 1, "illegal action not counted")
    need(any(e.unit == slot and e.result == RESULT_INVALID for e in pol.own_log), "own ledger lacks the invalid row")
    invalid_cost = g2.payoff()["cost_B"]
    need(abs(invalid_cost - 0.1) < 1e-9, f"InvalidAction billed {invalid_cost}, expected 0.1")
    # Legal model decisions become native actions in the environment.
    g3 = FlipThemCC4(seed=cfg["seed"], steps=6)
    g3.reset()
    stub = OfflineStubClient("blue")
    pol3 = make_llm_policy("blue", stub, "offline-stub", tools_enabled=False, context_augmented=True,
                           budget=RequestBudget(limit=None))
    submitted = []
    while g3._tick < g3.T:
        a = pol3(g3, g3.team_observation("blue"))
        submitted += [type(v).__name__ for v in a.values() if type(v).__name__ != "Sleep"]
        g3.step(a, None)
    pol3.flush(g3)
    need(submitted and pol3.telemetry.illegal_actions == 0, "stub produced no legal non-Sleep action")
    need("InvalidAction" not in submitted, "stub decision compiled to InvalidAction")
    return {"listed_pairs_for_owner": len(idx[slot].legal_pairs), "invalid_action_cost": invalid_cost,
            "illegal_actions_counted": pol.telemetry.illegal_actions,
            "ground_truth_marker_hits": marker_hits, "ground_truth_markers": list(GROUND_TRUTH_MARKERS),
            "stub_native_actions": sorted(set(submitted)), "stub_calls": stub.calls}


def check_holding_payoff(ctx):
    from CybORG.Shared.Session import RedAbstractSession
    from CybORG.Simulator.Actions import Sleep
    from flipthem.resources import OPERATIONAL_RESOURCES, build_cc4_env, compromise_of
    from flipthem.settlement import SYMMETRIC_VALUES, settle_control
    from flipthem.wrapper import FlipThemCC4
    cfg = ctx["cfg"]
    zone_a = OPERATIONAL_RESOURCES[0]
    env = build_cc4_env(cfg["seed"], steps=10)
    s0 = settle_control(env, SYMMETRIC_VALUES)
    need(s0["v_red"] == 0.0 and s0["v_blue"] == 2.0 and not any(s0["x"].values()), f"t0 settlement {s0}")
    env.environment_controller.state.add_session(RedAbstractSession(
        ident=None, hostname=zone_a, username="root", agent="red_agent_2", pid=None,
        session_type="RedAbstractSession"))
    s1 = settle_control(env, SYMMETRIC_VALUES)
    need(compromise_of(env, zone_a) == "red" and s1["x"][zone_a] == 1, "root session not read as red control")
    need(s1["v_red"] == 1.0 and s1["v_blue"] == 1.0, f"value split after compromise {s1}")
    # A 30-tick match in which Red takes a scored host at tick 10 (a root session is injected into
    # the live state): the per-tick split follows control, both sides' payoffs are
    # (held value - completion cost)/T, and the two payoffs sum to the constant-sum identity.
    game = FlipThemCC4(seed=cfg["seed"], steps=30)
    game.reset()
    vb = vr = 0.0
    red_ticks, flip_tick = 0, 10
    while game._tick < game.T:
        if game._tick == flip_tick:
            game.env.environment_controller.state.add_session(RedAbstractSession(
                ident=None, hostname=zone_a, username="root", agent="red_agent_2", pid=None,
                session_type="RedAbstractSession"))
        info = game.step({s: Sleep() for s in game.blue_slots}, {s: Sleep() for s in game.red_slots})
        need(abs(info["v_red_tick"] + info["v_blue_tick"] - sum(SYMMETRIC_VALUES.values())) < 1e-9,
             "per-tick value is not constant-sum")
        need((info["x"][zone_a] == 1) == (game._tick > flip_tick), f"control of {zone_a} at tick {game._tick}")
        vb += info["v_blue_tick"]
        vr += info["v_red_tick"]
        red_ticks += sum(info["x"].values())
    p, T = game.payoff(), game.T
    need(red_ticks == T - flip_tick, f"red held ticks {red_ticks}, expected {T - flip_tick}")
    need(abs(p["U_B"] * T + p["cost_B"] - vb) < 1e-9, "Blue payoff is not (held value - cost)/T")
    need(abs(p["U_R"] * T + p["cost_R"] - vr) < 1e-9, "Red payoff is not (held value - cost)/T")
    need(abs((p["U_R"] + p["U_B"]) - (p["sum_v"] - (p["cost_R"] + p["cost_B"]) / T)) < 1e-9, "payoff identity")
    return {"t0_settlement": s0, "after_red_root": {"x": s1["x"], "v_red": s1["v_red"], "v_blue": s1["v_blue"]},
            "match": {"T": T, "red_takes_control_after_tick": flip_tick, "red_resource_ticks": red_ticks,
                      "U_B": p["U_B"], "U_R": p["U_R"], "held_value_blue": vb, "held_value_red": vr,
                      "sum_identity_residual": abs((p["U_R"] + p["U_B"]) - (p["sum_v"] - (p["cost_R"] + p["cost_B"]) / T))}}


def check_completed_action_cost(ctx):
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Actions.AbstractActions import Restore
    from CybORG.Simulator.Actions.Action import InvalidAction
    from flipthem.billing import ft_cost
    from flipthem.execution_log import ExecutionRecorder
    from flipthem.resources import OPERATIONAL_RESOURCES
    from flipthem.wrapper import FlipThemCC4
    cfg = ctx["cfg"]
    zone_a = OPERATIONAL_RESOURCES[0]
    slot = "blue_agent_1"
    need(ft_cost(Restore(session=0, agent=slot, hostname=zone_a)) == 1.0, "Restore cost")
    need(ft_cost(InvalidAction()) == 0.1 and ft_cost(Sleep()) == 0.0, "InvalidAction/Sleep cost")
    game = FlipThemCC4(seed=cfg["seed"], steps=12)
    game.reset()
    rec = ExecutionRecorder("cost")
    game.step_observer = rec.on_step

    def sleep_all():
        return {s: Sleep() for s in game.blue_slots}
    game.step({**sleep_all(), slot: Restore(session=0, agent=slot, hostname=zone_a)}, None)
    timeline = []
    while game._tick < 9:
        busy = game._in_progress(slot)
        cost = game.billing.total(game.blue_slots)
        timeline.append({"tick": game._tick, "in_progress": busy, "cost_B": cost})
        # while the slot is busy, a resubmission is normalised to Sleep by the mechanism (never a
        # second action, never a second bill); once idle the slot only Sleeps
        resubmit = Restore(session=0, agent=slot, hostname=zone_a) if busy else Sleep()
        game.step({**sleep_all(), slot: resubmit}, None)
    need(all(t["cost_B"] == 0.0 for t in timeline if t["in_progress"]), "billed while in progress")
    need(any(t["in_progress"] for t in timeline), "Restore never in progress")
    done = [t for t in timeline if not t["in_progress"]]
    need(done and all(abs(t["cost_B"] - 1.0) < 1e-9 for t in done), f"completion billing {timeline}")
    rows = [r for r in rec.rows if r["slot"] == slot and r["action"] == "Restore"]
    need(len(rows) == 1 and rows[0]["executed"] and abs(rows[0]["billed_ft_cost"] - 1.0) < 1e-9,
         f"ledger rows for the Restore: {rows}")
    need(game.counts["completed_actions"] == 1, "completed action count")
    # the passive recorder does not change the game: the same scripted match without an observer
    # gives the same payoff, billing and control trace
    def scripted(attach):
        g = FlipThemCC4(seed=cfg["seed"], steps=12)
        g.reset()
        if attach:
            g.step_observer = ExecutionRecorder("probe").on_step
        tr = []
        for t in range(g.T):
            a = {s: Sleep() for s in g.blue_slots}
            if t == 0:
                a[slot] = Restore(session=0, agent=slot, hostname=zone_a)
            tr.append(g.step(a, None)["x"])
        return g.payoff(), dict(g.billing._charged), tr
    with_obs, without_obs = scripted(True), scripted(False)
    need(with_obs == without_obs, "attaching the execution recorder changed the game")
    # an action still in flight at the horizon is not billed
    g2 = FlipThemCC4(seed=cfg["seed"], steps=6)
    g2.reset()
    rec2 = ExecutionRecorder("horizon")
    g2.step_observer = rec2.on_step
    while g2._tick < 4:
        g2.step({s: Sleep() for s in g2.blue_slots}, None)
    g2.step({**{s: Sleep() for s in g2.blue_slots}, slot: Restore(session=0, agent=slot, hostname=zone_a)}, None)
    while g2._tick < g2.T:
        g2.step({s: Sleep() for s in g2.blue_slots}, None)
    need(g2._tick == g2.T and g2._in_progress(slot), "Restore is not in flight at the horizon")
    horizon_cost = g2.payoff()["cost_B"]
    horizon_rows = [r for r in rec2.rows if r["slot"] == slot]
    need(horizon_cost == 0.0 and not horizon_rows, "in-flight at horizon was billed")
    return {"timeline": timeline, "completed_tick": rows[0]["completed_tick"], "ledger_rows_for_slot": len(rows),
            "observer_detached_identical": with_obs == without_obs,
            "in_flight_at_horizon": {"in_progress": g2._in_progress(slot), "cost_B": horizon_cost,
                                     "ledger_rows": len(horizon_rows)}}


def _find_inflight_fork(ctx):
    """A fork instant under the REAL deadline rule with a Blue action genuinely in flight."""
    from testbed import fork as fk
    from testbed.config import build_match
    cfg, factory = ctx["cfg"], ctx["factory"]
    lo, hi = ctx["settings"].get("deadline_scan", [8, 30])
    tried = []
    for d in range(int(lo), int(hi)):
        m = build_match(cfg, factory, deadline_override=d)
        where = fk.drive(m.game, m.pol_b, m.pol_r, m.trace, stop=fk.fork_condition(m.pol_b))
        busy = [s for s in m.game.blue_slots if m.game._in_progress(s)]
        tried.append((d, where, m.game._tick, len(busy)))
        if where == "fork" and busy:
            return m, d, tried
    raise Fail(f"no deadline with an in-flight Blue action at the fork: {tried}")


def _settled_once(sw_pol, game, recorder, slot, name, fork, n_before, charged_at_fork):
    """The action that was in flight at the fork completes exactly once after it: one terminal
    row on the shared own-side ledger before the slot's next submission, exactly one executed
    ledger row in that window, and the slot's billed total grows by exactly the ledger rows."""
    from flipthem.llm.context import RESULT_IN_PROGRESS, RESULT_SUBMITTED
    entries = [e for e in sw_pol.comp[0].own_log if e.unit == slot]
    sub = max(i for i, e in enumerate(entries[:n_before]) if e.result == RESULT_SUBMITTED)
    nxt = next((i for i in range(sub + 1, len(entries)) if entries[i].result == RESULT_SUBMITTED), None)
    next_sub_tick = entries[nxt].round if nxt is not None else game.T
    terminal = [e for e in entries[sub + 1:(nxt if nxt is not None else len(entries))]
                if e.result not in (RESULT_SUBMITTED, RESULT_IN_PROGRESS)]
    need(len(terminal) == 1 and terminal[0].action == name,
         f"in-flight {name} resolved {len(terminal)} times on the own ledger")
    executed = [r for r in recorder.rows if r["slot"] == slot and r["completed_tick"] is not None
                and fork < r["completed_tick"] <= next_sub_tick and r["executed"] and r["action"] != "Sleep"]
    need(len(executed) == 1 and executed[0]["action"] == name,
         f"expected exactly one executed {name} row before the next submission, got {executed}")
    rows_after = [r for r in recorder.rows if r["slot"] == slot and r["completed_tick"] is not None
                  and r["completed_tick"] > fork]
    growth = game.billing._charged.get(slot, 0.0) - charged_at_fork.get(slot, 0.0)
    need(abs(growth - sum(r["billed_ft_cost"] for r in rows_after)) < 1e-9,
         "slot billing after the fork differs from the ledger rows")
    if name == "Restore":
        need(abs(executed[0]["billed_ft_cost"] - 1.0) < 1e-9, "Restore not billed 1.0")
    return {"action": name, "completed_tick": executed[0]["completed_tick"], "next_submission_tick": next_sub_tick,
            "billed": executed[0]["billed_ft_cost"], "terminal_ledger_entries": len(terminal),
            "executed_rows_in_window": len(executed), "slot_billing_growth_after_fork": growth}


def check_switch_b0_to_b1(ctx):
    from testbed import fork as fk
    from testbed.config import build_match, variant
    cfg, factory, st = ctx["cfg"], ctx["factory"], ctx["settings"]
    m, d, tried = _find_inflight_fork(ctx)
    ctx["fork_deadline"] = d
    game, sw, pol_r = m.game, m.pol_b, m.pol_r
    c0, c1 = m.clients["blue:B0"], m.clients["blue:B1"]
    fork = game._tick
    need(c1.calls == 0 and not sw._switched and sw._active == 0, "B1 touched before the switch")
    slot = next(s for s in game.blue_slots if game._in_progress(s))
    name, _target = sw.comp[0]._pending[slot]
    b0_calls_before = c0.calls
    n_before = sum(1 for e in sw.own_log if e.unit == slot)
    ledger_before = sw.comp[0]._own_log
    charged_at_fork = dict(game.billing._charged)
    wake_before = {"last_call_tick": sw._last_call_tick, "wake_spec": sw._wake_spec,
                   "b1_last_call_tick": sw.comp[1]._last_call_tick}
    need(wake_before["last_call_tick"] >= 0 and wake_before["b1_last_call_tick"] < 0,
         "unexpected wake state at the fork (B0 must have been called, B1 never)")
    # drive to the horizon, observing the switch instant
    at_switch = None
    while game._tick < game.T:
        active_before = sw._active
        b = sw(game, game.team_observation("blue"))
        if active_before == 0 and sw._active == 1:
            at_switch = {"tick": game._tick, "b1_last_call_tick": sw.comp[1]._last_call_tick,
                         "team_last_call_tick": sw._last_call_tick, "b1_calls": c1.calls, "b0_calls": c0.calls}
        r = pol_r(game, game.team_observation("red"))
        info = game.step(b, r)
        m.trace.append({k: int(v) for k, v in info["x"].items()})
    sw.flush(game)
    pol_r.flush(game)
    sl = sw.switch_log
    need(at_switch is not None and at_switch["tick"] == fork, f"switch not observed at the fork: {at_switch}")
    need(sl["switched"] and sl["switch_effective_tick"] == fork and sl["switch_reason"] == "deadline",
         f"switch log {sl}")
    need(sw._active == 1 and sw._switched, "policy not switched")
    need(c0.calls == b0_calls_before and at_switch["b0_calls"] == b0_calls_before, "B0 called after the switch")
    need(c1.calls > 0, "B1 never called after the switch")
    # wake clock carried, not reset: right after the switch call B1's clock is either the carried
    # B0 clock or the fork tick itself (if B1 was woken at once), never its never-called sentinel
    need(at_switch["b1_last_call_tick"] in (wake_before["last_call_tick"], fork)
         and at_switch["team_last_call_tick"] == at_switch["b1_last_call_tick"],
         f"wake clock not carried across the switch: before={wake_before} at_switch={at_switch}")
    need(sw.comp[0]._own_log is sw.comp[1]._own_log is ledger_before, "ledger not shared across the switch")
    need(sw.comp[0]._pending is sw.comp[1]._pending, "pending actions not shared")
    need(sw.comp[0].budget is sw.comp[1].budget, "budget not shared")
    need(sw.comp[0].budget.spent == c0.calls + c1.calls, "budget.spent != total physical calls")
    need(sl["budget_spent_final"] == sw.comp[0].budget.spent, "budget_spent_final")
    settled = _settled_once(sw, game, m.recorder, slot, name, fork, n_before, charged_at_fork)
    cap = float(st["max_cost_usd"])
    need(sl.get("cost_spent_before_switch", 0) > 0, "no cost accrued before the switch (usage_cost_usd must be > 0)")
    need(abs(sl["cost_cap_remaining_after_switch"] - (cap - sl["cost_spent_before_switch"])) < 1e-9, "inherited cost cap")
    need(abs(sw.comp[1].adapter.max_cost_usd - sl["cost_cap_remaining_after_switch"]) < 1e-9, "B1 cap not carried")
    need(game._tick == game.T, "match did not reach the horizon")
    # zero-switch parity: a never-firing selector is indistinguishable from the plain B0 policy
    base = variant(cfg, red={"kind": "native", "name": "fsm"}, settings={"steps": 20})
    never = variant(base, blue={**cfg["blue"], "arm": "never", "event_enabled": False, "deadline_tick": 10 ** 9})
    plain = variant(base, blue={"kind": "llm", "id": "B0", "model": cfg["blue"]["start"].get("model", "offline-stub-blue")})

    def seq(mm):
        out = []
        while mm.game._tick < mm.game.T:
            b = mm.pol_b(mm.game, mm.game.team_observation("blue"))
            out.append(tuple(sorted((s, type(a).__name__, repr(vars(a)) if hasattr(a, "__dict__") else "")
                                    for s, a in b.items())))
            mm.game.step(b, None)
        mm.pol_b.flush(mm.game)
        return out, mm.game.payoff()
    mn, mp = build_match(never, factory), build_match(plain, factory)
    sn, sp = seq(mn), seq(mp)
    need(sn == sp, "never-selector run differs from the plain B0 run")
    need(mn.clients["blue:B1"].calls == 0 and mn.clients["blue:B0"].calls == mp.clients["blue:B0"].calls,
         "zero-switch call counts differ")
    return {"deadline_tick": d, "switch_effective_tick": fork, "scan": tried, "in_flight_slot": slot,
            "in_flight_action": settled, "calls": {"B0": c0.calls, "B1": c1.calls, "red": m.clients["red"].calls},
            "wake_before_switch": wake_before, "at_switch": at_switch,
            "budget_spent": sw.comp[0].budget.spent, "cost_spent_before_switch": sl["cost_spent_before_switch"],
            "cost_cap_remaining_after_switch": sl["cost_cap_remaining_after_switch"],
            "zero_switch_parity": {"ticks": len(sn[0]), "sequences_equal": sn == sp,
                                   "never_B1_calls": mn.clients["blue:B1"].calls,
                                   "never_B0_calls": mn.clients["blue:B0"].calls,
                                   "plain_B0_calls": mp.clients["blue:B0"].calls},
            "payoff": game.payoff()}


def _fresh_handles(factory):
    from testbed.fork import HANDLE_TOKENS
    return {t: factory("red" if t == "red" else "blue", t, "offline-stub") for t in HANDLE_TOKENS}


def check_fork_save_restore(ctx):
    from testbed import fork as fk
    from testbed import fork_state as fs
    from testbed.config import build_match
    cfg, factory, d = ctx["cfg"], ctx["factory"], ctx["fork_deadline"]
    m = build_match(cfg, factory, deadline_override=d)
    where = fk.drive(m.game, m.pol_b, m.pol_r, m.trace, stop=fk.fork_condition(m.pol_b))
    need(where == "fork" and any(m.game._in_progress(s) for s in m.game.blue_slots), "fork instant not reproduced")
    game, sw, pol_r = m.game, m.pol_b, m.pol_r
    snap = fk.snapshot_match(game, sw, pol_r, m.trace, m.recorder)
    need(snap["copy_check"]["content"] == [], "live vs restored content differs")
    ha, hb = _fresh_handles(factory), _fresh_handles(factory)
    a, fa = fk.restore_branch(snap["blob"], ha, snap["reference_fp"])
    b, fb = fk.restore_branch(snap["blob"], hb, snap["reference_fp"])
    need(fs.compare_fingerprints(fa, fb) == {"content": [], "ordered": []}, "two restores differ")
    need(any(a["game"]._in_progress(s) for s in a["game"].blue_slots), "in-flight action lost in the copy")
    need(a["blue"]._wake_spec == sw._wake_spec and a["blue"]._last_call_tick == sw._last_call_tick, "wake state lost")
    need(a["blue"].comp[0].budget.spent == sw.comp[0].budget.spent > 0, "budget state lost")
    live_rng = game.env.environment_controller.state.np_random.bit_generator.state
    need(a["game"].env.environment_controller.state.np_random.bit_generator.state == live_rng, "env RNG not copied")
    need(a["blue"].comp[0].budget is a["blue"].comp[1].budget, "shared budget split by the copy")
    need(a["blue"].comp[0]._own_log is a["blue"].comp[1]._own_log, "shared ledger split by the copy")
    need(a["game"].step_observer.__self__ is a["recorder"], "recorder alias lost")
    need(a["game"] is not b["game"] and a["game"].env is not b["game"].env, "branches share an environment")
    need(a["blue"].comp[0].client is ha["blue:B0"] and b["blue"].comp[0].client is hb["blue:B0"]
         and a["red"].client is ha["red"], "clients not rebound to the fresh handles")
    need(b"OfflineStubClient" not in snap["blob"], "checkpoint serialised a client object")
    before_b = fs.state_fingerprint(b, hb)
    for _ in range(5):
        bb = a["blue"](a["game"], a["game"].team_observation("blue"))
        rr = a["red"](a["game"], a["game"].team_observation("red"))
        a["game"].step(bb, rr)
    need(fs.compare_fingerprints(before_b, fs.state_fingerprint(b, hb)) == {"content": [], "ordered": []},
         "advancing branch A changed branch B")
    need(all(c.calls == 0 for c in hb.values()) and sum(c.calls for c in ha.values()) > 0, "client isolation")
    try:
        fk.snapshot_match(game, sw, pol_r, m.trace, m.recorder,
                          handles={"blue:B0": sw.comp[0].client, "blue:B1": sw.comp[1].client})
    except fs.CheckpointError:
        pass
    else:
        raise Fail("a client without a handle was serialised")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "checkpoint.pkl"
        p.write_bytes(snap["blob"])
        out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fingerprint", str(p)],
                             env={**os.environ, "PYTHONHASHSEED": REQUIRED_HASH_SEED},
                             capture_output=True, text=True)
        need(out.returncode == 0, f"child restore failed: {out.stderr[-800:]}")
        child_fp = json.loads(out.stdout.strip().splitlines()[-1])
    cross = fs.compare_fingerprints(fa, child_fp)
    need(cross == {"content": [], "ordered": []}, f"cross-process restore differs: {cross}")
    ctx["snapshot"], ctx["fork_match"] = snap, m
    return {"fork_tick": game._tick, "checkpoint_bytes": snap["bytes"], "checkpoint_sha256": snap["sha256"],
            "handles": snap["handles"], "live_vs_restored": snap["copy_check"],
            "restored_vs_restored": fs.compare_fingerprints(fa, fb), "cross_process_restore": cross,
            "in_progress_at_fork": snap["at_fork"]["in_progress_slots"],
            "isolation": {"branch_B_fingerprint_unchanged": fs.compare_fingerprints(before_b, fs.state_fingerprint(b, hb)),
                          "branch_B_client_calls": {t: c.calls for t, c in hb.items()},
                          "branch_A_client_calls": {t: c.calls for t, c in ha.items()}},
            "client_class_in_checkpoint_bytes": b"OfflineStubClient" in snap["blob"],
            "fingerprint_all": fa["ordered"]["__all__"]}


def check_fork_continuation(ctx):
    from testbed import fork as fk
    from testbed.config import build_match
    cfg, factory, d = ctx["cfg"], ctx["factory"], ctx["fork_deadline"]
    snap, m = ctx["snapshot"], ctx["fork_match"]
    game, sw, pol_r = m.game, m.pol_b, m.pol_r
    fork, T, at_fork = game._tick, game.T, snap["at_fork"]
    slot = next(s for s in game.blue_slots if game._in_progress(s))
    name, _ = sw.comp[0]._pending[slot]
    n_before = sum(1 for e in sw.own_log if e.unit == slot)
    charged_at_fork = at_fork["charged_by_slot"]

    def segment_ok(seg):
        return (seg["ticks"] == T - fork and seg["trace_ticks_after_fork"] == T - fork
                and all(seg[s][k] < 1e-9 for s in ("blue", "red")
                        for k in ("trace_vs_wrapper_residual", "ledger_vs_wrapper_residual", "identity_residual")))
    runs, settled, segments = [], {}, {}
    for tag in ("A1", "A2"):
        h = _fresh_handles(factory)
        bd, _fp = fk.restore_branch(snap["blob"], h, snap["reference_fp"])
        b0 = fk.branch_policy(bd["blue"], "A")
        log = fk.tick_log(bd["game"], b0, bd["red"], bd["trace"])
        need(bd["game"]._tick == T and not bd["blue"]._switched, "branch A did not finish unswitched")
        settled[tag] = _settled_once(bd["blue"], bd["game"], bd["recorder"], slot, name, fork, n_before, charged_at_fork)
        segments[tag] = fk.segment_payoff(at_fork, bd["game"], bd["trace"], bd["recorder"].rows)
        need(segment_ok(segments[tag]), f"segment accounting {segments[tag]}")
        runs.append((log, list(bd["trace"]), bd["game"].payoff(), dict(bd["game"].billing._charged),
                     [c["user"] for c in b0.conversation], list(b0.own_log), h["blue:B1"].calls))
    a_identical = runs[0] == runs[1]
    need(a_identical, "two B0 continuations of the same checkpoint differ")
    need(runs[0][-1] == 0, "B1 called in branch A")
    live_log = fk.tick_log(game, fk.branch_policy(sw, "A"), pol_r, m.trace)
    a_equals_live = live_log == runs[0][0] and m.trace == runs[0][1] and game.payoff() == runs[0][2]
    need(a_equals_live, "restored B0 continuation differs from the unforked live continuation")
    hB = _fresh_handles(factory)
    bd, _fp = fk.restore_branch(snap["blob"], hB, snap["reference_fp"])
    resB = fk.continue_branch(bd, "B")
    sl = bd["blue"].switch_log
    need(resB["semantics_ok"] and sl["switch_effective_tick"] == fork and bd["game"]._tick == T, f"branch B {sl}")
    need(hB["blue:B1"].calls > 0 and hB["blue:B0"].calls == 0, "branch B call pattern")
    settled["B"] = _settled_once(bd["blue"], bd["game"], bd["recorder"], slot, name, fork, n_before, charged_at_fork)
    segments["B"] = fk.segment_payoff(at_fork, bd["game"], bd["trace"], bd["recorder"].rows)
    need(segment_ok(segments["B"]), f"segment accounting {segments['B']}")
    # prefix + branch B reproduces an unforked match under the same deadline rule
    m2 = build_match(cfg, factory, deadline_override=d)
    fk.drive(m2.game, m2.pol_b, m2.pol_r, m2.trace)
    m2.pol_b.flush(m2.game)
    m2.pol_r.flush(m2.game)
    need(m2.pol_b.switch_log["switch_effective_tick"] == fork, "unforked deadline arm switched elsewhere")
    b_equals_unforked = m2.trace == bd["trace"] and m2.game.payoff() == bd["game"].payoff()
    need(b_equals_unforked, "prefix + B differs from the unforked deadline arm")
    return {"fork_tick": fork, "segment_ticks": T - fork, "in_flight_action": name,
            "A_payoff": runs[0][2], "B_payoff": bd["game"].payoff(), "unforked_deadline_arm_payoff": m2.game.payoff(),
            "segments": segments, "in_flight_settled": settled,
            "A1_equals_A2": a_identical, "A_equals_unforked_B0_continuation": a_equals_live,
            "prefix_plus_B_equals_unforked_deadline_arm": b_equals_unforked,
            "B_switch_log": {k: sl[k] for k in ("switched", "switch_effective_tick", "switch_reason",
                                                 "budget_spent_final", "cost_cap_remaining_after_switch")},
            "B1_calls_in_A": runs[0][-1], "B0_calls_after_fork_in_B": hB["blue:B0"].calls,
            "B1_calls_in_B": hB["blue:B1"].calls}


CHECKS = [
    ("network_guard", check_network_guard),
    ("environment_startup", check_environment_startup),
    ("legal_action_interface", check_legal_action_interface),
    ("holding_payoff", check_holding_payoff),
    ("completed_action_cost", check_completed_action_cost),
    ("switch_b0_to_b1", check_switch_b0_to_b1),
    ("fork_save_restore", check_fork_save_restore),
    ("fork_continuation", check_fork_continuation),
]


# ------------------------------------------------------------------------------------------------
# entry points
# ------------------------------------------------------------------------------------------------

def _child_fingerprint(path: str) -> int:
    """Restore a checkpoint in THIS process with fresh stub handles and print its fingerprint."""
    from testbed import fork_state as fs
    from testbed.fake_client import OfflineStubClient
    from testbed.fork import HANDLE_TOKENS
    handles = {t: OfflineStubClient("red" if t == "red" else "blue") for t in HANDLE_TOKENS}
    bundle = fs.restore(Path(path).read_bytes(), handles)
    print(json.dumps(fs.state_fingerprint(bundle, handles)))
    return 0


def _relative(text: str) -> str:
    """Strip machine-specific prefixes (this tree, the interpreter prefix, the home directory)
    from a message so that receipts never carry absolute paths."""
    for prefix, label in ((str(ROOT), "<repo>"), (sys.prefix, "<python>"), (str(Path.home()), "<home>")):
        if prefix:
            text = text.replace(prefix, label).replace(prefix.replace("\\", "/"), label)
    return text


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def run(args) -> int:
    from testbed.config import load_config, settings_of
    from testbed.fake_client import stub_factory
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    ctx = {"cfg": cfg, "settings": settings_of(cfg), "factory": stub_factory(cfg.get("client"))}
    # Import the modules that reference `socket.socket` as a TYPE before the primitive is replaced.
    import testbed.fork  # noqa: F401
    import testbed.fork_state  # noqa: F401
    block_network()
    started = time.time()
    results = []
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            details = fn(ctx)
            ok, err = True, None
        except Exception as exc:   # noqa: BLE001 - every failure must land in the receipt
            details, ok = None, False
            err = {"type": type(exc).__name__, "message": _relative(str(exc))[:1000],
                   "traceback": _relative(traceback.format_exc())[-2500:]}
        results.append({"check": name, "ok": ok, "seconds": round(time.time() - t0, 2),
                        "details": _json_safe(details), "error": err})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({results[-1]['seconds']}s)"
              + ("" if ok else f": {err['type']}: {err['message'][:200]}"))
        if not ok and not args.keep_going:
            break
    all_ok = all(r["ok"] for r in results) and len(results) == len(CHECKS)
    packages = {}
    for dist in PINNED:
        try:
            packages[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            packages[dist] = None
    try:
        cfg_rel = cfg_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        cfg_rel = cfg_path.name
    receipt = {
        "schema": RECEIPT_SCHEMA, "all_ok": all_ok,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.time() - started, 1),
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "os": platform.system(), "machine": platform.machine(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "packages": packages, "cyborg_version": (ROOT / "CybORG" / "version.txt").read_text().strip(),
        "source_tree_sha256": source_tree_sha256(), "config_path": cfg_rel, "config": cfg,
        "checks": results,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"smoke-{receipt['utc'].replace(':', '').replace('-', '')}.json"
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        shown = out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = out.name
    print(f"{'ALL CHECKS PASSED' if all_ok else 'SMOKE FAILED'}: {sum(r['ok'] for r in results)}/{len(CHECKS)} "
          f"in {receipt['elapsed_seconds']}s -> {shown}")
    return 0 if all_ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "testbed" / "configs" / "smoke.json"))
    parser.add_argument("--out", default=str(ROOT / "receipts"))
    parser.add_argument("--keep-going", action="store_true", help="run every check even after a failure")
    parser.add_argument("--fingerprint", metavar="CHECKPOINT", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fingerprint:
        return _child_fingerprint(args.fingerprint)
    if os.environ.get("PYTHONHASHSEED") != REQUIRED_HASH_SEED:
        return _reexec_with_hash_seed(sys.argv[1:])
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

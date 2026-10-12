"""Shared-prefix fork: pause a live match at the deadline rule's own activation instant, checkpoint
the whole match state, and continue independent branches from identical restored copies.

Branch ``A`` continues the first Blue component (B0); branch ``B`` lets the switching policy
activate the second component (B1) at that same instant. Both inherit the ONE request budget, the
shared ledger, the pending (in-flight) actions, the wake state and the remaining cost cap.
"""
from __future__ import annotations

from dataclasses import asdict

from flipthem.llm.agent import LLMPolicy
from flipthem.switching_policy import SwitchingTeamPolicy
from testbed import fork_state as fs

HANDLE_TOKENS = ("blue:B0", "blue:B1", "red")
BRANCHES = {"A": "A_continue_B0", "B": "B_activate_B1"}


def fork_condition(pol_b: SwitchingTeamPolicy):
    """The deadline selector's own activation condition (event input off), evaluated BEFORE the
    policy call: this tick's activation would happen inside the next `pol_b` call."""
    sel = pol_b.selector
    if sel.event_enabled:
        raise ValueError("a fork prefix must use the deadline selector (event off)")

    def cond(game):
        return (not pol_b._switched and game._tick >= sel.deadline_tick
                and pol_b._has_decidable(game, game.team_observation("blue")))
    return cond


def drive(game, pol_b, pol_r, trace, stop=None):
    """The per-tick observe -> act -> step loop. Returns 'fork' when `stop` fires, else 'end'."""
    while game._tick < game.T:
        if stop is not None and stop(game):
            return "fork"
        b = pol_b(game, game.team_observation("blue"))
        r = pol_r(game, game.team_observation("red")) if pol_r else None
        info = game.step(b, r)
        trace.append({k: int(v) for k, v in info["x"].items()})
    return "end"


def _policies(pol_b, pol_r):
    comps = list(pol_b.comp) if isinstance(pol_b, SwitchingTeamPolicy) else [pol_b]
    if pol_r is not None and isinstance(pol_r, LLMPolicy):
        comps.append(pol_r)
    return comps


def handles_of(pol_b, pol_r) -> dict:
    """Every model client reachable from the policies, keyed by its handle token."""
    out = {}
    if isinstance(pol_b, SwitchingTeamPolicy):
        out["blue:B0"], out["blue:B1"] = pol_b.comp[0].client, pol_b.comp[1].client
    elif isinstance(pol_b, LLMPolicy):
        out["blue:B0"] = pol_b.client
    if pol_r is not None and isinstance(pol_r, LLMPolicy):
        out["red"] = pol_r.client
    return out


class _Placeholder:
    def __init__(self, token):
        self.token = token

    def next(self, *a, **k):
        raise AssertionError(f"placeholder client {self.token} must never be called")


def at_fork_state(game, pol_b, pol_r) -> dict:
    """Plain facts at the fork instant (identical for both branches by construction)."""
    comps = pol_b.comp if isinstance(pol_b, SwitchingTeamPolicy) else (pol_b, None)
    c0 = comps[0]
    out = {
        "tick": game._tick,
        "v_blue_sum": game._v_blue_sum, "v_red_sum": game._v_red_sum,
        "cost_B": game.billing.total(game.blue_slots), "cost_R": game.billing.total(game.red_slots),
        "charged_by_slot": dict(game.billing._charged),
        "counts": dict(game.counts),
        "in_progress_slots": sorted(s for s in game.blue_slots + game.red_slots if game._in_progress(s)),
        "blue_pending": {k: list(v) for k, v in c0._pending.items()},
        "blue_wake": {"last_call_tick": pol_b._last_call_tick, "wake_spec": pol_b._wake_spec},
        "blue_budget_spent": c0.budget.spent,
        "blue_cost_cap": c0.adapter.max_cost_usd,
        "blue_cost_spent_B0": c0.adapter._cost_spent,
        "request_log_len": {"blue:B0": len(c0.adapter.request_log)},
        "telemetry": {"blue": asdict(pol_b.telemetry)},
    }
    if comps[1] is not None:
        out["blue_cost_spent_B1"] = comps[1].adapter._cost_spent
        out["request_log_len"]["blue:B1"] = len(comps[1].adapter.request_log)
    if pol_r is not None and isinstance(pol_r, LLMPolicy):
        out.update({"red_pending": {k: list(v) for k, v in pol_r._pending.items()},
                    "red_wake": {"last_call_tick": pol_r._last_call_tick, "wake_spec": pol_r._wake_spec},
                    "red_budget_spent": pol_r.budget.spent,
                    "red_cost_cap": pol_r.adapter.max_cost_usd, "red_cost_spent": pol_r.adapter._cost_spent})
        out["request_log_len"]["red"] = len(pol_r.adapter.request_log)
        out["telemetry"]["red"] = asdict(pol_r.telemetry)
    return out


def snapshot_match(game, pol_b, pol_r, trace, recorder, *, handles=None) -> dict:
    """Checkpoint the live match graph. Every model client must be covered by a handle token
    (clients are never serialised); the checkpoint is then restored once with placeholders and
    its CONTENT fingerprint must equal the live one."""
    handles = dict(handles_of(pol_b, pol_r) if handles is None else handles)
    covered = {id(h) for h in handles.values()}
    for pol in _policies(pol_b, pol_r):
        if id(pol.client) not in covered:
            raise fs.CheckpointError(f"model client of {pol.team} policy has no handle; refusing to serialise it")
    bundle = {"game": game, "blue": pol_b, "red": pol_r, "trace": trace, "recorder": recorder}
    live_fp = fs.state_fingerprint(bundle, handles)
    blob = fs.snapshot(bundle, handles)
    placeholders = {t: _Placeholder(t) for t in handles}
    ref, stored_rng = fs.restore_with_rng(blob, placeholders)
    ref_fp = fs.state_fingerprint(ref, placeholders, global_rng=stored_rng)
    copy_check = fs.compare_fingerprints(live_fp, ref_fp)
    if copy_check["content"]:
        raise fs.CheckpointError(f"restored state differs from the live state: {copy_check}")
    return {"blob": blob, "sha256": fs.sha256(blob), "bytes": len(blob), "schema": fs.CHECKPOINT_SCHEMA,
            "handles": sorted(handles), "live_fp": live_fp, "reference_fp": ref_fp,
            "copy_check": copy_check, "at_fork": at_fork_state(game, pol_b, pol_r)}


def restore_branch(blob: bytes, handles: dict, reference_fp: dict | None = None):
    """A fresh, independent copy bound to `handles`; must equal the reference in BOTH modes."""
    bundle = fs.restore(blob, handles)
    fp = fs.state_fingerprint(bundle, handles)
    if reference_fp is not None:
        mismatch = fs.compare_fingerprints(reference_fp, fp)
        if mismatch["content"] or mismatch["ordered"]:
            raise fs.CheckpointError(f"restored branch differs from the reference: {mismatch}")
    return bundle, fp


def branch_policy(sw_pol: SwitchingTeamPolicy, label: str):
    """A: the first component alone, carrying the team's wake state. B: the switching policy."""
    if label == "A":
        pol = sw_pol.comp[0]
        pol._last_call_tick, pol._wake_spec = sw_pol._last_call_tick, sw_pol._wake_spec
        return pol
    if label == "B":
        return sw_pol
    raise ValueError(label)


def continue_branch(bundle: dict, label: str) -> dict:
    """Drive one restored branch to the horizon and flush both sides."""
    game, sw_pol, pol_r = bundle["game"], bundle["blue"], bundle["red"]
    pol_b = branch_policy(sw_pol, label)
    end = drive(game, pol_b, pol_r, bundle["trace"])
    if label == "A":
        pol_b.flush(game)
        sw_pol.switch_log["budget_spent_final"] = pol_b.budget.spent
    else:
        sw_pol.flush(game)
    if pol_r is not None:
        pol_r.flush(game)
    sl = sw_pol.switch_log
    if label == "B":
        semantics_ok = bool(sl["switched"] and sl["switch_reason"] == "deadline")
    else:
        semantics_ok = len(sw_pol.comp[1].adapter.request_log) == 0 and not sw_pol._switched
    return {"label": label, "name": BRANCHES[label], "end": end, "final_tick": game._tick,
            "payoff": game.payoff(), "switch_log": dict(sl), "semantics_ok": semantics_ok}


def tick_log(game, pol_b, pol_r, trace):
    """Continue to the horizon, logging every observation and every submitted action."""
    log = []
    while game._tick < game.T:
        ob, orr = game.team_observation("blue"), game.team_observation("red")
        b = pol_b(game, ob)
        r = pol_r(game, orr) if pol_r else None
        acts = {**b, **(r or {})}
        log.append((game._tick, fs._Canon(True, {}).h(ob).hex(), fs._Canon(True, {}).h(orr).hex(),
                    sorted((k, type(v).__name__, repr(vars(v)) if hasattr(v, "__dict__") else "")
                           for k, v in acts.items())))
        info = game.step(b, r)
        trace.append({k: int(v) for k, v in info["x"].items()})
    pol_b.flush(game)
    if pol_r is not None:
        pol_r.flush(game)
    return log


def segment_payoff(at_fork: dict, game, trace: list, rows: list) -> dict:
    """Post-fork segment under the original payoff definition, re-derived from INDEPENDENT
    records: the segment value from the per-tick control trace appended after the fork and the
    segment cost from the execution ledger rows completed after the fork. Each is compared with
    the wrapper's own accumulators, and the identity U * T == (prefix value - prefix cost) +
    U_segment * n is evaluated with the re-derived segment, so it is not a tautology."""
    fork = at_fork["tick"]
    n = game._tick - fork
    p = game.payoff()
    values = game.values
    post = trace[fork:]
    seg_rows = [r for r in rows if r.get("completed_tick") is not None and r["completed_tick"] > fork
                and r.get("executed") and r.get("action") not in (None, "Sleep")]
    out = {"fork_tick": fork, "ticks": n, "T": game.T, "trace_ticks_after_fork": len(post),
           "ledger_rows_after_fork": len(seg_rows)}
    for side, slots, v_key, c_key, u_key in (("blue", game.blue_slots, "v_blue_sum", "cost_B", "U_B"),
                                             ("red", game.red_slots, "v_red_sum", "cost_R", "U_R")):
        held = (lambda x, r: 1 - x.get(r, 0)) if side == "blue" else (lambda x, r: x.get(r, 0))
        dv_trace = float(sum(values[r] * held(x, r) for x in post for r in values))
        dc_ledger = float(sum(r["billed_ft_cost"] for r in seg_rows if r["slot"] in set(slots)))
        v_now = game._v_blue_sum if side == "blue" else game._v_red_sum
        dv_wrapper = v_now - at_fork[v_key]
        dc_wrapper = game.billing.total(slots) - at_fork[c_key]
        u_seg = (dv_trace - dc_ledger) / n if n else None
        identity = abs(p[u_key] * game.T - ((at_fork[v_key] - at_fork[c_key]) + (u_seg or 0.0) * n))
        out[side] = {"value_segment_from_trace": dv_trace, "value_segment_from_wrapper": dv_wrapper,
                     "ft_cost_segment_from_ledger": dc_ledger, "ft_cost_segment_from_wrapper": dc_wrapper,
                     "U_segment": u_seg, "U_full": p[u_key],
                     "trace_vs_wrapper_residual": abs(dv_trace - dv_wrapper),
                     "ledger_vs_wrapper_residual": abs(dc_ledger - dc_wrapper),
                     "identity_residual": identity}
    return out

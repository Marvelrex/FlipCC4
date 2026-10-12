"""Bounded, episode-local RED memory, conditional plans, and observed feedback.

Only LegalContext and the declared action manual enter this module. There is no
environment, rollout, probability estimate, reward, network, or random source.
Model goals/hypotheses/status are recorded separately from program observations.
Plans never authorize execution: current frontier legality remains a membership
check, and future timing is an optimistic bound using declared durations.

Limits are fixed: 16 tasks/plans, 32 steps per plan, 8 conditions per step,
16 hypotheses per task, 128 events/outcomes, and 512 values per observed fact
collection. Old events/outcomes are dropped, never replayed from the cumulative
own_log. The append cursor itself uses constant memory. Create a fresh instance
for a new episode; acknowledge feedback only after delivering a decision.
"""
from __future__ import annotations

from copy import deepcopy
import ipaddress
import json
import re

from flipthem.llm.context import LegalContext
from flipthem.llm.tools import action_manual


CAMPAIGN_MODES = ("off", "memory", "planning", "feedback")
CAMPAIGN_VERSION = "8"
CAMPAIGN_LIMITS = {
    "max_tasks": 16, "max_steps_per_plan": 32, "max_events": 128,
    "max_outcomes": 128, "max_hypotheses": 16, "max_conditions_per_step": 8,
    "max_text_length": 2048, "max_target_length": 256, "max_id_length": 64,
    "max_fact_values": 512,
}
# The durable campaign ledger remains available through ``campaign_state`` and is saved in full
# in every conversation artifact. Repeating retained facts/events/outcomes in every user message
# made the prompt grow linearly and duplicated the compact history/current state from prompts.py.
# The automatic campaign view carries only its unique tasks/plans and recent task-associated
# events; the full bounded ledger remains saved and callable.
CAMPAIGN_PROMPT_RECENT = 20
CAMPAIGN_PROMPT_MAX_CHARS = 8000
_RED_ACTIONS = ("DiscoverRemoteSystems", "AggressiveServiceDiscovery",
                "StealthServiceDiscovery", "ExploitRemoteService", "PrivilegeEscalate", "Withdraw")
_CONDITIONS = ("session", "root", "discovered_host", "discovered_ip", "discovered_subnet")
_TERMINAL = ("succeeded", "failed", "invalid", "session_lost")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def _mode(mode):
    if mode not in CAMPAIGN_MODES:
        raise ValueError(f"campaign mode must be one of {CAMPAIGN_MODES}")
    return mode


def _text(value, label, limit, token=False):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be a nonempty string of at most {limit} characters")
    if token and not _TOKEN.fullmatch(value):
        raise ValueError(f"{label} must be an identifier without whitespace")
    return value


def _object(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError("arguments must be an object")
    if any(key not in allowed for key in value):
        raise ValueError("unknown argument field")
    if any(key not in value for key in required):
        raise ValueError("missing required argument field")


def _array(value, label, maximum, minimum=0):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{label} must be a list with {minimum} to {maximum} items")
    return value


def _target(action, target):
    _text(target, "target", CAMPAIGN_LIMITS["max_target_length"])
    kind = action_manual(action)["target"]
    try:
        if kind == "ip_address":
            ipaddress.ip_address(target)
        elif kind == "ip_address+hostname":
            parts = target.split("|")
            if len(parts) != 2:
                raise ValueError("Withdraw target must be ip|hostname")
            ipaddress.ip_address(parts[0])
            _text(parts[1], "hostname", 256, token=True)
        elif kind == "hostname":
            _text(target, "hostname", 256, token=True)
            try:
                ipaddress.ip_address(target)
            except ValueError:
                pass
            else:
                raise ValueError("hostname target cannot be an IP address")
        elif kind == "subnet":
            # Native CC4 subnet names are identifiers; CIDR strings are also
            # shape-correct, but are legal only if ctx.decidable says so.
            if "/" in target:
                ipaddress.ip_network(target, strict=False)
            else:
                _text(target, "subnet", 256, token=True)
    except ValueError as exc:
        raise ValueError(f"invalid target shape for {action}: {exc}") from None


def _values(values):
    """Bound legal observed strings as well as model-controlled memory."""
    return sorted({v[:256] for v in values if isinstance(v, str)})[:512]


class CampaignState:
    def __init__(self, mode="off"):
        self.mode = _mode(mode)
        self._tick = None
        self._tasks = {}
        self._plans = {}
        self._plan_cursors = {}
        self._events = []
        self._facts = {"holdings": {"sessions": [], "root": [], "discovered": {
            "hosts": [], "ips": [], "subnets": []}}, "red_sessions_by_unit": {}, "outcomes": []}
        self._cursor = 0
        self._pending = False
        self._public_objectives_seeded = False

    def _seed_public_objectives(self, ctx: LegalContext):
        """Create auditable objective tasks from public scoring identities only.

        The full feedback profile previously depended on the model voluntarily calling
        ``remember_task``.  A live run made 286 tool calls yet created no task or plan, leaving
        the strongest assistance layer inert.  These two tasks contain no route, IP, service,
        control, opponent, or ground-truth fact: they restate the public objective already in
        every prompt and give ``check_campaign_plan`` stable task IDs from tick zero.
        """
        if self.mode != "feedback" or self._public_objectives_seeded:
            return
        self._public_objectives_seeded = True
        for index, target in enumerate(ctx.target_hosts, start=1):
            ident = f"scoring_target_{index}"
            self._tasks[ident] = {
                "id": ident,
                "goal": "Public scoring target",
                "target": target,
                "hypotheses": [],
                "model_status": "active",
                "observed_status": "unverified",
                "source": "program_public_objective",
                "created_tick": ctx.round,
                "updated_tick": ctx.round,
            }

    def snapshot(self):
        return deepcopy({"mode": self.mode, "version": CAMPAIGN_VERSION, "tick": self._tick,
                         "tasks": self._tasks, "plans": self._plans, "facts": self._facts,
                         "events": self._events, "pending_replan": self._pending,
                         "limits": CAMPAIGN_LIMITS})

    def prompt_snapshot(self):
        """Bound the automatic JSON view; full task/plan content stays in campaign_state.

        Keep identifiers exact. Only goal previews are shortened, with an explicit flag.
        Omissions are counted so a missing item cannot be mistaken for absent memory.
        The cap measures JSON with the same encoding used by the runtime prompt.
        """
        state = self.snapshot()
        all_events = state["events"]
        relevant_events = [event for event in all_events if event.get("task_ids")]
        outcomes = state["facts"]["outcomes"]
        out = {
            "tick": state["tick"],
            "pending_replan": state["pending_replan"],
            "tasks": {}, "plans": {}, "recent_task_events": [],
            "prompt_view": {
            "recent_limit": CAMPAIGN_PROMPT_RECENT,
            "max_chars": CAMPAIGN_PROMPT_MAX_CHARS,
            "audit_outcomes_total": len(outcomes),
            "audit_events_total": len(all_events),
            "task_events_total": len(relevant_events),
            "task_events_omitted": len(relevant_events),
            "tasks_omitted": len(state["tasks"]),
            "plan_steps_omitted": sum(len(p["steps"]) for p in state["plans"].values()),
            "hypotheses_in_full_ledger": True,
            "full_ledger_available_via": "campaign_state",
            },
        }
        def fits():
            return len(json.dumps(out, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False)) <= CAMPAIGN_PROMPT_MAX_CHARS

        # Public objectives first, then active/replan work before completed/abandoned tasks.
        tasks = sorted(state["tasks"].items(), key=lambda item: (
            item[1]["source"] != "program_public_objective",
            item[1]["model_status"] != "active", item[0]))
        for ident, task in tasks:
            out["tasks"][ident] = {k: task[k] for k in (
                "target", "source", "model_status", "observed_status")}
            out["tasks"][ident].update(goal_preview=task["goal"][:120],
                                       goal_truncated=len(task["goal"]) > 120)
            if not fits():
                del out["tasks"][ident]
            else:
                out["prompt_view"]["tasks_omitted"] -= 1

        # Feedback precedes plan details: a long plan must not bury a recent loss notice.
        for event in reversed(relevant_events[-CAMPAIGN_PROMPT_RECENT:]):
            out["recent_task_events"].insert(0, event)
            if not fits():
                out["recent_task_events"].pop(0)
            else:
                out["prompt_view"]["task_events_omitted"] -= 1
        # Round-robin across tasks, up to four unfinished steps each. Completed history,
        # dependency edges and hypotheses remain available through campaign_state.
        candidates = []
        for ident, _ in tasks:
            plan = state["plans"].get(ident)
            if plan:
                steps = sorted((s for s in plan["steps"] if s["status"] != "succeeded"), key=lambda s: (
                    not s.get("ready", False), s["status"] not in ("in_progress", "blocked", "failed"),
                    s["status"] == "succeeded"))[:4]
                candidates.append((ident, plan, steps))
        for index in range(CAMPAIGN_LIMITS["max_steps_per_plan"]):
            for ident, plan, steps in candidates:
                if index >= len(steps):
                    continue
                step = steps[index]
                view = out["plans"].setdefault(ident, {"status": plan["status"],
                    "steps_total": len(plan["steps"]), "steps": []})
                view["steps"].append({k: step[k] for k in (
                    "id", "unit", "action", "target", "status", "validity", "ready", "current_legal")})
                if not fits():
                    view["steps"].pop()
                    if not view["steps"]:
                        del out["plans"][ident]
                else:
                    out["prompt_view"]["plan_steps_omitted"] -= 1
        return out

    def acknowledge(self):
        self._pending = False

    def dispatch(self, ctx: LegalContext, name, args):
        """Validate fully before committing model memory; malformed calls are atomic."""
        if self.mode == "off" or ctx.team != "red":
            return {"error": "campaign tools require an enabled RED campaign"}
        try:
            if name == "campaign_state":
                _object(args, ())
                return self.snapshot()
            if name == "remember_task":
                return self._remember(ctx, args)
            if name == "check_campaign_plan" and self.mode in ("planning", "feedback"):
                return self._store_plan(ctx, args)
            return {"error": "tool is not available in this campaign mode"}
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}

    def _remember(self, ctx, args):
        _object(args, ("id", "goal", "target", "hypotheses", "status"), ("id", "goal"))
        ident = _text(args["id"], "id", 64, token=True)
        goal = _text(args["goal"], "goal", 2048)
        old = self._tasks.get(ident, {})
        target = args.get("target", old.get("target"))
        if target is not None:
            _text(target, "target", 256)
        hypotheses = _array(args.get("hypotheses", old.get("hypotheses", [])), "hypotheses", 16)
        for hypothesis in hypotheses:
            _text(hypothesis, "hypothesis", 2048)
        status = args.get("status", old.get("model_status", "active"))
        if status not in ("active", "completed", "abandoned"):
            raise ValueError("status must be active, completed, or abandoned")
        if ident not in self._tasks and len(self._tasks) >= 16:
            raise ValueError("task capacity reached (16); update an existing task")
        identity_changed = bool(old) and (goal != old["goal"] or target != old["target"])
        item = {"id": ident, "goal": goal, "target": target, "hypotheses": list(hypotheses),
                "model_status": status, "observed_status": (
                    "unverified" if identity_changed else old.get("observed_status", "unverified")),
                "source": "model", "created_tick": old.get("created_tick", ctx.round),
                "updated_tick": ctx.round}
        if identity_changed:
            # A reused ID is not evidence that the new goal/target was achieved.
            # Global outcomes retain their original action/target provenance.
            self._plans.pop(ident, None)
            self._plan_cursors.pop(ident, None)
        self._tasks[ident] = item
        return {"task": deepcopy(item)}

    def _store_plan(self, ctx, args):
        _object(args, ("task_id", "steps"), ("task_id", "steps"))
        ident = _text(args["task_id"], "task_id", 64, token=True)
        if ident not in self._tasks:
            raise ValueError("remember_task must establish task_id first")
        supplied = _array(args["steps"], "steps", 32, minimum=1)
        steps = []
        ids = set()
        for raw in supplied:
            _object(raw, ("id", "unit", "action", "target", "depends_on", "requires"),
                    ("id", "unit", "action", "target", "depends_on"))
            sid = _text(raw["id"], "step id", 64, token=True)
            if sid in ids:
                raise ValueError("duplicate step id")
            ids.add(sid)
            unit = _text(raw["unit"], "unit", 64, token=True)
            if unit not in ctx.decidable:
                raise ValueError("unknown unit: use a slot in the legal context")
            action = raw["action"]
            if action == "Sleep":
                raise ValueError("Sleep has no observed lifecycle completion; wait using the final decision/wake instead")
            if not isinstance(action, str) or action not in _RED_ACTIONS:
                raise ValueError("unknown RED action")
            _target(action, raw["target"])
            depends = _array(raw["depends_on"], "depends_on", 32)
            for dep in depends:
                _text(dep, "dependency id", 64, token=True)
            if len(set(depends)) != len(depends):
                raise ValueError("duplicate dependency id")
            requires = _array(raw.get("requires", []), "requires", 8)
            checked = []
            for requirement in requires:
                _object(requirement, ("kind", "target"), ("kind", "target"))
                if requirement["kind"] not in _CONDITIONS:
                    raise ValueError("unknown condition kind")
                _text(requirement["target"], "condition target", 256)
                if requirement["kind"] == "discovered_ip":
                    _target("ExploitRemoteService", requirement["target"])
                elif requirement["kind"] == "discovered_subnet":
                    _target("DiscoverRemoteSystems", requirement["target"])
                else:
                    _text(requirement["target"], "condition target", 256, token=True)
                checked.append(dict(requirement))
            steps.append({"id": sid, "unit": unit, "action": action, "target": raw["target"],
                          "depends_on": list(depends), "requires": checked, "requirements": [],
                          "duration": action_manual(action)["duration"], "status": "pending",
                          "validity": "not_invalidated", "completed_tick": None,
                          "submitted_tick": None})
        if any(dep not in ids for s in steps for dep in s["depends_on"]):
            raise ValueError("unknown dependency id")
        # Stable topological order; the input order breaks ties. A previous step
        # on the same unit becomes an explicit effective dependency, so a model
        # cannot spend the same slot twice in the same tick.
        ordered = []
        remaining = list(steps)
        done = set()
        previous = {}
        while remaining:
            ready = next((s for s in remaining if set(s["depends_on"]) <= done), None)
            if ready is None:
                raise ValueError("dependency cycle")
            ready["serialized_after"] = previous.get(ready["unit"])
            effective = list(ready["depends_on"])
            if ready["serialized_after"] and ready["serialized_after"] not in effective:
                effective.append(ready["serialized_after"])
            ready["effective_depends_on"] = effective
            previous[ready["unit"]] = ready["id"]
            ordered.append(ready)
            done.add(ready["id"])
            remaining.remove(ready)
        candidate = {"task_id": ident, "created_tick": ctx.round, "updated_tick": ctx.round,
                     "status": "active", "steps": ordered,
                     "timing_basis": "optimistic lower bound using declared durations; busy slots, "
                                     "unknown prerequisites and future legality can delay or prevent execution",
                     "outcome_guaranteed": False}
        self._refresh_plan(ctx, candidate)
        self._plans[ident] = candidate
        self._plan_cursors[ident] = len(ctx.own_log)
        self._tasks[ident]["observed_status"] = "plan_pending"
        return {"plan": deepcopy(candidate)}

    def _condition(self, ctx, requirement, unit):
        kind, target = requirement["kind"], requirement["target"]
        if kind in ("session", "root"):
            by_unit = getattr(ctx, "red_sessions_by_unit", {})
            data = by_unit.get(unit, {}) if by_unit else ctx.holdings
            values = data.get("sessions" if kind == "session" else "root", ())
        else:
            key = {"discovered_host": "hosts", "discovered_ip": "ips", "discovered_subnet": "subnets"}[kind]
            values = ctx.holdings.get("discovered", {}).get(key, ())
        return target in values

    def _refresh_plan(self, ctx, plan):
        by_id = {s["id"]: s for s in plan["steps"]}
        for step in plan["steps"]:
            old = {(r["kind"], r["target"]): r["state"] for r in step["requirements"]}
            step["requirements"] = [
                {**r, "state": "satisfied" if self._condition(ctx, r, step["unit"]) else (
                    "unsatisfied" if old.get((r["kind"], r["target"])) in ("satisfied", "unsatisfied")
                    else "unknown"), "observed_tick": ctx.round}
                for r in step["requires"]]
            deps = [by_id[d] for d in step["effective_depends_on"]]
            frontier = step["status"] == "pending" and step["validity"] != "invalidated" and all(
                d["status"] == "succeeded" and d["validity"] != "invalidated" for d in deps) and all(
                r["state"] == "satisfied" for r in step["requirements"])
            step["conditional"] = not frontier
            step["current_legal"] = ((step["action"], step["target"]) in ctx.decidable.get(step["unit"], ())) if frontier else None
            step["ready"] = frontier and step["current_legal"] is True and all(
                r["state"] == "satisfied" for r in step["requirements"])
            if step["status"] == "succeeded":
                finish = step["completed_tick"]
                start = step["submitted_tick"]
                if start is None:
                    start = max(plan["created_tick"], finish - step["duration"])
            else:
                start = max([ctx.round] + [d["earliest_finish"] for d in deps])
                if step["status"] == "in_progress" and step["submitted_tick"] is not None:
                    start = step["submitted_tick"]
                    finish = max(ctx.round, start + step["duration"])
                else:
                    finish = start + step["duration"]
            step["earliest_start"] = start
            step["earliest_finish"] = finish
        plan["earliest_finish"] = max(s["earliest_finish"] for s in plan["steps"])
        plan["within_horizon"] = plan["earliest_finish"] <= ctx.horizon
        plan["updated_tick"] = ctx.round

    def _event(self, kind, tick, task_ids=(), **details):
        ids = sorted(set(task_ids))
        self._events.append({"kind": kind, "tick": tick, "source": "legal_context", "task_ids": ids, **details})
        del self._events[:-128]
        if self.mode == "feedback" and any(self._tasks[i]["model_status"] != "abandoned" for i in ids):
            self._pending = True

    def _invalidate(self, ident, step_ids, reason):
        plan = self._plans.get(ident)
        if plan:
            invalid = set(step_ids)
            for step in plan["steps"]:
                if step["id"] in invalid or any(d in invalid for d in step["effective_depends_on"]):
                    invalid.add(step["id"])
                    step["validity"] = "invalidated"
                    step["invalidated_reason"] = reason
                    if step["status"] not in ("succeeded", "failed"):
                        step["status"] = "blocked"
            if invalid:
                plan["status"] = "needs_replan"
        self._tasks[ident]["observed_status"] = "needs_replan"

    def _observe_entry(self, index, entry):
        related = {ident for ident, task in self._tasks.items()
                   if task["target"] is not None and task["target"] == entry.target}
        for ident, plan in self._plans.items():
            if index < self._plan_cursors[ident] or entry.round < plan["created_tick"]:
                continue
            by_id = {s["id"]: s for s in plan["steps"]}
            matching = next((s for s in plan["steps"]
                             if s["unit"] == entry.unit and s["action"] == entry.action and s["target"] == entry.target
                             and s["status"] in ("pending", "in_progress") and s["validity"] != "invalidated"
                             and all(by_id[d]["status"] == "succeeded" and by_id[d]["validity"] != "invalidated"
                                     for d in s["effective_depends_on"])), None)
            if matching is not None:
                related.add(ident)
                if entry.result in ("submitted", "in_progress"):
                    matching["status"] = "in_progress"
                    if entry.result == "submitted":
                        matching["submitted_tick"] = entry.round
                elif entry.result in _TERMINAL:
                    matching["completed_tick"] = entry.round
                    if entry.result == "succeeded":
                        matching["status"] = "succeeded"
                        matching["validity"] = "observed_success"
                        if all(s["status"] == "succeeded" for s in plan["steps"]):
                            plan["status"] = "completed"
                            self._tasks[ident]["observed_status"] = "plan_completed"
                    else:
                        matching["status"] = "failed"
                        self._invalidate(ident, [matching["id"]], entry.result)
            # The legal own log reports loss even when no exact planned action
            # can be matched. A lost unit invalidates that unit's dependencies.
            if entry.result == "session_lost":
                affected = [s["id"] for s in plan["steps"] if s["unit"] == entry.unit]
                if affected:
                    related.add(ident)
                    self._invalidate(ident, affected, "session_lost")
            elif entry.result == "invalid":
                # Compiler rejection discards the requested action/target in
                # the native log. Flag pending work on this unit for review;
                # never invent which planned action ran or failed.
                affected = [s["id"] for s in plan["steps"] if s["unit"] == entry.unit
                            and s["status"] in ("pending", "in_progress")]
                if affected:
                    related.add(ident)
                    self._invalidate(ident, affected, "invalid_submission_on_unit")
        if entry.result in _TERMINAL:
            row = {"tick": entry.round, "unit": entry.unit, "action": entry.action,
                   "target": entry.target, "result": entry.result, "observed": _values(entry.observed)}
            self._facts["outcomes"].append(row)
            del self._facts["outcomes"][:-128]
            for ident in related:
                if entry.result != "succeeded":
                    self._tasks[ident]["observed_status"] = "needs_replan"
                elif ident not in self._plans:
                    self._tasks[ident]["observed_status"] = "action_succeeded"
            self._event("action_" + entry.result, entry.round, related, **{k: v for k, v in row.items() if k != "tick"})

    def _loss(self, ctx, kind, target, unit=None):
        affected = set()
        observed_ips = set(((ctx.holdings.get("discovered") or {}).get("host_to_ips") or {})
                           .get(target, ()))
        for ident, task in self._tasks.items():
            matches = task["target"] == target
            plan = self._plans.get(ident)
            ids = []
            if plan:
                for step in plan["steps"]:
                    target_match = step["target"] == target or step["target"] in observed_ips or (
                        step["action"] == "Withdraw" and step["target"].split("|")[-1] == target)
                    requirement_match = any(r["target"] == target and r["kind"] in ("session", "root")
                                            for r in step["requires"])
                    if (unit is not None and step["unit"] == unit) or target_match or requirement_match:
                        ids.append(step["id"])
                if matches and not ids:
                    # A task targeted the lost holding but no step carries either the hostname
                    # or one of its legally-observed IPs. Conservatively invalidate the plan.
                    ids = [s["id"] for s in plan["steps"]]
            if matches or ids:
                affected.add(ident)
                self._invalidate(ident, ids, kind + "_lost")
        # Aggregate holdings and per-unit holdings can report the same access loss in the same
        # tick. Root loss commonly implies session loss too. Keep one auditable event per
        # (tick,target), with the observed loss kinds and units unioned, rather than presenting
        # several copies of one state transition to the model.
        existing = next((event for event in reversed(self._events)
                         if event.get("kind") == "access_lost"
                         and event.get("tick") == ctx.round
                         and event.get("target") == target), None)
        if existing is None:
            self._event("access_lost", ctx.round, affected, target=target,
                        lost_kinds=[kind], units=[] if unit is None else [unit],
                        aggregate_loss_observed=unit is None)
        else:
            existing["task_ids"] = sorted(set(existing["task_ids"]) | affected)
            existing["lost_kinds"] = sorted(set(existing["lost_kinds"]) | {kind})
            if unit is not None:
                existing["units"] = sorted(set(existing["units"]) | {unit})
            else:
                existing["aggregate_loss_observed"] = True
            if (self.mode == "feedback"
                    and any(self._tasks[i]["model_status"] != "abandoned"
                            for i in existing["task_ids"])):
                self._pending = True

    def observe(self, ctx: LegalContext):
        """Process each appended own-log row once, including ticks with no legal slot."""
        if self.mode == "off" or ctx.team != "red":
            return False
        self._seed_public_objectives(ctx)
        for index in range(self._cursor, len(ctx.own_log)):
            self._observe_entry(index, ctx.own_log[index])
        self._cursor = max(self._cursor, len(ctx.own_log))
        previous = self._facts["holdings"]
        holdings = {"sessions": _values(ctx.holdings.get("sessions", ())),
                    "root": _values(ctx.holdings.get("root", ())),
                    "discovered": {k: _values(ctx.holdings.get("discovered", {}).get(k, ()))
                                   for k in ("hosts", "ips", "subnets")}}
        by_unit = {unit: {kind: _values(data.get(kind, ())) for kind in ("sessions", "root")}
                   for unit, data in sorted(getattr(ctx, "red_sessions_by_unit", {}).items())[:512]}
        if self._tick is not None:
            for key, label in (("sessions", "session"), ("root", "root")):
                for target in sorted(set(previous[key]) - set(ctx.holdings.get(key, ()))):
                    self._loss(ctx, label, target)
            for unit, prior in self._facts["red_sessions_by_unit"].items():
                for key, label in (("sessions", "session"), ("root", "root")):
                    current = getattr(ctx, "red_sessions_by_unit", {}).get(unit, {}).get(key, ())
                    for target in sorted(set(prior[key]) - set(current)):
                        self._loss(ctx, label, target, unit)
        self._facts["holdings"] = holdings
        self._facts["red_sessions_by_unit"] = by_unit
        for ident, plan in self._plans.items():
            before = {s["id"]: [(r["kind"], r["target"], r["state"]) for r in s["requirements"]]
                      for s in plan["steps"]}
            self._refresh_plan(ctx, plan)
            for step in plan["steps"]:
                after = [(r["kind"], r["target"], r["state"]) for r in step["requirements"]]
                if before[step["id"]] != after:
                    self._event("prerequisite_changed", ctx.round, [ident], step_id=step["id"], requirements=deepcopy(step["requirements"]))
                    if any(r["state"] == "unsatisfied" for r in step["requirements"]):
                        self._invalidate(ident, [step["id"]], "prerequisite_lost")
            # Invalidation changes the actionable frontier computed above.
            self._refresh_plan(ctx, plan)
        self._tick = ctx.round
        return self._pending


def campaign_instructions(mode):
    _mode(mode)
    if mode == "off":
        return ""
    sections = [
        "## CAMPAIGN ASSISTANCE",
        f"Mode: {mode}.",
        ("The automatic ledger is episode-local and capped at 8000 JSON characters. It contains "
         "task previews, ordered plan steps, observed feedback and omission counts. campaign_state "
         "returns the full bounded ledger."),
        ("remember_task stores model-authored goals, hypotheses and model_status. Omitted optional "
         "fields retain their previous values; target=null clears the target. A changed goal or "
         "target removes the prior plan and resets observed_status."),
        ("facts, action outcomes and current holdings are observation-derived fields. Task goals, "
         "hypotheses and model_status are model-authored fields. Submitted and succeeded are "
         "distinct outcome states."),
    ]
    if mode in ("planning", "feedback"):
        sections.append(
            "check_campaign_plan stores a conditional DAG for an existing task. Each step has id, "
            "unit, action, target and depends_on; optional requires entries use session, root, "
            "discovered_host, discovered_ip or discovered_subnet. Same-unit steps are serialized. "
            "The dependency-free frontier is compared with current legal pairs; later steps remain "
            "conditional. ready also requires observed prerequisites. Timing is the sum of declared "
            "durations. A stored plan does not submit an action. Sleep is not a plan step."
        )
    if mode == "feedback":
        sections.extend((
            ("Two program_public_objective records, one for each public scoring target, exist from "
             "tick zero. Their task IDs are stable."),
            ("Observed failures, access losses and prerequisite changes enter the feedback ledger. "
             "Feedback remains pending until a decision is delivered. Invalidated successful steps "
             "remain historical records and do not satisfy later dependencies."),
            ("For this profile, final decisions contain exactly one entry for every RED unit in the "
             "current legal options. Sleep with target=null is a valid entry."),
        ))
    return "\n".join(sections)


def campaign_tool_schemas(mode):
    _mode(mode)
    if mode == "off":
        return []
    def obj(properties, required=()):
        return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
    def string(limit=256):
        return {"type": "string", "minLength": 1, "maxLength": limit}
    def tool(name, description, parameters):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
    result = [tool("campaign_state", "Read bounded task memory, observed facts and conditional plans.", obj({})),
              tool("remember_task", "Store or update a model goal/hypotheses; status is a model claim, not observed success.", obj({
                  "id": string(64), "goal": string(2048), "target": {"type": ["string", "null"], "maxLength": 256},
                  "hypotheses": {"type": "array", "maxItems": 16, "items": string(2048)},
                  "status": {"type": "string", "enum": ["active", "completed", "abandoned"]}}, ("id", "goal")))]
    if mode in ("planning", "feedback"):
        condition = obj({"kind": {"type": "string", "enum": list(_CONDITIONS)}, "target": string()}, ("kind", "target"))
        step = obj({"id": string(64), "unit": string(64),
                    "action": {"type": "string", "enum": list(_RED_ACTIONS)},
                    "target": {"type": ["string", "null"], "maxLength": 256},
                    "depends_on": {"type": "array", "maxItems": 32, "items": string(64)},
                    "requires": {"type": "array", "maxItems": 8, "items": condition}},
                   ("id", "unit", "action", "target", "depends_on"))
        result.append(tool("check_campaign_plan", "Validate and store a conditional plan. Only its pending frontier is checked for present legality; no outcomes are predicted.",
                           obj({"task_id": string(64), "steps": {"type": "array", "minItems": 1, "maxItems": 32, "items": step}}, ("task_id", "steps"))))
    return result

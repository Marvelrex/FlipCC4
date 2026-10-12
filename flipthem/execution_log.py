"""Passive per-tick execution + business recorder (execution-ledger design).

Consumes the read-only `FlipThemCC4.step_observer` events (see wrapper.py) AFTER the mechanism has
already billed and settled a tick. It NEVER submits an action, reads a policy's private state, feeds
anything back into an observation/prompt, or consumes the RNG: with the observer detached the game
is byte-for-byte unchanged (asserted by the testbed smoke).

Two records, kept apart:

* the ACTION LEDGER -- one row per action that resolved this tick (executed or filtered), taken from
  the wrapper's own billing determination, plus, at finalisation, rows for actions still pending at
  the horizon. `submitted_tick` is paired in from each side's own_log when it is supplied; when it
  is not, it is left NA rather than guessed (design: missing legacy fields are marked NA).

* the GREEN BUSINESS log -- per-tick counts of the native Green mission actions (GreenAccessService,
  GreenLocalWork), classified submitted / executed / success / fail from Green's OWN legal
  observation. Precise "blocked_by_zone_rule" attribution needs a native-source diagnostic the
  design defers , so it is reported as unavailable here; only success/failure counts and
  their timing (which a later report can correlate with a block) are recorded. A 0 executed
  denominator yields a null rate, never a flattering 0.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from flipthem.llm.context import (
    RESULT_SUBMITTED, RESULT_IN_PROGRESS, RESULT_SUCCEEDED, RESULT_FAILED, RESULT_SESSION_LOST,
)

EXECUTION_LOG_SCHEMA_VERSION = "exec-ledger-v1"
GREEN_ACTIONS = ("GreenAccessService", "GreenLocalWork")
_COMPLETION_RESULTS = (RESULT_SUCCEEDED, RESULT_FAILED, RESULT_SESSION_LOST)


class GreenBusiness:
    """Per-tick tally of the two native Green mission actions from Green's own observation."""

    def __init__(self):
        # action -> counters
        self._c = {name: {"submitted": 0, "executed": 0, "success": 0, "fail": 0,
                          "in_progress": 0, "unknown": 0} for name in GREEN_ACTIONS}
        self._green_slots = None

    def observe(self, env):
        if self._green_slots is None:
            self._green_slots = sorted(
                s for s in env.environment_controller.agent_interfaces if s.startswith("green"))
        for slot in self._green_slots:
            taken = env.get_last_action(slot)
            taken = taken[0] if isinstance(taken, list) else taken
            name = type(taken).__name__ if taken is not None else None
            if name not in self._c:
                continue
            self._c[name]["submitted"] += 1
            obs = env.get_observation(slot)
            success = str(obs.get("success"))
            if success == "IN_PROGRESS":
                self._c[name]["in_progress"] += 1
                continue
            if obs.get("action") is not taken:
                # Filtered before executing (parallel to the blue/red identity check).
                self._c[name]["unknown"] += 1
                continue
            self._c[name]["executed"] += 1
            if success == "TRUE":
                self._c[name]["success"] += 1
            elif success == "FALSE":
                self._c[name]["fail"] += 1
            else:
                self._c[name]["unknown"] += 1

    def summary(self) -> dict:
        out = {}
        for name, c in self._c.items():
            executed = c["executed"]
            out[name] = {
                **c,
                # Rate over ACTIONS THAT ACTUALLY EXECUTED; 0 denominator -> null, never 0.0.
                "success_rate_of_executed": (c["success"] / executed) if executed else None,
                "blocked_by_zone_rule": None,   # NA: precise attribution deferred (design the design)
            }
        out["_note"] = ("Success rate is over the CURRENT CC4 implementation's business-action "
                        "executions, not physical service availability; block attribution is NA.")
        return out


class ExecutionRecorder:
    """One recorder per episode; attach `.on_step` as `game.step_observer` before the drive loop."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.rows: list = []
        self._decision_seq = 0
        self._last_tick = 0
        # slot -> True while an action is mid-flight (in_progress seen, no completion yet).
        self.green = GreenBusiness()

    def on_step(self, event: dict, env) -> None:
        tick = int(event["tick"])
        self._last_tick = tick
        for f in event["slots"]:
            # A resolved, real action this tick: executed or filtered (both are billing-relevant).
            # Sleep no-ops and mid-flight placeholders carry no ledger information.
            if f["in_progress"] or f["action"] in (None, "Sleep"):
                continue
            self._decision_seq += 1
            self.rows.append({
                "episode_id": self.episode_id,
                "decision_id": self._decision_seq,
                "slot": f["slot"],
                "action": f["action"],
                "target": f["target"],
                "submitted_tick": None,          # paired from own_log at finalise, else NA
                "completed_tick": tick,
                "executed": f["executed"],
                "success": f["success"],
                "billed_ft_cost": f["billed_ft_cost"],
                "pending_at_horizon": False,
            })
        self.green.observe(env)

    def finalize(self, own_logs: dict | None = None, horizon: int | None = None) -> None:
        """Enrich `submitted_tick` and append pending-at-horizon rows from each side's own_log.

        `own_logs`: {"blue": tuple[LogEntry], "red": tuple[LogEntry]} (either may be omitted). When a
        side's log is absent, its rows keep `submitted_tick=None` (NA) and contribute no pending
        rows -- the recorder never fabricates a submission time it cannot prove."""
        own_logs = own_logs or {}
        horizon = horizon if horizon is not None else self._last_tick
        # FIFO submission queues per (slot, action); completions consume the earliest submission.
        submissions: dict = {}
        completed_keys: dict = {}
        for team_log in own_logs.values():
            for e in team_log:
                if e.unit is None or e.action is None:
                    continue
                key = (e.unit, e.action)
                if e.result == RESULT_SUBMITTED:
                    submissions.setdefault(key, []).append(e.round)

        # Pair completion rows (in decision order) to the earliest not-yet-consumed submission.
        consumed: dict = {}
        for row in self.rows:
            key = (row["slot"], row["action"])
            queue = submissions.get(key, [])
            idx = consumed.get(key, 0)
            if idx < len(queue):
                row["submitted_tick"] = queue[idx]
                consumed[key] = idx + 1
        # Any submission never consumed by a completion row = pending at the horizon.
        for (slot, action), rounds in submissions.items():
            idx = consumed.get((slot, action), 0)
            for submitted_round in rounds[idx:]:
                self._decision_seq += 1
                self.rows.append({
                    "episode_id": self.episode_id,
                    "decision_id": self._decision_seq,
                    "slot": slot,
                    "action": action,
                    "target": None,
                    "submitted_tick": submitted_round,
                    "completed_tick": None,
                    "executed": False,
                    "success": None,
                    "billed_ft_cost": 0.0,
                    "pending_at_horizon": True,
                })

    def summary(self) -> dict:
        billed_by_slot: dict = {}
        counts_by_action: dict = {}
        pending = 0
        executed = 0
        filtered = 0
        for row in self.rows:
            billed_by_slot[row["slot"]] = billed_by_slot.get(row["slot"], 0.0) + row["billed_ft_cost"]
            counts_by_action[row["action"]] = counts_by_action.get(row["action"], 0) + 1
            if row["pending_at_horizon"]:
                pending += 1
            elif row["executed"]:
                executed += 1
            else:
                filtered += 1
        return {
            "schema_version": EXECUTION_LOG_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "rows": len(self.rows),
            "executed": executed,
            "filtered": filtered,
            "pending_at_horizon": pending,
            "billed_ft_cost_by_slot": billed_by_slot,
            "counts_by_action": counts_by_action,
            "green_business": self.green.summary(),
        }

    def write_jsonl(self, path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                line = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                stream.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))
        return digest.hexdigest()

    def write_summary(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")

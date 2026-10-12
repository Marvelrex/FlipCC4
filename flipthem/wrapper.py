"""FlipThemCC4: the mechanism wrapper (spec the design-7, 13).

Owns a real CybORG CC4 env, the two-resource map, control settlement, and completion billing. The
runner (or a scripted policy) supplies per-slot actions; the wrapper submits them, settles the
declared FT control value each tick, bills completed actions, and exposes each team's own legal
observations. The FT reward is never placed into an observation.

Three API points reconciled against the live CybORG source (`CybORG/Simulator/SimulationController.py`)
and verified empirically (see the wrapper tests):

1. `env.get_observation(slot)` returns `environment_controller.get_last_observation(slot).data`, a
   plain dict. While a slot's action is mid-flight, `SimulationController.step` appends
   `Observation(TernaryEnum.IN_PROGRESS)` as the *first* entry of that tick's `ObservationSet`, and
   `Observation.combine_obs` explicitly skips the "success"/"action" keys when folding in later
   entries (e.g. the Sleep-execution observation appended the same tick) -- so the combined
   observation's `success` field reliably stays `TernaryEnum.IN_PROGRESS` for the whole in-progress
   window and flips away from it exactly on the completion tick. `_in_progress` below matches this.
2. `env.get_last_action(slot)` returns `environment_controller.action[slot]`, which is always a
   `list` with exactly one `Action` (never a bare `Action`): the real action object on its completion
   tick, or a `Sleep()` placeholder on every tick it is still mid-flight (`SimulationController.step`
   assigns `self.action = actions_to_execute` -- a dict of one-action lists -- before sorting). The
   skeleton's `taken[0] if isinstance(taken, list) else taken` unwrap already matches this shape.
3. `get_last_action` is a PRE-filter record: `self.action = actions_to_execute` (point 2) is set
   BEFORE `sort_action_order` -> `filter_actions` runs, so it cannot tell an action that genuinely
   executed from one `filter_actions` dropped (e.g. its session vanished by the completion tick).
   The combined observation's `'action'` entry can: it is set only for actions that reached
   `execute_action`, and (per point 1's `combine_obs` behaviour) it is kept from the FIRST entry
   appended that tick. A filtered slot's first -- and only -- entry that tick instead comes from
   its unconditional per-tick end-turn default (Monitor for blue, RedSessionCheck for red; see
   `EnterpriseScenarioGenerator`), a freshly-constructed instance every tick. So `step` below bills
   a slot's own submission only when the observation's `'action'` IS (identity) `get_last_action`'s
   record -- see `step`'s billing loop and
   tests/test_wrapper.py::test_filtered_completing_action_is_billed_zero.
"""
from __future__ import annotations

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import Sleep
from CybORG.Shared.BlueRewardMachine import BlueRewardMachine

from flipthem.resources import build_cc4_env, discover_resources
from flipthem.settlement import settle_control, SYMMETRIC_VALUES
from flipthem.billing import Billing, ft_cost


def _action_target_repr(action) -> str | None:
    """A compact target string for the passive execution ledger only (never used by the mechanism).
    Mirrors the fields the compiler/agent use for a logical target, without importing frozen code."""
    if action is None:
        return None
    if hasattr(action, "from_subnet") and hasattr(action, "to_subnet"):
        return f"{action.from_subnet}->{action.to_subnet}"
    if hasattr(action, "hostname") and hasattr(action, "ip_address"):
        return f"{action.ip_address}|{action.hostname}"
    for attr in ("hostname", "subnet", "ip_address"):
        if hasattr(action, attr):
            return str(getattr(action, attr))
    return None


class FlipThemCC4:
    def __init__(self, seed, values=None, steps=100, mode="ft", red_agent_class=None):
        assert mode in ("ft", "c4")
        self.seed = seed
        self.values = dict(values or SYMMETRIC_VALUES)
        self.T = steps
        self.mode = mode
        self.red_agent_class = red_agent_class
        self.env = None
        # PASSIVE, read-only observer (the scripted-baseline design item 3). When set, it
        # is called ONCE at the very end of each `step()` with the per-slot execution facts and the
        # settlement this tick already produced -- after ALL state mutation and billing, reading
        # only values already computed. It never influences actions, observations, the RNG, billing
        # or the returned payoff: with it left None (the default) `step()` is byte-for-byte the
        # behaviour it had before. It exists so an evaluation-side recorder can journal the ledger
        # without a second drive of the environment. See flipthem/execution_log.py.
        self.step_observer = None

    def reset(self):
        self.env = build_cc4_env(self.seed, steps=self.T, red_agent_class=self.red_agent_class)
        discover_resources(self.env)
        agents = self.env.environment_controller.agent_interfaces
        self.blue_slots = [a for a in agents if a.startswith("blue_agent_")]
        self.red_slots = [a for a in agents if a.startswith("red_agent_")]
        self.billing = Billing()
        self._v_red_sum = 0.0
        self._v_blue_sum = 0.0
        self._sum_v_sum = 0.0
        self._c4_blue_sum = 0.0
        self._tick = 0
        self.counts = {"llm_calls": 0, "decisions": 0, "completed_actions": 0}
        return self

    def team_observation(self, team):
        slots = self.blue_slots if team == "blue" else self.red_slots
        return {s: self.env.get_observation(s) for s in slots}

    def _in_progress(self, slot):
        obs = self.env.get_observation(slot)
        return obs.get("success", None) == TernaryEnum.IN_PROGRESS

    def step(self, blue_actions, red_actions=None):
        actions = dict(blue_actions)
        if red_actions is not None:
            actions.update(red_actions)
        # Enforce "submit only to idle slots" (brief's implementer note). CybORG itself already
        # discards a resubmitted real action for *execution* purposes on a busy slot -- it keeps
        # running the original in-flight action and forces a Sleep placeholder there -- so a
        # resubmission is harmless for FT billing (get_last_action returns that forced Sleep).
        # But native per-tick action_cost (SimulationController.py: `action_cost =
        # sum(actions.get(agent, Action()).cost for agent in self.team[team_name])`) sums cost
        # over the RAW SUBMITTED actions dict, not the internally-tracked one-per-slot record --
        # so in mode="c4" a caller that keeps resubmitting a real action to a busy slot would
        # silently inflate action_cost (and thus U_B_c4/U_R_c4) every such tick even though
        # nothing re-executes. Normalize here, before submission, so this can never happen in
        # any mode. _in_progress(slot) reflects the most recently completed tick's observation,
        # i.e. exactly "is this slot still busy as we go into this submission" -- the same notion
        # of busy that makes CybORG itself discard a resubmission for execution.
        for slot in self.blue_slots + self.red_slots:
            if slot in actions and not isinstance(actions[slot], Sleep) and self._in_progress(slot):
                actions[slot] = Sleep()
        # count decisions actually issued (non-Sleep) this tick
        self.counts["decisions"] += sum(1 for a in actions.values() if not isinstance(a, Sleep))
        obs, rew, done, info = self.env.parallel_step(actions)

        # Bill both teams from what each slot ACTUALLY EXECUTED this tick (spec the design: a
        # filtered action is charged 0, same as one still in progress). `get_last_action` is the
        # PRE-filter record (SimulationController.py sets `self.action = actions_to_execute`
        # before `sort_action_order` -> `filter_actions` can drop an entry whose session vanished
        # by the time it is ready to run), so it cannot by itself distinguish a filtered action
        # from one that executed. The combined observation can: `filtered_obs.data['action'] =
        # action` (SimulationController.step) is only ever set for actions that reached
        # `execute_action`, and `Observation.combine_obs` keeps the FIRST observation's
        # 'action'/'success' verbatim when folding in later entries -- so a slot's own submission
        # truly executed this tick iff the combined observation's 'action' IS (identity, not
        # equality: every scenario agent has an unconditional per-tick end-turn default -- Monitor
        # for blue, RedSessionCheck for red, see EnterpriseScenarioGenerator -- that reconstructs
        # a fresh instance every tick and surfaces as the FIRST/only entry precisely when the real
        # submission was filtered out) that exact object. Verified empirically in
        # tests/test_wrapper.py::test_filtered_completing_action_is_billed_zero.
        exec_facts = []   # PASSIVE snapshot for step_observer; never read back by the mechanism
        for slot in self.blue_slots + self.red_slots:
            taken = self.env.get_last_action(slot)
            taken = taken[0] if isinstance(taken, list) else taken
            ip = self._in_progress(slot)
            executed = (not ip and taken is not None
                        and self.env.get_observation(slot).get("action") is taken)
            if executed and not isinstance(taken, Sleep):
                self.counts["completed_actions"] += 1
            self.billing.on_step(slot, taken if executed else Sleep(), in_progress=ip)
            if self.step_observer is not None:
                # Only reads locals/observations already computed above -- no extra env mutation.
                billed = ft_cost(taken) if (executed and not isinstance(taken, Sleep)) else 0.0
                exec_facts.append({
                    "slot": slot,
                    "in_progress": bool(ip),
                    "executed": bool(executed),
                    "action": type(taken).__name__ if taken is not None else None,
                    "target": _action_target_repr(taken),
                    "success": str(self.env.get_observation(slot).get("success")),
                    "billed_ft_cost": billed,
                })

        s = settle_control(self.env, self.values)
        self._v_red_sum += s["v_red"]
        self._v_blue_sum += s["v_blue"]
        self._sum_v_sum += s["v_red"] + s["v_blue"]
        if self.mode == "c4":
            calc = BlueRewardMachine("Blue").calculate_simulation_reward(self.env.environment_controller)
            # native per-tick team reward lives on environment_controller.reward["Blue"], keyed by
            # TEAM name; `rew` from parallel_step is keyed by AGENT (slot) name, so it is read here
            # instead of `rew` (verified live: rew["blue_agent_1"] is environment_controller.reward["Blue"]).
            blue_reward = self.env.environment_controller.reward.get("Blue", {})
            action_cost = blue_reward.get("action_cost", 0)
            self._c4_blue_sum += calc + action_cost
        self._tick += 1
        if self.step_observer is not None:
            # Called last: every state mutation and billing for this tick is already done, so the
            # observer can only read, never steer. Exceptions are the recorder's own problem and
            # are not swallowed here (a silent recorder is worse than a loud one), but the recorder
            # is written to never raise on a normal tick.
            self.step_observer({"tick": self._tick, "x": dict(s["x"]),
                                "v_red": s["v_red"], "v_blue": s["v_blue"],
                                "slots": exec_facts}, self.env)
        return {"tick": self._tick, "x": s["x"], "v_red_tick": s["v_red"],
                "v_blue_tick": s["v_blue"], "done": self._tick >= self.T}

    def payoff(self):
        self.billing.finalise()
        cost_R = self.billing.total(self.red_slots)
        cost_B = self.billing.total(self.blue_slots)
        T = max(1, self._tick)
        out = {
            "U_R": (self._v_red_sum - cost_R) / T,
            "U_B": (self._v_blue_sum - cost_B) / T,
            "sum_v": self._sum_v_sum / T,
            "cost_R": cost_R, "cost_B": cost_B, "counts": dict(self.counts),
        }
        if self.mode == "c4":
            out["U_B_c4"] = self._c4_blue_sum / T
            out["U_R_c4"] = -out["U_B_c4"]
        return out

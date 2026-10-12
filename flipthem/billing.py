"""Completion-based FT billing with a non-negative cost (spec the design).

k(a) = -native action.cost, so Restore->1, InvalidAction->0.1, else 0. An action is charged once, on
the tick it actually executes (its observation leaves IN_PROGRESS). Still-in-progress-at-episode-end
and filtered actions are charged 0. Native action_cost (read at submission) is never reused here.
"""
from __future__ import annotations

from CybORG.Simulator.Actions.Action import Sleep


def ft_cost(action) -> float:
    native = getattr(action, "cost", 0) or 0
    return max(0.0, -float(native))


class Billing:
    def __init__(self) -> None:
        self._charged: dict[str, float] = {}

    def on_step(self, slot: str, action_taken, in_progress: bool) -> None:
        """Called once per slot per tick. Charge on the completion tick (not in progress, real action)."""
        if in_progress:
            return
        if isinstance(action_taken, Sleep):
            return
        self._charged[slot] = self._charged.get(slot, 0.0) + ft_cost(action_taken)

    def finalise(self) -> None:
        """No-op: in-progress actions were never charged, which is the intended rule."""

    def total(self, side_slots) -> float:
        return sum(self._charged.get(s, 0.0) for s in side_slots)

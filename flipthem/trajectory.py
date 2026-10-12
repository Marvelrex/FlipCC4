"""Per-resource control trajectories (design per-resource occupancy, the time to
first scoring root, how many times control was lost, and how long re-entry took.

Everything here is a pure function over the saved vectors -- no CybORG, no env, no policy. The
runner captures `x` straight off `step()` (ground truth it observes itself; no policy ever hands
it over and no prompt ever sees it) and calls `summarise_control` once at the end.

**Censoring, not zero-filling.** "Never captured" is `first_red_tick = None` plus
`first_red_censored = True` at `censoring_tick` -- never 0 (which would read as "captured
immediately") and never a dropped row (which would bias the survival estimate by deleting exactly
the slowest cases). A final loss with no subsequent re-entry is likewise reported as one censored
re-entry rather than a completed one or nothing at all.
"""
from __future__ import annotations


def summarise_control(control_ticks, values: dict) -> dict:
    """`control_ticks`: the per-tick `x` vectors in order, as `step()` returned them.
    `values`: the declared resource -> v_s map (the resource SET comes from here, never from
    whatever keys the trace happens to carry -- a resource absent from every tick is still a
    resource that was never captured, and a stray trace key is not a resource)."""
    ticks = len(control_ticks)
    resources = {
        name: _summarise_one(float(v), [1 if t.get(name) else 0 for t in control_ticks], ticks)
        for name, v in values.items()
    }
    # "Time to first SCORING root": only v_s > 0 roots score, so a zero-valued root being taken
    # is not the event of interest.
    reached = [r["first_red_tick"] for r in resources.values()
               if r["value"] > 0.0 and r["first_red_tick"] is not None]
    first_scoring = min(reached) if reached else None
    return {
        "ticks": ticks,
        "censoring_tick": ticks,
        "resources": resources,
        "first_scoring_tick": first_scoring,
        "first_scoring_censored": first_scoring is None,
    }


def _summarise_one(value: float, series: list, ticks: int) -> dict:
    red_ticks = sum(series)
    # 0 ticks played -> occupancy is UNKNOWN (0/0), never a fabricated 0.0 that would read as
    # "red held nothing", the same unknown-is-null rule the cost recorder uses.
    occupancy = (red_ticks / ticks) if ticks else None
    first_red = next((i + 1 for i, x in enumerate(series) if x), None)   # 1-based, like `tick`

    gains = losses = 0
    reentry_times: list = []
    open_loss_tick = None   # a loss still waiting for its re-entry
    prev = 0                # blue holds every root before tick 1
    for i, x in enumerate(series):
        tick = i + 1
        if x and not prev:
            gains += 1
            if open_loss_tick is not None:
                reentry_times.append(tick - open_loss_tick)
                open_loss_tick = None
        elif prev and not x:
            losses += 1
            open_loss_tick = tick
        prev = x

    return {
        "value": value,
        "red_ticks": red_ticks,
        "occupancy_red": occupancy,
        "first_red_tick": first_red,
        "first_red_censored": first_red is None,
        "censoring_tick": ticks,
        "gains": gains,
        "losses": losses,
        "reentry_times": reentry_times,
        # A loss never followed by a gain: a real re-entry attempt that the horizon cut short.
        # Counted here rather than appended to reentry_times (it is not a completed duration) and
        # rather than discarded (that would silently drop the worst cases).
        "reentry_censored": 1 if open_loss_tick is not None else 0,
    }

"""Per-tick control settlement and the declared FT control-value (spec the design).

x_s(t)=1 iff red holds a privileged session on resource s at the settlement point (end of tick).
The controlling side accrues v_s that tick. Values are a declared, CC4-inspired parameterisation
(the per-event RIA magnitude reinterpreted as a per-tick control value), not a native reward.
"""
from __future__ import annotations

from flipthem.resources import OPERATIONAL_RESOURCES, compromise_of

Values = dict

SYMMETRIC_VALUES = {OPERATIONAL_RESOURCES[0]: 1.0, OPERATIONAL_RESOURCES[1]: 1.0}
ASYMMETRIC_VALUES = {OPERATIONAL_RESOURCES[0]: 10.0, OPERATIONAL_RESOURCES[1]: 1.0}


def settle_control(env, values: Values) -> dict:
    x, v_red, v_blue = {}, 0.0, 0.0
    for resource, v in values.items():
        red = compromise_of(env, resource) == "red"
        x[resource] = 1 if red else 0
        v_red += v if red else 0.0
        v_blue += 0.0 if red else v
    return {"x": x, "v_red": v_red, "v_blue": v_blue}

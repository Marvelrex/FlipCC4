"""Explicit JSON boundary for own-side tool data; no arbitrary object stringification."""
import math

from flipthem.llm.context import LogEntry


def tool_json(value):
    if isinstance(value, LogEntry):
        return tool_json({"round": value.round, "unit": value.unit, "action": value.action,
                          "target": value.target, "result": value.result,
                          "observed": value.observed})
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [tool_json(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: tool_json(v) for k, v in value.items()}
    raise TypeError(f"Unsupported tool JSON value: {type(value).__name__}")

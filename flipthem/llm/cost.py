"""Shared usage/cost helpers (review fix items 3 + 5): the single home for token counting and
price ($) computation, imported by BOTH `agent.py` (per-round, as decisions happen) and
`conversation.py` (assembling the separate per-match conversation JSON) -- previously duplicated
verbatim in both, which could silently diverge.

Review fix item 3 (Important -- cost clarity): a price is either EXPLICITLY configured (a real
float, including a genuine 0.0) or left `None` ("not priced" -- the caller never set it). Cost is
only ever a real number when BOTH `price_input` and `price_output` are explicitly set; otherwise
it is `None`, never a fabricated 0.0 that would misread as "$0 spent" rather than "unknown/not
configured". This is a DIFFERENT `None` from "no usage at all was ever returned" (also `None`,
but for a different reason) -- callers that need to distinguish the two reasons should check
`usage`/`final_usage` presence separately from the cost value itself.
"""
from __future__ import annotations


def usage_tokens(usage) -> int:
    """Total token count from one usage dict -- prefers the provider's own `total_tokens` when
    present, else sums `prompt_tokens` + `completion_tokens`. 0 for anything not usage-shaped."""
    if not isinstance(usage, dict):
        return 0
    if usage.get("total_tokens") is not None:
        return usage["total_tokens"]
    return (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)


def cost_for_usage(usage, price_input, price_output) -> float | None:
    """`cost = prompt_tokens * price_input/1e6 + completion_tokens * price_output/1e6` -- `None`
    when there is no usage to cost, OR when either price is `None` (not configured). A genuine
    0.0 is returned only when there IS usage and BOTH prices are explicit real numbers (0.0
    included)."""
    if not isinstance(usage, dict):
        return None
    # prefer the provider's ACTUAL billed cost when it reports one (OpenRouter returns
    # usage.cost in USD when accounting is requested) -- more accurate than our tokens*price
    # estimate, and correct even when the operator did not configure prices.
    actual = usage.get("cost")
    if actual is not None:
        try:
            return float(actual)
        except (TypeError, ValueError):
            pass
    if price_input is None or price_output is None:
        return None
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return (prompt_tokens / 1_000_000) * price_input + (completion_tokens / 1_000_000) * price_output


def conversation_totals(conversation: list) -> tuple:
    """Sums tokens/cost over a full `LLMPolicy.conversation` list. Returns `(total_tokens,
    total_cost_usd)`:
    - `total_tokens` is `None` iff not one single round ever returned any usage at all.
    - `total_cost_usd` is `None` iff not one single round's cost was ever a real number (i.e.
      pricing was never configured) -- kept strictly separate from `total_tokens`'s own None
      reason, so a genuinely unpriced-but-metered policy reports real tokens with a None cost,
      never a fabricated 0.0 cost that would misread as "$0 spent"."""
    any_usage = False
    any_priced = False
    total_tokens = 0
    total_cost = 0.0
    for entry in conversation:
        for rt in entry["tool_round_trips"]:
            if rt.get("usage") is not None:
                any_usage = True
                total_tokens += usage_tokens(rt["usage"])
                if rt.get("cost_usd") is not None:
                    any_priced = True
                    total_cost += rt["cost_usd"]
        # New records retain every terminal attempt because a bounded semantic repair can make a
        # second final request. Older records have only final_usage/final_cost_usd. Never sum both.
        final_attempts = entry.get("final_attempts")
        if final_attempts is None:
            final_attempts = [{"usage": entry.get("final_usage"),
                               "cost_usd": entry.get("final_cost_usd")}]
        for attempt in final_attempts:
            if attempt.get("usage") is not None:
                any_usage = True
                total_tokens += usage_tokens(attempt["usage"])
                if attempt.get("cost_usd") is not None:
                    any_priced = True
                    total_cost += attempt["cost_usd"]
    return (total_tokens if any_usage else None), (total_cost if any_priced else None)

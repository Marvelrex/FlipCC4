"""Prompt templates + parse_final .

Both system prompts are self-contained (objective, every action with its effect, the exact cost
rules, the tool menu, the output contract) -- a model given only (system + user) can play. `neutral`
is the only main-experiment framing and never mentions Nash/equilibrium/reward numerics ; `nash` is the pre-registered framing treatment appended as
an extra section, never the default.

the design: each template is BASE + TOOLS + the output contract, assembled by `build_system`; the
`tools_enabled` flag (default True, unchanged from every prior call) includes or ENTIRELY OMITS
the TOOLS section -- the `no-tool` registrable strategy (`flipthem.anchors`) is this flag set
False, never just `max_tool_calls=0` left in an otherwise unchanged prompt.

Substitution uses plain `str.replace`, not `str.format` -- the templates themselves contain
literal JSON braces (the output contract), which `str.format` would otherwise try to parse as
format fields.
"""
from __future__ import annotations

import json

from flipthem.llm.context import LegalContext
from flipthem.llm.action_reference import render_action_reference

# context-assisted recent-history window (design: "deterministic legal-context augmentation with
# one model request per decision" -- see build_user's context_augmented). Not pinned by the
# brief to an exact number; a small, documented, deliberate default.
CONTEXT_RECENT_N = 20

_OUTPUT_CONTRACT_CORE = (
    '## OUTPUT SCHEMA — FINAL RESPONSE\n{"decisions":[{"unit":"<id>","action":"<name>","target":"<listed target or null>"}],"wake":null}\n'
    'Set wake to {"after":<non-negative ticks>} for a delay capped by the stated wake limit; '
    'relevant events may wake you earlier. Keep wake null for event-or-limit scheduling. '
    'You may decide a subset of your decidable units; at least one entry is required. '
    'Omitted units take no new action; explicit Sleep with target null means intentional waiting.'
    '\nFORMAT: the complete final response is one JSON object of at most 512 tokens, beginning with '
    '{ and ending with }. It contains no analysis, preface, confirmation, Markdown or code fence. '
    'Each unit occurs at most once, with either one action or Sleep. Copy every '
    'unit, action and non-null target exactly from the current legal options.'
)
_OUTPUT_CONTRACT_TOOL_NOTE = (
    ' To call a tool instead, use the\ntool-call format; a plain JSON object with "decisions" ends your turn.'
)
_OUTPUT_CONTRACT = _OUTPUT_CONTRACT_CORE + _OUTPUT_CONTRACT_TOOL_NOTE

# Transport uses the same final-output contract, not a second divergent schema.
FINAL_TURN_INSTRUCTION = (
    'FINAL RESPONSE NOW. No tool calls remain. Use the original legal options and completed '
    'tool results. Return the JSON object with "decisions" under the system OUTPUT schema.'
)

# Used only by Backend's explicitly bounded semantic repair path. The repair is a fresh final
# request over the same current legal context; it cannot call a tool or reuse an invalid draft.
# Keeping this separate from the main framing text also makes the intervention easy to fingerprint
# and audit in the request/round records.
FINAL_REPAIR_INSTRUCTION = (
    'FORMAT RECOVERY. Use the original legal options and retained tool results to produce '
    'the decisions JSON under the system OUTPUT schema. This response contains no tool call or '
    'commentary and is at most 256 tokens.'
)

_RED_SYSTEM_BASE = """You are the RED (attacker) commander in a stealthy control-timing game on a segmented
enterprise network. You direct all of red's footholds.

## OBJECTIVE
There are 2 target servers. You score every round you HOLD ROOT on a target, and keep
scoring until blue reclaims it. Service disruption has zero scoring value. Each target is worth
{value}/round while RED holds root. The round budget is finite (given each round).

## ACTION DEPENDENCIES
The mechanism exposes this dependency chain per host: subnet discovery -> service discovery ->
remote exploit (foothold) -> privilege escalation (root). Actions are listed only for observed
subnets, IPs or hosts. PrivilegeEscalate is listed only when the same RED unit has a session on
the exact target hostname. A discovered hostname or root on another host does not satisfy that
session condition.

## PUBLIC HIGH-LEVEL ROUTES
- target A: contractor network -> restricted zone A -> operational zone A
- target B: contractor network -> restricted zone B -> operational zone B
The public/admin/office branch contains no scoring target. Exact CIDRs, IP-to-host mappings,
services and live access enter RED observations through RED actions.

## OBSERVABILITY
RED observations contain RED footholds, discovery and action results. They contain no BLUE moves
or BLUE detection status. A completed BLUE Restore removes RED sessions on that host.

## COSTS (exact)
RED actions have zero FT action cost. Actions have the durations and detection behavior listed
below. An illegal/invalid action has FT action cost 0.1.

"""

_RED_TOOLS = """## TOOLS
The available function tools may be called up to {max_tool_calls} times before a decision. Their
API schemas define the arguments. Results contain RED observations and current compiler-menu data.
Available functions: history, action_manual, known_targets, check_plan.

"""

_RED_SYSTEM = _RED_SYSTEM_BASE + _RED_TOOLS + _OUTPUT_CONTRACT

_BLUE_SYSTEM_BASE = """You are the BLUE (defender) commander in a stealthy control-timing game on a segmented
enterprise network. You direct all of blue's zone defenders.

## OBJECTIVE
There are 2 target servers. BLUE scores each round on every target without RED root. RED root
suspends BLUE scoring on that target until a completed Restore returns control.
Each target is worth {value}/round you deny it. The round budget is finite (given each round).

## COSTS (exact)
Restore has FT action cost 1. An illegal/invalid action has FT action cost 0.1. Sleep, Remove,
Analyse, BlockTrafficZone and AllowTrafficZone have zero FT action cost.

## DETECTION
BLUE receives Monitor alerts each round. Monitor alerts report observed activity. Analyse returns
host evidence. Neither observation directly reports RED control of a scoring target.

"""

_BLUE_TOOLS = """## TOOLS
The available function tools may be called up to {max_tool_calls} times before a decision. Their
API schemas define the arguments. Results contain BLUE observations and current compiler-menu data.
Available functions: history, action_manual, known_targets, check_plan.

"""

_BLUE_SYSTEM = _BLUE_SYSTEM_BASE + _BLUE_TOOLS + _OUTPUT_CONTRACT

# A framing arm must differ from `neutral` by its BEHAVIOURAL INSTRUCTION ONLY. The earlier text
# carried research metadata ("explicit ablation only -- never the main experiment", "this is a
# labelled ablation prompt") that told the model it was in an ablation and not the main experiment.
# `neutral` has no counterpart for that, so it was experimenter-demand information riding along with
# the treatment, and any effect could not be attributed to the reasoning style. The research
# labelling lives in the run manifest instead; the prompt keeps only the instruction.
_NASH_NOTE = """

## DECISION FRAME
This frame compares the current policy with unilateral deviations while holding opponent
behaviour fixed. The comparison variable is your own outcome."""

_ABLATION_NOTES = {"nash": _NASH_NOTE}

# `neutral` is the base prompt with no note appended; `nash` appends exactly one frame.
FRAMINGS = ("neutral",) + tuple(_ABLATION_NOTES)


class UnknownFraming(ValueError):
    """Raised for a framing outside `FRAMINGS`.

    This exists because the silent alternative is far worse than a crash. `build_system` used to
    look the note up with `.get(framing)` and append nothing when it missed, so a misspelled or
    wrong-case framing -- `"Nash"`, `"nash-equilibrium"` -- produced a prompt BYTE-IDENTICAL to
    `neutral`, while `MatchRecord.blue_framing`/`red_framing` (free-form strings) still recorded
    the name that was requested. The framing ablation would then compare two arms that received
    the same prompt: every contrast is exactly 0, and the result reads as "the nash prompt has no
    effect" rather than "the nash prompt was never sent". `anchors.LLM_STRATEGIES` already
    validates the other manipulated factor the same way."""


def build_system(team: str, framing: str = "neutral", max_tool_calls: int = 4,
                  value: float = 1.0, tools_enabled: bool = True) -> str:
    """`tools_enabled=False` (the design "no-tool" strategy) OMITS the "## TOOLS" section ENTIRELY --
    a clean single-prompt control isolating the tools' effect, never a `max_tool_calls=0` prompt
    that still advertises "up to 0 times" (the design). The action-execution
    reference is common to BOTH values of `tools_enabled`; only the TOOLS
    menu and its output-contract note differ between those assistance arms.

    a: `tools_enabled=False` ALSO drops the OUTPUT contract's trailing
    "To call a tool instead, use the tool-call format..." clause (`_OUTPUT_CONTRACT_TOOL_NOTE`) --
    the design only omitted the "## TOOLS" menu itself, leaving this one residual mention. A no-tool
    prompt now has zero occurrences of the substring "tool" anywhere."""
    if team not in ("blue", "red"):
        raise ValueError(f"unknown team {team!r}; expected 'blue' or 'red'")
    base = _BLUE_SYSTEM_BASE if team == "blue" else _RED_SYSTEM_BASE
    tools = (_BLUE_TOOLS if team == "blue" else _RED_TOOLS) if tools_enabled else ""
    contract = _OUTPUT_CONTRACT if tools_enabled else _OUTPUT_CONTRACT_CORE
    # The detailed reference is shared by every assistance/framing arm. It is rendered from the
    # compiler action order plus action_manual metadata so target types, durations and FT costs
    # cannot drift into a second prompt-only table.
    action_reference = "\n\n" + render_action_reference(team) + "\n\n"
    text = (base + action_reference + tools + contract).replace("{value}", str(value))
    if tools_enabled:
        text = text.replace("{max_tool_calls}", str(max_tool_calls))
    if framing not in FRAMINGS:
        raise UnknownFraming(
            f"unknown framing {framing!r}; expected one of {sorted(FRAMINGS)}. Framing is a "
            f"manipulated factor, so an unrecognised value is refused rather than silently "
            f"rendered as the neutral prompt (which would make the framing contrast compare two "
            f"identical prompts while the record still claimed they differed).")
    note = _ABLATION_NOTES.get(framing)
    if note:
        text += note
    return text


def _render_legal_options(ctx: LegalContext) -> list[str]:
    """Render every legal pair once, grouping actions with identical target lists.

    BLUE commonly has Restore/Analyse/Remove over the same dozens of hosts; RED commonly has
    three scan/exploit actions over the same IP list.  Repeating that list once per action can
    dominate the request.  Joining only actions whose complete ordered target lists are equal is
    lossless: expanding ``A|B->[x,y]`` produces exactly the original pairs.
    """
    lines = [
        "Currently legal options by unit (A|B->[x,y] means each named action is legal on every listed target):"
    ]
    for slot, pairs in ctx.decidable.items():
        if not pairs:
            continue
        by_action = {}
        for action, target in pairs:
            targets = by_action.setdefault(action, [])
            if target not in targets:
                targets.append(target)
        by_targets = {}
        for action, targets in by_action.items():
            by_targets.setdefault(tuple(targets), []).append(action)
        rendered = []
        for targets, actions in by_targets.items():
            target_text = ",".join("null" if target is None else target for target in targets)
            rendered.append(f"{'|'.join(actions)}->[{target_text}]")
        lines.append(f"- {slot}: {'; '.join(rendered)}")
    return lines


def _prompt_history_summary(ctx: LegalContext, summary: dict) -> dict:
    """Keep cumulative decision evidence; omit state duplicated by Current known state."""
    out = {key: summary[key] for key in ("env_ticks", "submissions", "completions", "successes")}
    if ctx.team == "blue":
        out["restores_done"] = summary["restores_done"]
        out["alert_rounds"] = summary["alerts_seen"]
    return out


def _prompt_known_state(ctx: LegalContext) -> dict:
    """Compact prompt view; the callable known_targets tool retains the complete details."""
    from flipthem.llm.tools import known_targets

    state = known_targets(ctx)
    if ctx.team == "blue":
        return {"defense_state": state["defense_state"]}
    holdings = state["holdings"]
    discovered = holdings.get("discovered", {})
    progress = {}
    for target, item in state["target_progress"].items():
        progress[target] = {
            "seen": item["observed_by_reconnaissance"],
            "ips": item.get("observed_ips", []),
            "session": item["session_on_exact_target"],
            "root": item["root_on_exact_target"],
        }
    return {
        "target_progress": progress,
        "holdings": {"sessions": holdings.get("sessions", ()), "root": holdings.get("root", ())},
        # Observed counts only. Deliberately no recommended response or failure-derived advice.
        "consecutive_failures": state["consecutive_failures"],
        "discovered_counts": {kind: len(discovered.get(kind, ()))
                              for kind in ("subnets", "ips", "hosts")},
    }


def build_user(ctx: LegalContext, tools_enabled: bool = True,
               context_augmented: bool = False) -> str:
    """`tools_enabled=False` (review fix item 2) drops the trailing "You may call tools before
    deciding." sentence -- the design/a later revision-preflight task 1a cleaned the SYSTEM prompt's tool
    references for the no-tool strategy, but left this PER-ROUND user message unconditionally
    advertising tools, a confound between the two strategies' actual per-round prompt. Default
    True is byte-identical to every prior `build_user` call (only the caller -- agent.py -- now
    passes `tools_enabled=self.tools_enabled` explicitly).

    `context_augmented=True` is the `context-assisted` strategy ("deterministic legal-context
    augmentation with one model request per decision" -- NOT tool-assisted, the model never
    calls a tool): the PROGRAM pre-assembles, from THIS `ctx` alone, a cumulative decision
    summary, compact recent action/observation events, and compact current known state. Static
    action manuals are already complete in the system prompt and are not repeated each round.
    The current legal pairs are losslessly grouped only when actions have identical target lists.
    Never derives from anything but `ctx` -- no true state, no env clone, no look-ahead, no reward
    numerics (the same isolation `flipthem.llm.tools` itself already guarantees)."""
    lines = [
        f"Round {ctx.round} of {ctx.horizon}.",
        f"Public scoring targets: {', '.join(ctx.target_hosts)}.",
        *_render_legal_options(ctx),
    ]

    if ctx.team == "blue":
        if ctx.alerts:
            lines.append(f"New Monitor alerts this round: {', '.join(ctx.alerts)}.")
        else:
            lines.append("New Monitor alerts this round: none.")
    elif not context_augmented:
        sessions = list(ctx.holdings.get("sessions", ())) if ctx.holdings else []
        root = list(ctx.holdings.get("root", ())) if ctx.holdings else []
        held = [f"foothold@{s}" for s in sessions] + [f"ROOT@{h}" for h in root]
        lines.append(f"You currently hold: {', '.join(held) if held else 'nothing yet'}.")

    if context_augmented:
        lines.append("")
        lines.extend(_render_context_augmentation(ctx, tools_enabled=tools_enabled))

    if tools_enabled:
        lines.append("You may call tools before deciding. When ready, output the final "
                     "decisions JSON.")
    else:
        lines.append("When ready, output the final decisions JSON.")
    return "\n".join(lines)


def _render_context_augmentation(ctx: LegalContext, tools_enabled: bool = False) -> list:
    """Compact deterministic context derived only from ``LegalContext``.

    The system prompt already contains the complete action manual and execution pseudocode, so
    repeating per-action manuals in every user message adds no information.  History is the same
    compact view returned by the history tool; current known state omits full discovery lists
    already represented by the exact legal options (the full list remains callable).
    """
    from flipthem.llm.tools import history

    hist = history(ctx, last_n=CONTEXT_RECENT_N)
    lines = [
        "## PRE-ASSEMBLED CONTEXT (deterministic)",
        f"Episode summary: {json.dumps(_prompt_history_summary(ctx, hist['summary']), separators=(',', ':'))}",
        (f"Recent compact events (last {CONTEXT_RECENT_N}; one row per action attempt or "
         f"observation round): {json.dumps(hist['recent'], separators=(',', ':'))}"),
        f"Current known state: {json.dumps(_prompt_known_state(ctx), separators=(',', ':'))}",
    ]
    return lines


def parse_final(text: str):
    """Returns (decisions, wake) or (None, None) if the text does not carry a valid
    {"decisions": [...], "wake": {...}|None} JSON object (a malformed final -> MODEL_ERROR upstream,
    design the design).

    Item 5 (fix round 2): a genuinely EMPTY decisions list, or any decision entry missing
    "unit"/"action", is a FORMAT ERROR and fails the WHOLE response closed (None, None) -- never
    silently drop just the bad entry and let the rest through (design the design's output contract:
    "one decisions entry per decidable unit"; a model that submits nothing, or an unaddressable
    entry, has not met that contract, and the agent must bill this as MODEL_ERROR, never a clean,
    unrequested Sleep)."""
    # The frozen contract says the complete response is the JSON object. Extracting an object out
    # of prose/Markdown silently turns a format violation into a success and makes the recorded
    # model-error rate disagree with the prompt used in the experiment.
    obj = _try_json(text)
    if not isinstance(obj, dict) or "decisions" not in obj:
        return None, None
    decisions = obj.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return None, None
    units = set()
    for d in decisions:
        if not isinstance(d, dict) or "unit" not in d or "action" not in d:
            return None, None
        if any(not isinstance(d[k], str) or not d[k] for k in ("unit", "action")):
            return None, None
        if d.get("target") is not None and not isinstance(d["target"], str):
            return None, None
        if d["unit"] in units:
            return None, None
        units.add(d["unit"])
    wake = _coerce_wake(obj.get("wake"))
    return decisions, wake


def _coerce_wake(wake):
    """I3 (review round 2): a model can hand back a malformed wake (a non-integer/negative
    "after", e.g. {"after": "soon"} or {"after": [1, 2]} or {"after": -5}) -- this must never
    reach `should_call` unvalidated (it would crash `int(...)` there and kill the match). Any
    "after" that cannot be coerced to a non-negative int drops the WHOLE wake object (the caller
    falls back to event-or-cap, design the design's documented "wake absent" behaviour) rather than
    guessing at partial intent. Other shapes (e.g. {"on": "event"}, or no wake at all) pass through
    untouched -- only "after" ever needs numeric validation."""
    if not isinstance(wake, dict):
        return None
    if "after" in wake:
        try:
            n = int(wake["after"])
        except (TypeError, ValueError):
            return None
        if n < 0:
            return None
        return {"after": n}
    return wake


def _try_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_first_json_object(text: str):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    while end > start:
        obj = _try_json(text[start:end + 1])
        if obj is not None:
            return obj
        end = text.rfind("}", 0, end)
    return None

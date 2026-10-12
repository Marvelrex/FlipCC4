"""Backend, closed budget & error handling (the design; the design moves budget to the adapter).

Every LLM path talks to an INJECTED client (offline constraint; no module ever reads a key)
through a common interface:

    client.next(system: str, user: str, transcript: list, force_final: bool) -> dict

returning either `{"type": "tool_calls", "calls": [{name,args,id?}, ...]}` or `{"type": "final",
"text": str}` (the model's raw final-turn text, parsed by `flipthem.llm.prompts.parse_final`;
the design extends this so a final response MAY also carry `"usage"`, a token-count dict the stub can
fake), or raising `ProviderError` (a provider outage) or any other exception (a programming
error -- see below).

Two layers, one budget : `ServiceAdapter` wraps the RAW injected client and exposes this
SAME `.next(...)` interface, so `Backend` can treat an adapter exactly like an ordinary client --
`Backend` itself no longer knows what a budget even is. `ServiceAdapter` owns ALL retry/budget
accounting: `budget.before_request()` is called before EVERY physical attempt (the first, and
each of up to `max_retries` retries), so a physical attempt is counted exactly once, matching the
the design worst-case formula `(#LLM sides) x T x (max_tool_calls+1) x (max_retries+1)`. `Backend`
drives the bounded tool loop on top: the initial turn, every tool call, and the forced final turn
(issued once `max_tool_calls` tool calls have been made -- the client is asked one more time with
`force_final=True` and is not permitted a further tool call even if it asks for one) are each ONE
logical round-trip through the adapter. Three terminal states, never conflated: `DECISION`
(compile per slot), `MODEL_ERROR` (malformed/invalid even on the forced final turn -- never a
fabricated decisions object), `PROVIDER_FAILURE` (no usable response at all after the adapter's
own retries -- distinct from MODEL_ERROR, the match gets flagged/excluded upstream).

Only `ProviderError` is ever treated as a provider outage (review round 2, C1; still true post-
the design, just enforced one layer down in `ServiceAdapter`): an injected client signals a genuine
network/timeout/rate-limit-shaped failure by raising `ProviderError` (or a subclass); every OTHER
exception -- a programming error in the client, in `tool_dispatch`, or anywhere else in the call
chain (`KeyError`, `AssertionError`, `TypeError`, ...) -- propagates uncaught, with NO retry.
Swallowing every exception into PROVIDER_FAILURE (or silently retrying it) would disguise real
bugs as provider outages (and neuters any test tripwire that raises on misuse, e.g. a scripted
stub asked for more calls than it was given).
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field

from flipthem.llm.prompts import parse_final
from flipthem.llm.cost import cost_for_usage

DECISION = "DECISION"
MODEL_ERROR = "MODEL_ERROR"
PROVIDER_FAILURE = "PROVIDER_FAILURE"

# design the diagnostic code for a CONFIRMED content refusal. It is a sub-class of
# MODEL_ERROR (same scoring, same "keep the episode" rule), not a fourth terminal state -- the
# classification exists so refusal rate can be reported apart from malformed output, which the
# design requires because it is itself part of the treatment effect.
REFUSAL_CODE = "content_refusal"


def _refusal_signal(response: dict):
    """The provider-authored refusal signal on a response, or None.

    the design forbids inferring a content refusal from an HTTP status code ("auth, routing and
    request-parameter errors must stay separate from a content-policy refusal"). No HTTP error
    reaches this function at all, but they do not all take the same route out (an earlier version
    of this note wrongly said they were all ProviderErrors):

      * 429 and 5xx  -> `real_client` converts them to `ProviderError` -> PROVIDER_FAILURE.
      * every other status (400/401/403/404/...) -> the `HTTPError` PROPAGATES unchanged, past
        this function entirely, out of `run_match` to the CLI, which reports it as a refused
        REQUEST (exit code 8). A 403 is a config/auth problem, not a content-policy answer,
        however its body reads.

    Only two signals count, and both are written by the provider INTO a 200 response body:
    OpenAI-compatible `finish_reason == "content_filter"`, and OpenAI's structured
    `message.refusal` string (passed through by real_client._parse_response). Both routes are
    covered by tests that drive a monkeypatched `urlopen`, not a hand-built ProviderError."""
    if response.get("finish_reason") == "content_filter":
        return "finish_reason=content_filter"
    refusal = response.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal
    return None


def _final_excerpt(text):
    """Bounded visible final answer only; no reasoning field or request headers."""
    if not isinstance(text, str):
        return f"<non-text content: {type(text).__name__}>"
    text = re.sub(r"(?is)<think(?:ing)?>.*?(?:</think(?:ing)?>|$)", "[omitted]", text)
    text = re.sub(r"(?i)sk-[\w-]+|Bearer\s+[^\s\"']+", "[REDACTED]", text)
    return text[:4096]


def _error_detail(exc, limit: int = 400) -> str:
    """A bounded, key-free description of a failure for the run artifacts. `ProviderError` messages
    are built from the provider label plus the HTTP status or the network error, so they carry no
    credentials; the redaction below is belt-and-braces for any other exception type."""
    text = " ".join(f"{exc}".split())
    text = re.sub(r"(?i)sk-[\w-]+|Bearer\s+[^\s\"']+", "[REDACTED]", text)
    return text[:limit]


class BudgetExceeded(Exception):
    """Raised by `RequestBudget.before_request` once the request limit is reached. Propagates
    all the way out of `ServiceAdapter.next` / `Backend.run_decision` uncaught -- the caller
    (agent.py / eval_runner.py) decides how to react (design the design/7: the match is marked
    incomplete/excluded, never a faked payoff)."""


class ProviderError(Exception):
    """Raise this (or a subclass) from an injected client's `.next(...)` to signal a genuine
    provider outage (network error, timeout, rate limit, ...) -- the ONLY exception type
    `ServiceAdapter`/`Backend` ever treats as a provider failure. Any other exception is a
    programming error and is left to propagate, with no retry (review round 2, C1)."""


class RequestBudget:
    """Counts EVERY physical model round-trip (design the design: "counts EVERY call"; the design:
    counted exclusively by `ServiceAdapter`, once per physical attempt including retries).
    `limit=None` is unbounded (used by offline unit tests that don't care about budget
    exhaustion)."""

    def __init__(self, limit: int | None):
        self._limit = limit
        self._spent = 0

    def before_request(self) -> None:
        if self._limit is not None and self._spent >= self._limit:
            raise BudgetExceeded(f"request budget exhausted (limit={self._limit})")
        self._spent += 1

    @property
    def spent(self) -> int:
        return self._spent


@dataclass(frozen=True)
class Result:
    kind: str                       # DECISION | MODEL_ERROR | PROVIDER_FAILURE
    decisions: list | None = None
    wake: dict | None = None
    # : one entry per PHYSICAL response actually inspected this round,
    # in order (each the raw `usage` dict that response carried, or None if it carried none) --
    # `ServiceAdapter.last_usage` only ever keeps the LATEST value, which cannot support a
    # per-tool-call cost breakdown. Empty for PROVIDER_FAILURE (no response was ever obtained on
    # the failing attempt to read a usage from).
    usages: tuple = ()
    # calls dispatched in each tool-DISPATCHING turn, in order (e.g. (2,) = one turn of two parallel
    # calls; (1,1) = two single-call turns). Lets the caller attribute each physical response's one
    # usage to the whole turn it covers, instead of assuming one usage per individual tool call.
    tool_turn_sizes: tuple = ()
    diagnostic: dict | None = None


@dataclass(frozen=True)
class ReproRecord:
    """the design: enough to reproduce (or at least audit) one decision round-trip after the fact --
    never the reasoning itself, only what was asked and what it cost. `prompt_hash` is a stable
    digest of the exact assembled (system, user) pair sent this round; `token_usage` is whatever
    the client's last response reported (`None` if the client/stub never provided any -- usage is
    optional on the contract, not guaranteed).

    `provider_failures` here is `ServiceAdapter.provider_failures`: a CUMULATIVE count of
    PHYSICAL attempts that raised `ProviderError`, over the adapter's whole lifetime (every
    retried attempt on every decision round adds to it, and it is never reset) -- distinct from
    `agent.Telemetry.provider_failures`, which counts LOGICAL decision rounds that ended in a
    `PROVIDER_FAILURE` result (i.e. only rounds where every retry was exhausted; one such round
    can itself account for several of the adapter's physical-attempt failures, so this field is
    always >= `Telemetry.provider_failures`, never the same counter under two names).

    : `provider`/`strategy`/
    `tool_call_count`/`model_errors`/`total_tokens`/`total_cost_usd` are POLICY-level facts
    (`agent.LLMPolicy` knows its own client's provider, its own tools_enabled-derived strategy
    label, and its own `Telemetry` counts) that the adapter itself has no way to know -- they are
    threaded in by `LLMPolicy.repro_record()` as plain values, never derived here. All six default
    to a neutral/empty value so every pre-existing `ReproRecord`/`repro_record(...)` call site
    (none of which ever passed these) is completely unaffected. The API key NEVER appears on this
    (or any) field -- it is used only inside one HTTP request's Authorization header
    (`real_client.RealClient.next`) and is never plumbed anywhere near a record/log."""
    model: str
    params: dict
    prompt_hash: str | None
    tool_config: dict
    request_count: int
    token_usage: dict | None
    provider_failures: int
    provider: str | None = None                    # the GATEWAY label (e.g. "openrouter")
    upstream_provider_requested: str | None = None  # the pinned OpenRouter upstream, if any
    reasoning_effort: str | None = None             # the requested reasoning effort, if any
    returned_model: str | None = None               # the ACTUAL serving model ("unknown" if absent)
    returned_provider: str | None = None            # the ACTUAL serving provider ("unknown" if absent)
    strategy: str = "tool-assisted"
    tool_call_count: int = 0
    model_errors: int = 0
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    # ``total_cost_usd`` is exact only when every physical request has an attributable cost.
    # A timeout/non-response may conceivably have been billed but exposes no usage. Preserve the
    # sum we can prove separately and mark completeness instead of under-reporting it as a total.
    known_cost_subtotal_usd: float | None = None
    cost_complete: bool = True


class ServiceAdapter:
    """Wraps an INJECTED raw client (no key read anywhere, ever) and exposes the SAME
    `.next(system, user, transcript, force_final) -> dict` interface `Backend` expects, so
    `Backend` can treat one of these exactly like an ordinary client . Owns ALL budget and
    retry accounting for the physical attempts underneath one logical round-trip: `max_retries=N`
    means up to `N+1` physical attempts, each preceded by exactly ONE `budget.before_request()`
    call -- a `ProviderError` from the raw client is retried (up to the limit, then re-raised so
    `Backend` can flag `PROVIDER_FAILURE`); any OTHER exception propagates on the FIRST attempt,
    with no retry at all (a programming error is not made "more provider-shaped" by retrying it).
    """

    def __init__(self, client, model: str, params: dict, max_retries: int, budget: RequestBudget,
                 price_input: float | None = None, price_output: float | None = None,
                 max_cost_usd: float | None = None):
        self.client = client
        self.model = model
        self.params = dict(params or {})
        self.max_retries = max_retries
        self.budget = budget
        self.last_usage: dict | None = None
        self.last_prompt_hash: str | None = None
        self.provider_failures = 0
        # a cumulative COST stop-loss, SEPARATE from the request-count budget. Needs both
        # prices to accrue anything (cost_for_usage is None otherwise), so it is inert when unpriced.
        self.price_input = price_input
        self.price_output = price_output
        self.max_cost_usd = max_cost_usd
        self._cost_spent = 0.0
        # the ACTUAL serving model/provider from the latest response (None until reported).
        self.last_returned_model: str | None = None
        self.last_returned_provider: str | None = None
        # Only accounting/identifiers, never request headers or response reasoning text.
        self.request_log: list = []
        # Optional durability hook installed by eval_runner when conversation journaling is on.
        # It is invoked after each physical attempt is appended, before a retry/tool continuation.
        self.request_observer = None

    def _append_request(self, entry: dict) -> None:
        self.request_log.append(entry)
        if self.request_observer is not None:
            self.request_observer(entry)

    def next(self, system: str, user: str, transcript: list, force_final: bool) -> dict:
        # cost stop-loss checked BEFORE any request -- once cumulative cost has reached the
        # cap, no further request is sent (the request that crossed the line already completed;
        # its cost is only knowable after it returned). Distinct from the request-count budget.
        if self.max_cost_usd is not None and self._cost_spent >= self.max_cost_usd:
            raise BudgetExceeded(
                f"cost stop-loss reached (${self._cost_spent:.6f} >= ${self.max_cost_usd})")
        self.last_prompt_hash = _hash_prompt(system, user)
        last_exc: ProviderError | None = None
        for _attempt in range(self.max_retries + 1):
            self.budget.before_request()   # every PHYSICAL attempt, no path bypasses this
            try:
                response = self.client.next(system=system, user=user, transcript=transcript,
                                            force_final=force_final)
            except ProviderError as exc:
                # Record the MESSAGE, not just the type: a PROVIDER_FAILURE is otherwise
                # unattributable (rate limit vs 5xx vs timeout all look identical in the
                # artifacts). Provider-authored text only -- the API key never appears in it.
                retrying = _attempt < self.max_retries
                try:
                    requested_delay = float(getattr(exc, "retry_after_seconds", 0.0))
                except (TypeError, ValueError):
                    requested_delay = 0.0
                if not math.isfinite(requested_delay):
                    requested_delay = 0.0
                retry_delay = (max(0.0, min(requested_delay, 60.0)) if retrying else 0.0)
                self._append_request({"attempt": self.budget.spent, "error": "ProviderError",
                                      "error_detail": _error_detail(exc), "usage": None,
                                      "http_status": getattr(exc, "http_status", None),
                                      "retry_delay_seconds": retry_delay})
                last_exc = exc
                self.provider_failures += 1
                if retry_delay:
                    time.sleep(retry_delay)
                continue   # retry, up to max_retries
            except Exception as exc:
                self._append_request({"attempt": self.budget.spent,
                                      "error": type(exc).__name__,
                                      "error_detail": _error_detail(exc), "usage": None})
                raise   # persist accounting, never swallow or retry programming errors
            self._record_response(response)
            return response
        raise last_exc   # every physical attempt failed -- let Backend classify PROVIDER_FAILURE

    def _record_response(self, response, error=None):
        response = response if isinstance(response, dict) else {}
        entry = {k: response.get(k) for k in (
            "usage", "response_id", "finish_reason", "returned_model", "returned_provider")}
        entry.update(attempt=self.budget.spent, error=error)
        entry["cost_usd"] = cost_for_usage(entry["usage"], self.price_input, self.price_output)
        self._append_request(entry)
        self.last_usage = entry["usage"]
        if entry["cost_usd"] is not None:
            self._cost_spent += entry["cost_usd"]
        if entry["returned_model"] is not None:
            self.last_returned_model = entry["returned_model"]
        if entry["returned_provider"] is not None:
            self.last_returned_provider = entry["returned_provider"]

    @property
    def cost_spent(self) -> float:
        return self._cost_spent

    def repro_record(self, tool_config: dict, provider: str | None = None,
                     upstream_provider_requested: str | None = None,
                     reasoning_effort: str | None = None,
                     strategy: str = "tool-assisted", tool_call_count: int = 0,
                     model_errors: int = 0, total_tokens: int | None = None,
                     total_cost_usd: float | None = None,
                     known_cost_subtotal_usd: float | None = None,
                     cost_complete: bool = True) -> ReproRecord:
        return ReproRecord(
            model=self.model, params=dict(self.params), prompt_hash=self.last_prompt_hash,
            tool_config=dict(tool_config), request_count=self.budget.spent,
            token_usage=self.last_usage, provider_failures=self.provider_failures,
            provider=provider, upstream_provider_requested=upstream_provider_requested,
            reasoning_effort=reasoning_effort,
            returned_model=self.last_returned_model or "unknown",
            returned_provider=self.last_returned_provider or "unknown",
            strategy=strategy, tool_call_count=tool_call_count,
            model_errors=model_errors, total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            known_cost_subtotal_usd=known_cost_subtotal_usd,
            cost_complete=cost_complete,
        )


def _hash_prompt(system: str, user: str) -> str:
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(user.encode("utf-8"))
    return digest.hexdigest()


class Backend:
    """Drives the bounded tool loop on top of an injected `client` -- in production this is
    ALWAYS a `ServiceAdapter` instance (the design: budget lives there, not here); offline tests may
    pass a raw stub directly when they have no need to exercise retries/budget at all."""

    def __init__(self, client, model: str, max_tool_calls: int,
                 max_final_repairs: int = 0):
        if max_final_repairs < 0:
            raise ValueError("max_final_repairs must be nonnegative")
        self.client = client
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_final_repairs = max_final_repairs

    def _repair_final(self, system: str, user: str, tool_dispatch, original: Result,
                      transcript: list) -> Result:
        """Retry a model-authored terminal formatting failure without hiding it.

        Each repair uses a fresh, forced-final Backend with no tool allowance and no recursive
        repairs. The shared ServiceAdapter still records every physical request, token count and
        cost. A recovered decision carries the original failure in ``diagnostic``; exhausting the
        repair allowance remains MODEL_ERROR. A transport failure during the optional repair does
        not turn an already-observed model error into an excluded provider-failure episode.
        """
        if self.max_final_repairs == 0:
            return original
        from flipthem.llm.prompts import FINAL_REPAIR_INSTRUCTION

        all_usages = list(original.usages)
        attempts = []
        last = original
        repair_user = user + "\n\n" + FINAL_REPAIR_INSTRUCTION
        for number in range(1, self.max_final_repairs + 1):
            repaired = Backend(self.client, self.model, max_tool_calls=0,
                               max_final_repairs=0).run_decision(
                                   system, repair_user, tool_dispatch,
                                   initial_transcript=transcript)
            all_usages.extend(repaired.usages)
            attempts.append({"attempt": number, "outcome": repaired.kind,
                             "diagnostic": repaired.diagnostic})
            if repaired.kind == DECISION:
                diagnostic = dict(repaired.diagnostic or {})
                diagnostic["semantic_repair"] = {
                    "attempted": number, "recovered": True,
                    "original": original.diagnostic, "attempts": attempts,
                }
                return Result(kind=DECISION, decisions=repaired.decisions, wake=repaired.wake,
                              usages=tuple(all_usages),
                              tool_turn_sizes=original.tool_turn_sizes,
                              diagnostic=diagnostic)
            if repaired.kind == PROVIDER_FAILURE:
                # Preserve the original scored model error. The repair failure is still visible in
                # both this diagnostic and ServiceAdapter.request_log.
                break
            last = repaired

        diagnostic = dict((last if last.kind == MODEL_ERROR else original).diagnostic or {})
        diagnostic["semantic_repair"] = {
            "attempted": len(attempts), "recovered": False,
            "original": original.diagnostic, "attempts": attempts,
        }
        return Result(kind=MODEL_ERROR, usages=tuple(all_usages),
                      tool_turn_sizes=original.tool_turn_sizes, diagnostic=diagnostic)

    def run_decision(self, system: str, user: str, tool_dispatch, *,
                     initial_transcript: list | None = None) -> Result:
        """`tool_dispatch(name, args) -> dict` executes one tool and returns its payload.
        `BudgetExceeded` (raised by the adapter underneath `self.client`, if any) propagates
        uncaught -- including on what would have been the forced final turn."""
        # Repair reuses completed tool turns without dispatching or billing them again.
        transcript: list = list(initial_transcript or ())
        tool_calls_made = 0
        structured_final_requested = False
        tool_turn_sizes: list = []   # calls dispatched per tool turn (for per-turn usage attribution)
        usages: list = []   # task 4: one entry per physical response actually inspected

        while True:
            forced_final = (structured_final_requested
                            or tool_calls_made >= self.max_tool_calls)
            try:
                response = self.client.next(system=system, user=user, transcript=list(transcript),
                                             force_final=forced_final)
            except ProviderError as exc:
                return Result(kind=PROVIDER_FAILURE, usages=tuple(usages),
                              tool_turn_sizes=tuple(tool_turn_sizes),
                              diagnostic={"code": "provider_failure",
                                          "error_detail": _error_detail(exc)})
            if not isinstance(response, dict):
                return Result(kind=MODEL_ERROR, usages=tuple(usages),
                              tool_turn_sizes=tuple(tool_turn_sizes),
                              diagnostic={"code": "invalid_response_type"})

            usages.append(response.get("usage"))
            diagnostic = {k: response.get(k) for k in
                          ("response_id", "finish_reason", "returned_model", "returned_provider")}
            if response.get("type") == "final":
                diagnostic["final_text_excerpt"] = _final_excerpt(response.get("text", ""))
            refusal = _refusal_signal(response)
            if refusal is not None:
                # Checked BEFORE the truncation branch: a provider that sets both has told us the
                # specific reason, and "truncated" would hide it.
                return Result(kind=MODEL_ERROR, usages=tuple(usages),
                              tool_turn_sizes=tuple(tool_turn_sizes),
                              diagnostic={"code": REFUSAL_CODE, **diagnostic,
                                          "refusal_excerpt": _final_excerpt(refusal)})
            # Some provider routes reject custom tools and structured output in one request.
            # RealClient therefore uses an ordinary tool-capable turn to let the model decide
            # whether it needs another tool. A content response on that phase means "ready";
            # issue one fresh forced-final request over the same legal context/tool transcript.
            # The final request has strict response_format and no tools. It is ordinary protocol
            # work, fully billed and recorded, not a semantic repair.
            if (response.get("type") == "final"
                    and response.get("requires_structured_final") is True
                    and response.get("structured_output_applied") is not True
                    and not forced_final):
                structured_final_requested = True
                continue
            if response.get("finish_reason") == "length":
                error = Result(kind=MODEL_ERROR, usages=tuple(usages),
                               tool_turn_sizes=tuple(tool_turn_sizes),
                               diagnostic={"code": "output_truncated", **diagnostic})
                return self._repair_final(system, user, tool_dispatch, error, transcript)

            rtype = response.get("type")
            if rtype == "model_error":
                return Result(kind=MODEL_ERROR, usages=tuple(usages), tool_turn_sizes=tuple(tool_turn_sizes),
                              diagnostic={"code": response.get("code", "invalid_response"), **diagnostic})
            if rtype in ("tool_call", "tool_calls") and not forced_final:
                # Normalise both shapes to a list of {name, args, id}: the real client emits the
                # plural "tool_calls" (one or many); stubs and older callers may still emit the
                # singular "tool_call". A turn may legitimately batch several calls (all 4 tools are
                # pure reads of an immutable per-decision context).
                if rtype == "tool_call":
                    calls = [{"name": response.get("name"), "args": response.get("args", {}),
                              "id": response.get("id")}]
                else:
                    calls = response.get("calls") or []
                if not calls:
                    return Result(kind=MODEL_ERROR, usages=tuple(usages),
                                  tool_turn_sizes=tuple(tool_turn_sizes),
                                  diagnostic={"code": "invalid_tool_call", **diagnostic})
                for c in calls:
                    if not isinstance(c.get("name"), str) or not isinstance(c.get("args"), dict):
                        return Result(kind=MODEL_ERROR, usages=tuple(usages),
                                      tool_turn_sizes=tuple(tool_turn_sizes),
                                      diagnostic={"code": "invalid_tool_arguments", **diagnostic})
                # Dispatch EVERY call in the turn, in order, recording one result each. The whole
                # turn is executed even if it exceeds the remaining max_tool_calls budget -- never
                # truncate or rewrite what the model actually asked for; the NEXT request is then
                # forced-final (tool_calls_made crosses the cap below), so the overshoot is at most
                # one turn's worth.
                executed = [{"name": c["name"], "args": c["args"],
                             "result": tool_dispatch(c["name"], c["args"]), "id": c.get("id")}
                            for c in calls]
                # One transcript entry per assistant TURN (carries all N calls). Wire-protocol
                # item 2: each call's id lets _build_messages reconstruct one assistant message with
                # N tool_calls + N matching tool_call_id turns on the NEXT physical request.
                entry = {"calls": executed}
                if response.get("assistant_message") is not None:
                    entry["assistant_message"] = response["assistant_message"]
                # carry the reasoning trace so _build_messages can re-send it verbatim.
                if response.get("reasoning_details") is not None:
                    entry["reasoning_details"] = response["reasoning_details"]
                if response.get("reasoning") is not None:
                    entry["reasoning"] = response["reasoning"]
                transcript.append(entry)
                tool_turn_sizes.append(len(executed))
                tool_calls_made += len(executed)   # budget counts individual calls
                continue

            if rtype == "final":
                decisions, wake = parse_final(response.get("text", ""))
                if decisions is None:
                    code = "empty_final" if not response.get("text") else "invalid_final_json"
                    error = Result(kind=MODEL_ERROR, usages=tuple(usages),
                                   tool_turn_sizes=tuple(tool_turn_sizes),
                                   diagnostic={"code": code, **diagnostic})
                    return self._repair_final(system, user, tool_dispatch, error, transcript)
                return Result(kind=DECISION, decisions=decisions, wake=wake,
                              usages=tuple(usages), tool_turn_sizes=tuple(tool_turn_sizes),
                              diagnostic=diagnostic)

            # forced_final but the client still tried a tool call, or an unrecognised response
            # type -- both are malformed-final situations: never silently fabricate a decision.
            return Result(kind=MODEL_ERROR, usages=tuple(usages), tool_turn_sizes=tuple(tool_turn_sizes),
                          diagnostic={"code": "unexpected_response_type", **diagnostic})

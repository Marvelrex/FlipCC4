"""Final (Action, typed target) -> native action compiler (the design; the design).

Two responsibilities from ONE scan of `env.get_action_space(slot)` (the design integration
point 1: "the caller/agent fetches game.env.get_action_space(slot) and filters to True-valued
entries + our action set, then passes that data in"):

1. `build_slot_action_index` -- the name->class map is built from the action space's own
   'action' keys (`{ActionClass: bool}`), matched by `cls.__name__`, so there is no import-path
   guessing (per-verified, see the design notes). It also
   enumerates every currently-legal (action_name, target_str | None) pair for the slot -- this IS
   the "filtered per-slot action space" `flipthem.llm.context.build_legal_context` receives.
2. `compile_action` reverses one such (action_name, target_str) pair back to a real native action
   instance (re-resolving the string to the correctly-typed target object recorded at index-build
   time) -- illegal/unknown pairs compile to `InvalidAction` (native k=0.1), never
   `skip_valid_action_check`.

Target kind per action (signature params minus self/session/agent; verified live against
`inspect.signature`, brief's "CC4 / wrapper API facts"): Restore/Analyse/Remove/PrivilegeEscalate
-> hostname; DiscoverRemoteSystems -> subnet; AggressiveServiceDiscovery/StealthServiceDiscovery/
ExploitRemoteService -> ip_address; Sleep -> none (its native constructor takes no args at all,
not even session/agent).

One native action-space exception needs a stricter intersection: its hostname menu may expose a
discovered host for `PrivilegeEscalate` even when this RED slot has no session there, although the
action's own executor then fails immediately. With `own_obs`, the index therefore retains only
hostnames holding this slot's session. This is an observable precondition filter, not look-ahead.

BlockTrafficZone/AllowTrafficZone -> chokepoint (the design, genuine): the declared target is the
OPERATIONAL SERVER to protect; the compiler derives `(from_subnet, to_subnet)` from the fixed
zone-naming rule (each operational zone is reachable only via its own restricted zone) via the
owner map below, and enumerates the pair ONLY into that zone's owner slot, and only when that
slot has a valid session (never every slot -- avoids cross-slot duplicate blocks and a
session-less InvalidAction). The native validator (`replace_action_if_invalid`) SKIPS
`from_subnet`/`to_subnet` entirely (they are not action-space keys), so correctness lives
entirely in the NAMES this compiler derives -- tests assert the actual derived names, never just
"not InvalidAction" (which would be vacuous here).

Withdraw -> held_session (the design, genuine): pairs `(ip, hostname)` from the SAME held red session
-- never an ip x hostname cartesian (a red slot can know many more ips than it holds sessions on;
blindly cross-producting would offer mostly-misleading pairs). Native `execute` uses `hostname`
as the actual target and ignores `ip_address` (verified live), so the ip half only needs to
satisfy the constructor's non-optional parameter.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Mapping

from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.Action import InvalidAction

# Owner map + fixed zone-naming rule for the Block/Allow chokepoint (the design;
# mechanism spec the design). Deliberately duplicated from flipthem.anchors's identical
# RESOURCE_OWNER/RESOURCE_RESTRICTED_SUBNET/RESOURCE_OPERATIONAL_SUBNET (never imported from
# there): anchors.py will gain an llm:* dispatch branch that constructs an LLMPolicy ,
# which would make `anchors -> agent -> compiler -> anchors` a circular import if this module
# imported the constants back from anchors.py. Both copies encode the SAME fixed, low-churn game-
# topology fact (each operational zone is reachable only via its own restricted zone) and must be
# kept in sync if that topology ever changes -- neither is a "scripted strategy".
RESOURCE_OWNER = {
    "operational_zone_a_subnet_server_host_0": "blue_agent_1",
    "operational_zone_b_subnet_server_host_0": "blue_agent_3",
}
RESOURCE_RESTRICTED_SUBNET = {
    "operational_zone_a_subnet_server_host_0": "restricted_zone_a_subnet",
    "operational_zone_b_subnet_server_host_0": "restricted_zone_b_subnet",
}
RESOURCE_OPERATIONAL_SUBNET = {
    "operational_zone_a_subnet_server_host_0": "operational_zone_a_subnet",
    "operational_zone_b_subnet_server_host_0": "operational_zone_b_subnet",
}

BLUE_ACTIONS = ("Sleep", "Restore", "BlockTrafficZone", "AllowTrafficZone", "Analyse", "Remove")
RED_ACTIONS = ("Sleep", "DiscoverRemoteSystems", "AggressiveServiceDiscovery",
               "StealthServiceDiscovery", "ExploitRemoteService", "PrivilegeEscalate", "Withdraw")

_TARGET_KIND = {
    "Sleep": "none",
    "Restore": "hostname",
    "Analyse": "hostname",
    "Remove": "hostname",
    "PrivilegeEscalate": "hostname",
    "DiscoverRemoteSystems": "subnet",
    "AggressiveServiceDiscovery": "ip_address",
    "StealthServiceDiscovery": "ip_address",
    "ExploitRemoteService": "ip_address",
    "BlockTrafficZone": "chokepoint",      # the design: target = the operational server to protect
    "AllowTrafficZone": "chokepoint",      # the design: target = the operational server to protect
    "Withdraw": "held_session",            # the design: (ip, hostname) from the SAME held session
}


@dataclass(frozen=True)
class SlotActionIndex:
    slot: str
    session: int | None
    legal_pairs: tuple                      # tuple[(action_name, target_str | None), ...]
    _classes: dict = field(repr=False)      # action_name -> ActionClass, restricted to allowed
    _reverse: dict = field(repr=False)      # (action_name, target_str|None) -> native kwargs


def build_slot_action_index(slot: str, raw_action_space: dict, allowed_names,
                            target_hosts: tuple = (), own_obs: Mapping | None = None
                            ) -> SlotActionIndex:
    """`target_hosts` : the operational servers to protect -- threads the owner map into
    Block/Allow's enumeration (only the owning slot, only with a valid session; omit for a red
    slot or any caller that does not need Block/Allow, which then simply never enumerates them,
    unchanged from the first revision). `own_obs` : this slot's OWN legal observation dict (i.e.
    `team_obs[slot]`) -- needed ONLY to pair Withdraw's (ip, hostname) from the SAME held red
    session; omit for a blue slot or any caller that does not need Withdraw."""
    allowed_names = tuple(allowed_names)
    classes = {cls.__name__: cls for cls, ok in raw_action_space.get("action", {}).items()
               if ok and cls.__name__ in allowed_names}

    session_true = [s for s, ok in raw_action_space.get("session", {}).items() if ok]
    session = session_true[0] if session_true else None

    hostnames = [h for h, ok in raw_action_space.get("hostname", {}).items() if ok]
    subnets = [s for s, ok in raw_action_space.get("subnet", {}).items() if ok]
    ips = [ip for ip, ok in raw_action_space.get("ip_address", {}).items() if ok]

    pairs: list = []
    reverse: dict = {}

    def _add(action_name, target_str, kwargs):
        # Native observations can expose the same logical held host through several sessions.
        # The model-facing contract is a SET of legal (action, target) choices: repeating an
        # identical pair adds no capability, bloats the prompt, and caused dozens of duplicate
        # Withdraw entries in long campaigns.  Keep the first typed reverse mapping; equivalent
        # later entries compile to the same logical native target.
        key = (action_name, target_str)
        if key in reverse:
            return
        pairs.append(key)
        reverse[key] = kwargs

    for name in allowed_names:
        if name not in classes:
            continue
        kind = _TARGET_KIND.get(name)
        if kind == "none":
            _add(name, None, {})
            continue
        if session is None:
            # No active session on this slot (verified live: a red zone slot with no current
            # foothold there reports an EMPTY 'session' dict, e.g. red_agent_1..5 at reset --
            # only red_agent_0 starts with one) -- nothing else could ever compile, so only
            # Sleep is legal here. Matches design the design's decidable gate ("valid active
            # session AND ...") at the source, rather than leaving an uncompilable pair in
            # legal_pairs for context.py to filter out later.
            continue
        if kind == "hostname":
            # CybORG's raw action space exposes every currently-known hostname for
            # PrivilegeEscalate, even though native execution immediately fails unless THIS
            # red agent already owns a session on that exact host. Calling all of those pairs
            # "legal" led the live LLM to waste four turns escalating a scoring server from a
            # session on a different host. Production supplies own_obs, so narrow this action
            # to targets that satisfy its session precondition. Other hostname actions retain
            # the native action-space menu.
            eligible_hostnames = hostnames
            if name == "PrivilegeEscalate" and isinstance(own_obs, Mapping):
                held = set(_held_session_hostnames(own_obs, slot))
                eligible_hostnames = [h for h in hostnames if str(h) in held]
            for h in eligible_hostnames:
                _add(name, h, {"hostname": h})
        elif kind == "subnet":
            for s in subnets:
                _add(name, str(s), {"subnet": s})
        elif kind == "ip_address":
            for ip in ips:
                _add(name, str(ip), {"ip_address": ip})
        elif kind == "chokepoint":
            for host in target_hosts:
                if RESOURCE_OWNER.get(host) != slot:
                    continue   # the design: enumerate ONLY into the owner slot of that operational zone
                from_subnet = RESOURCE_RESTRICTED_SUBNET.get(host)
                to_subnet = RESOURCE_OPERATIONAL_SUBNET.get(host)
                if from_subnet is None or to_subnet is None:
                    continue
                _add(name, host, {"from_subnet": from_subnet, "to_subnet": to_subnet})
        elif kind == "held_session":
            for ip_str, hostname in _held_sessions(own_obs, slot, ips):
                _add(name, f"{ip_str}|{hostname}", {"ip_address": _to_ip(ip_str, ips),
                                                    "hostname": hostname})

    return SlotActionIndex(slot=slot, session=session, legal_pairs=tuple(pairs),
                            _classes=classes, _reverse=reverse)


def _to_ip(ip_str: str, known_ips):
    """Reverse an ip string back to the SAME typed object the action space itself uses."""
    for ip in known_ips:
        if str(ip) == ip_str:
            return ip
    return ip_str


def _held_session_hostnames(own_obs, slot: str) -> list[str]:
    """Hostnames on which this red slot currently has a session, from its own observation."""
    if not isinstance(own_obs, Mapping):
        return []
    held = set()
    for host, value in own_obs.items():
        if host in ("success", "action") or _looks_like_ip(host) or not isinstance(value, Mapping):
            continue
        for session in value.get("Sessions") or ():
            if isinstance(session, Mapping) and session.get("agent") == slot:
                held.add(str(host))
                break
    return sorted(held)


def _held_sessions(own_obs, slot: str, known_ips) -> list:
    """the design: pairs (ip, hostname) from the SAME held red session -- never an ip x hostname
    cartesian. Verified live (the design): a held session can appear under an
    IP-shaped obs key, a hostname-shaped obs key, or both, always sharing the SAME `session_id`
    when both exist; the very first foothold (present since reset) typically has only the
    hostname-shaped key. Every returned pair's hostname genuinely has an active session; the ip
    is the one sharing that session_id when known, else the first legally-known ip (Withdraw's
    native `execute` uses hostname as the target and ignores `ip_address` entirely -- verified
    live -- so this fallback never changes WHICH session is evicted, only satisfies the
    constructor's non-optional `ip_address` parameter)."""
    if not isinstance(own_obs, Mapping):
        return []
    by_session: dict = {}   # session_id -> {"ip": str|None, "hostname": str|None}
    for host, val in own_obs.items():
        if host in ("success", "action") or not isinstance(val, Mapping):
            continue
        for sess in (val.get("Sessions") or ()):
            if not isinstance(sess, Mapping) or sess.get("agent") != slot:
                continue
            sid = sess.get("session_id")
            entry = by_session.setdefault(sid, {"ip": None, "hostname": None})
            if _looks_like_ip(host):
                entry["ip"] = host
            else:
                entry["hostname"] = host

    fallback_ip = str(known_ips[0]) if known_ips else None
    out = []
    for entry in by_session.values():
        hostname = entry["hostname"]
        if not hostname:
            continue   # no hostname representation for this session -- Withdraw needs one
        ip_str = entry["ip"] or fallback_ip
        if ip_str is None:
            continue   # no ip at all is legally known yet -- cannot satisfy the constructor
        out.append((ip_str, hostname))
    return out


def _looks_like_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except (ValueError, TypeError):
        return False


def compile_action(index: SlotActionIndex, action_name: str, target):
    """Compile one (action_name, target_str|None) decision to a native action. Illegal/unknown
    (not in this slot's own legal_pairs, or no active session) -> InvalidAction; never raises."""
    key = (action_name, target)
    cls = index._classes.get(action_name)
    if cls is None or key not in index._reverse:
        return InvalidAction()
    kwargs = dict(index._reverse[key])
    if cls is not Sleep:
        if index.session is None:
            return InvalidAction()
        kwargs["session"] = index.session
        kwargs["agent"] = index.slot
    try:
        return cls(**kwargs)
    except Exception:
        return InvalidAction()

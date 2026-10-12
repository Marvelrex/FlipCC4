"""Model-visible action pseudocode generated from the execution metadata.

All actions use one fixed explanatory template. Ordering comes from the compiler's allowed action
tuples and duration/cost/target kinds come from the same metadata used by the policy layer. The
remaining text describes the checked-in CybORG action implementations; it is static game-rule
documentation and never exposes live hidden state.
"""
from __future__ import annotations

from flipthem.llm.compiler import BLUE_ACTIONS, RED_ACTIONS, _TARGET_KIND
from flipthem.llm.tools import action_manual


_DETAILS = {
    "Sleep": {
        "purpose": "Deliberately wait for this unit without changing the environment.",
        "model_target": "null",
        "conversion": "No target conversion. Sleep is constructed without session or agent parameters.",
        "native_call": "Sleep()",
        "preconditions": ("The unit must be currently decidable and (Sleep, null) must be listed for it.",),
        "execution": ("Return an empty Observation.",),
        "state_change": ("No host, process, session, route, firewall block, or knowledge state is changed.",),
        "observation": ("Empty Observation; no new target information.",),
        "failures": ("No action-specific execution failure after the compiler legality gate.",),
        "detection": "No network or host event is generated.",
    },
    "Restore": {
        "purpose": "Re-image a host and evict malicious activity, including privileged RED access.",
        "model_target": "one exact hostname offered for this unit",
        "conversion": "Use target unchanged as native hostname; inject this BLUE unit's current session.",
        "native_call": "Restore(session=current_session, agent=unit, hostname=target)",
        "preconditions": (
            "The submitted pair must be currently listed for the BLUE unit.",
            "The source session must still exist when execution completes.",
            "The BLUE agent must have an own session on target so RestoreFromBackup can be addressed.",
        ),
        "execution": (
            "Collect the BLUE agent's sessions located on target; fail if the set is empty.",
            "Choose one target session using the environment RNG.",
            "Execute RestoreFromBackup against that target session.",
            "Restore the host from backup, then return success.",
        ),
        "state_change": (
            "The target host is restored; malicious processes/files and RED sessions are removed.",
            "Sessions preserved by the restored host are re-registered in the global session table.",
            "If RED held root on a scored target, FT control returns to BLUE after completion.",
        ),
        "observation": ("Boolean success/failure; no direct disclosure of RED's hidden control state.",),
        "failures": (
            "The source session no longer exists.",
            "No BLUE session can be found on the chosen hostname at execution time.",
        ),
        "detection": "No stochastic action-specific detection rate; RED may later observe that its own session/root was lost.",
    },
    "BlockTrafficZone": {
        "purpose": "Add the fixed directional firewall block from a target's restricted zone to its operational zone.",
        "model_target": "the offered operational target hostname, not a subnet name",
        "conversion": (
            "operational_zone_a_subnet_server_host_0 -> "
            "(restricted_zone_a_subnet, operational_zone_a_subnet), owned by blue_agent_1; "
            "operational_zone_b_subnet_server_host_0 -> "
            "(restricted_zone_b_subnet, operational_zone_b_subnet), owned by blue_agent_3"
        ),
        "native_call": (
            "BlockTrafficZone(session=current_session, agent=unit, "
            "from_subnet=restricted_zone, to_subnet=operational_zone)"
        ),
        "preconditions": (
            "The pair must be listed for the target's owning BLUE unit.",
            "That unit's current session must still exist and be active.",
            "Both compiler-derived subnet names must exist in the simulator.",
            "The same directional block must not already exist.",
        ),
        "execution": (
            "Verify the source session is active.",
            "Verify from_subnet and to_subnet are valid subnet names.",
            "Check whether state.blocks[to_subnet] already contains from_subnet.",
            "If absent, append from_subnet to state.blocks[to_subnet] and return success.",
        ),
        "state_change": (
            "Traffic from the restricted zone into the operational zone becomes blocked.",
            "The block affects later remote actions whose route crosses this chokepoint.",
        ),
        "observation": ("Boolean success/failure for the firewall change.",),
        "failures": (
            "The BLUE unit has no active session.",
            "Either derived subnet name is invalid.",
            "The directional block already exists.",
        ),
        "detection": "No stochastic action-specific detection rate; this is a BLUE firewall state change.",
    },
    "AllowTrafficZone": {
        "purpose": "Remove the fixed directional firewall block for a target's approach path.",
        "model_target": "the offered operational target hostname, not a subnet name",
        "conversion": "Use the same hostname-to-subnet-pair and BLUE owner mapping as BlockTrafficZone.",
        "native_call": (
            "AllowTrafficZone(session=current_session, agent=unit, "
            "from_subnet=restricted_zone, to_subnet=operational_zone)"
        ),
        "preconditions": (
            "The pair must be listed for the target's owning BLUE unit.",
            "That unit's current session must still exist and be active.",
            "Both compiler-derived subnet names must exist in the simulator.",
            "The directional block must currently exist.",
        ),
        "execution": (
            "Verify the source session and both subnet names.",
            "Check whether state.blocks[to_subnet] contains from_subnet.",
            "If present, remove from_subnet from that list and return success.",
        ),
        "state_change": (
            "Traffic from the restricted zone into the operational zone is allowed again.",
            "Later remote actions may use that route if every other routing condition also holds.",
        ),
        "observation": ("Boolean success/failure for the firewall change.",),
        "failures": (
            "The BLUE unit has no active session.",
            "Either derived subnet name is invalid.",
            "The requested directional block is not currently present.",
        ),
        "detection": "No stochastic action-specific detection rate; this is a BLUE firewall state change.",
    },
    "Analyse": {
        "purpose": "Inspect a host for suspicious file evidence without directly revealing RED control.",
        "model_target": "one exact hostname offered for this unit",
        "conversion": "Use target unchanged as native hostname; inject this BLUE unit's current session.",
        "native_call": "Analyse(session=current_session, agent=unit, hostname=target)",
        "preconditions": (
            "The submitted pair must be currently listed for the BLUE unit.",
            "The BLUE agent must still have an own session on target at execution time.",
        ),
        "execution": (
            "Collect the BLUE agent's sessions located on target; fail if the set is empty.",
            "Choose one target session using the environment RNG.",
            "Run DensityScout and collect file name/path/density fields.",
            "Run SigCheck and collect file name/path/signature fields.",
            "Combine both sub-action observations.",
        ),
        "state_change": ("No host or session state is changed; only BLUE's legally observed evidence is updated.",),
        "observation": (
            "Success plus combined file density/signature evidence returned by both scanners.",
            "Evidence is a clue and is not a direct root/session truth value.",
        ),
        "failures": ("No BLUE session can be found on target at execution time.",),
        "detection": "No stochastic action-specific detection rate in the Analyse implementation.",
    },
    "Remove": {
        "purpose": "Stop suspicious non-root processes/footholds known to BLUE on a host.",
        "model_target": "one exact hostname offered for this unit",
        "conversion": "Use target unchanged as native hostname; inject this BLUE unit's current session.",
        "native_call": "Remove(session=current_session, agent=unit, hostname=target)",
        "preconditions": (
            "The submitted pair must be currently listed for the BLUE unit.",
            "The source BLUE session and an own BLUE session on target must still exist.",
        ),
        "execution": (
            "Read suspicious process IDs recorded by the source Velociraptor session for target.",
            "For each recorded suspicious PID, execute StopProcess on the selected target session.",
            "Return success after processing the known suspicious PIDs.",
        ),
        "state_change": (
            "Matching suspicious processes and associated non-root RED footholds can be removed.",
            "Privileged/root RED sessions persist; Restore is required to evict rooted access.",
        ),
        "observation": ("Boolean success/failure; no direct hidden-root disclosure.",),
        "failures": (
            "No BLUE session can be found on target at execution time.",
            "A rooted compromise remains even when the Remove action itself returns success.",
        ),
        "detection": "No stochastic action-specific detection rate in the Remove implementation.",
    },
    "DiscoverRemoteSystems": {
        "purpose": "Discover reachable, ping-responsive host IP addresses in a known subnet.",
        "model_target": "one exact CIDR subnet string offered for this unit",
        "conversion": "Resolve target to the same IPv4Network object stored in the current action space.",
        "native_call": (
            "DiscoverRemoteSystems(subnet=IPv4Network(target), "
            "session=current_session, agent=unit)"
        ),
        "preconditions": (
            "The submitted pair must be currently listed for the RED unit.",
            "The source RED session must still exist and be active.",
            "A reported host must respond to ping and have a reachable originating interface.",
        ),
        "execution": (
            "Construct Pingsweep with the source session and target subnet.",
            "For every IP assigned to the subnet, test respond_to_ping.",
            "For responsive hosts, verify a usable originating interface/route.",
            "Add each reachable IP and subnet interface to the returned observation.",
        ),
        "state_change": ("The network is unchanged; returned IPs enter RED's legally known discovery state.",),
        "observation": (
            "Reachable active IP addresses and subnet interface information.",
            "Success is false for a missing/inactive source; an empty sweep may contain no host entries.",
        ),
        "failures": (
            "The source session is missing or inactive.",
            "A host that does not respond or cannot be reached is omitted.",
        ),
        "detection": "Uses native Pingsweep behavior; the high-level action has no explicit detection_rate parameter.",
    },
    "AggressiveServiceDiscovery": {
        "purpose": "Rapidly enumerate services/ports on a known target IP with higher detection risk.",
        "model_target": "one exact IP address offered for this unit",
        "conversion": "Resolve target to the same IPv4Address object stored in the current action space.",
        "native_call": (
            "AggressiveServiceDiscovery(session=current_session, agent=unit, "
            "ip_address=IPv4Address(target))"
        ),
        "preconditions": (
            "The submitted pair must be currently listed for the RED unit.",
            "The source must be an active RedAbstractSession.",
            "No firewall block may prevent the source-to-target route.",
            "Portscan must be able to select an originating IP address.",
        ),
        "execution": (
            "Reject a missing/wrong-type source session or blocked route.",
            "Execute Portscan(target, detection_rate=0.75).",
            "Enumerate listening local ports from target process connections.",
            "Replace the source session's stored port knowledge for target with observed ports.",
        ),
        "state_change": (
            "RED's source session gains updated target-port knowledge.",
            "The target can receive NetworkConnection events that BLUE Monitor may later observe.",
        ),
        "observation": ("Success plus observed process/connection fields containing open local ports.",),
        "failures": (
            "Missing, inactive, or wrong-type source session.",
            "Blocked route or no originating interface.",
        ),
        "detection": "One RNG draw is shared across the scan; events are added when draw <= 0.75, and exploit decoys always add them.",
    },
    "StealthServiceDiscovery": {
        "purpose": "Enumerate services/ports on a known target IP more slowly with lower detection risk.",
        "model_target": "one exact IP address offered for this unit",
        "conversion": "Resolve target to the same IPv4Address object stored in the current action space.",
        "native_call": (
            "StealthServiceDiscovery(session=current_session, agent=unit, "
            "ip_address=IPv4Address(target))"
        ),
        "preconditions": (
            "The submitted pair must be currently listed for the RED unit.",
            "The source must be an active RedAbstractSession.",
            "No firewall block may prevent the source-to-target route.",
            "Portscan must be able to select an originating IP address.",
        ),
        "execution": (
            "Reject a missing/wrong-type source session or blocked route.",
            "Execute Portscan(target, detection_rate=0.25).",
            "Enumerate listening local ports from target process connections.",
            "Replace the source session's stored port knowledge for target with observed ports.",
        ),
        "state_change": (
            "RED's source session gains updated target-port knowledge.",
            "The target can receive NetworkConnection events that BLUE Monitor may later observe.",
        ),
        "observation": ("Success plus observed process/connection fields containing open local ports.",),
        "failures": (
            "Missing, inactive, or wrong-type source session.",
            "Blocked route or no originating interface.",
        ),
        "detection": "One RNG draw is shared across the scan; events are added when draw <= 0.25, and exploit decoys always add them.",
    },
    "ExploitRemoteService": {
        "purpose": "Use discovered service information to attempt a remote exploit and obtain a RED foothold.",
        "model_target": "one exact IP address offered for this unit",
        "conversion": "Resolve target to the same IPv4Address object stored in the current action space.",
        "native_call": (
            "ExploitRemoteService(ip_address=IPv4Address(target), "
            "session=current_session, agent=unit)"
        ),
        "preconditions": (
            "The submitted pair must be currently listed for the RED unit.",
            "The source must be a valid RedAbstractSession.",
            "Target ports must have been discovered in that source session.",
            "No firewall block may prevent the route.",
            "At least one exploit class must have nonzero applicability.",
        ),
        "execution": (
            "Build applicability weights for EternalBlue, BlueKeep, HTTP/HTTPS RFI, SSH brute force, SQL injection, Haraka RCE, and FTP traversal.",
            "Discard exploit classes with zero weight; fail when none remain.",
            "If one remains, select it. If several remain, the current selector removes the maximum-weight class and RNG-selects among the others because top-choice probability is 0.",
            "Execute the selected exploit over the route.",
            "On success, store learned hostname/OS information and the created session.",
        ),
        "state_change": (
            "A successful exploit creates a RED user session/foothold and associated process state.",
            "Exploit-specific network/process events can become observable to BLUE.",
        ),
        "observation": ("Success/failure, selected-exploit raw text, and exploit-specific process/session/system information.",),
        "failures": (
            "Missing/wrong-type source, undiscovered ports, blocked route, or no applicable exploit.",
            "The selected exploit can fail because of OS, patches, process state, credentials, or decoys.",
        ),
        "detection": "Determined by the selected exploit and route; this action has no single fixed detection_rate.",
    },
    "PrivilegeEscalate": {
        "purpose": "Upgrade a held RED session on a hostname to root/SYSTEM privilege.",
        "model_target": "one exact hostname offered for this unit",
        "conversion": "Use target unchanged as native hostname; inject this RED unit's current session.",
        "native_call": "PrivilegeEscalate(hostname=target, session=current_session, agent=unit)",
        "preconditions": (
            "The submitted pair must be currently listed for the RED unit.",
            "At least one RED session for this agent must still exist on target.",
            "A non-privileged session must not be a sandbox and must support an applicable escalation.",
        ),
        "execution": (
            "Collect this RED agent's sessions on target; fail if none exist.",
            "If any is already privileged, use it and mark success.",
            "Otherwise RNG-select a target session; a sandbox removes its process and fails.",
            "Select JuicyPotato for known Windows, otherwise V4L2KernelExploit, then execute it.",
            "On success, run ExploreHost through the privileged target session.",
            "If OTService is observed, mark it in the source session.",
        ),
        "state_change": (
            "On success, the chosen session/process becomes root or SYSTEM and an escalation artefact is added.",
            "Root/SYSTEM on a scored operational target changes FT control to RED.",
        ),
        "observation": ("Success/failure, privileged session information, and ExploreHost system/process details on success.",),
        "failures": (
            "No RED session remains on target.",
            "Sandbox, incompatible OS/process state, decoy prerequisite, or escalation sub-action failure.",
            "ExploreHost can fail if a required session disappears.",
        ),
        "detection": "No single high-level detection_rate; escalation changes a process and writes an artefact that can become evidence.",
    },
    "Withdraw": {
        "purpose": "Voluntarily remove this RED agent's sessions and processes from a held host.",
        "model_target": "one exact offered composite string: <ip>|<hostname>",
        "conversion": (
            "Resolve hostname to a held RED session and use its observed IP when available; if no "
            "IP-shaped observation exists, use a legally known IP only to satisfy the constructor, "
            "because native target selection uses hostname"
        ),
        "native_call": (
            "Withdraw(session=current_session, agent=unit, "
            "ip_address=paired_or_placeholder_ip, hostname=held_hostname)"
        ),
        "preconditions": (
            "The exact composite pair must be currently listed for the RED unit.",
            "The source session must still exist.",
            "A route from source hostname to target hostname must exist.",
            "This RED agent must still hold a removable session on target.",
        ),
        "execution": (
            "Find this agent's sessions on target hostname.",
            "Order child sessions first, non-primary parents next, and the source session last when it is on target.",
            "For each ordered session, execute StopProcess(stop_all=True).",
            "Stop and return failure if any StopProcess fails; otherwise return the last success.",
        ),
        "state_change": (
            "All successfully processed sessions/processes for this RED agent are removed from target.",
            "Withdrawing privileged access from a scored target can return FT control to BLUE.",
        ),
        "observation": ("The success/failure observation returned by the final attempted StopProcess.",),
        "failures": (
            "Missing source session, no route, no relevant held session, or any StopProcess failure.",
            "Withdrawing from the source host can remove the session used to perform the action.",
        ),
        "detection": "No explicit stochastic detection_rate in Withdraw.",
    },
}


_TEMPLATE_KEYS = (
    "purpose", "model_target", "conversion", "native_call", "preconditions", "execution",
    "state_change", "observation", "failures", "detection",
)


def action_reference_data(team: str) -> tuple[tuple[str, dict], ...]:
    """Return ordered, fully populated template data for one team's compiler whitelist."""
    if team == "blue":
        actions = BLUE_ACTIONS
    elif team == "red":
        actions = RED_ACTIONS
    else:
        raise ValueError(f"unknown team {team!r}; expected 'blue' or 'red'")
    for name in actions:
        missing = set(_TEMPLATE_KEYS) - set(_DETAILS[name])
        extra = set(_DETAILS[name]) - set(_TEMPLATE_KEYS)
        if missing or extra:
            raise RuntimeError(f"{name}: action template mismatch; missing={missing}, extra={extra}")
    return tuple((name, _DETAILS[name]) for name in actions)


def _bullets(lines) -> list[str]:
    return [f"  - {line}" for line in lines]


def render_action_reference(team: str) -> str:
    """Render every allowed action with the same field names and order."""
    lines = [
        "## ACTION PARAMETERS AND EXECUTION PSEUDOCODE (actual wrapper semantics)",
        "Your final JSON has unit, action, target, and wake. Only unit/action/target control",
        "execution; wake controls",
        "when you are asked again and is not an action parameter.",
        "COMMON PARAMETERS: unit is a currently-decidable unit ID; action is its exact ACTION heading;",
        "target is copied exactly from that unit's listed options.",
        "COMMON CONVERSIONS: hostname -> unchanged; ip_address -> the listed IPv4Address;",
        "subnet -> the listed IPv4Network. Inject current_session and agent=unit except for Sleep.",
        "",
        "COMMON COMPILER GATE FOR EVERY ACTION:",
        "  pair = (action, target)",
        "  if unit has no current decidable options: ignore its entry; busy actions continue",
        "  elif pair is not listed for a decidable unit:",
        "      execute InvalidAction(duration=0, FT_cost=0.1); requested action is not constructed",
        "  else:",
        "      recover typed parameters and native class from that unit's current action space",
        "      inject current_session and agent=unit when required, construct, then execute",
        "Listed records current compiler-menu membership. Execution separately checks source",
        "sessions, routes and action-specific prerequisites. Busy units are not charged again.",
        "FT action cost is charged once on completion, even if execution fails; Sleep is free.",
        "",
        "Every action below uses this identical template and field order.",
    ]
    for action, detail in action_reference_data(team):
        manual = action_manual(action)
        kind = _TARGET_KIND[action]
        conversion = (f"Common {kind} conversion." if kind in ("hostname", "ip_address", "subnet")
                      else detail['conversion'])
        preconditions = [line for line in detail["preconditions"]
                         if "must be currently listed" not in line
                         and "pair must be listed" not in line
                         and "composite pair must be currently listed" not in line]
        lines.extend((
            "",
            f"### ACTION: {action}",
            f"PURPOSE: {detail['purpose']}",
            "MODEL DECISION PARAMETERS:",
            f"  - target: {detail['model_target']}.",
            f"COMPILER TARGET KIND: {_TARGET_KIND[action]}.",
            f"COMPILER CONVERSION: {conversion}",
            f"NATIVE CONSTRUCTION: {detail['native_call']}.",
            "EXECUTION-TIME PRECONDITIONS:",
            *_bullets(preconditions or ("Common compiler gate.",)),
            "EXECUTION PSEUDOCODE:",
            *_bullets(detail["execution"]),
            "STATE CHANGE ON SUCCESS:",
            *_bullets(detail["state_change"]),
        "RETURNED OBSERVATION:",
            *_bullets(detail["observation"]),
            "FAILURE CONDITIONS / NO EFFECT:",
            *_bullets(detail["failures"]),
            f"DETECTION / VISIBILITY: {detail['detection']}",
            f"TIMING AND FT COST: duration={manual['duration']} tick(s); "
            f"FT action cost={manual['cost']}.",
        ))
    lines.extend((
        "",
        "EPISODE-HORIZON RULE: an action still in progress when the episode ends does not complete",
        "and is not charged its FT action cost.",
    ))
    return "\n".join(lines)

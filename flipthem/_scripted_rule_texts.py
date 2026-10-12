"""Scripted-rule specification text (testbed edition).

The research artefact pins each scripted rule to the sha256 of a frozen design-document section.
This testbed ships a restatement of the same rules written for this edition, so `rule_text_sha256()`
values here differ from the frozen research values on purpose; the DECISION LOGIC in scripted_policies.py is
unchanged and the testbed smoke exercises it on the real CC4 environment.
"""
RULE_TEXTS = {
    'S0': """S0 alert-restore `alert-restore-legal-v1`

Inherits the target-local rule of `restore-on-alert`:

1. At decision tick t=0 every slot Sleeps; the initial Processes snapshot is not an alert.
2. For t>=1 and each scored target, submit Restore only if this round's own legal Monitor alerts
   include the target, the owner slot is decidable and (Restore, target) is listed.
3. Otherwise Sleep. A busy owner does not queue old alerts; there is no cooldown and no retry
   after an execution failure.

The adapter must agree with the legal signal that `LegalContext.alerts` expresses for the two
targets. Alerts may come from ordinary business traffic or false positives and are never read as
confirmed compromise. All other BLUE slots Sleep. Parameters: skip one initial round; the
pending-alert queue is empty.
""",
    'S1': """S1 periodic restore `periodic10-legal-v1`

Frozen k=10, phase=0: at decision ticks 0, 10, ..., 190 each of the two owners submits a legal
Restore to its own scored target. Only decidability and the legal menu are checked, never
infection or alerts. A busy or unlisted opportunity is skipped with no catch-up and no drift of
the next scheduled tick. Every other tick and slot Sleeps.

k=10 is the pre-registered value of an existing baseline, not a choice tuned on a 5/10/20
comparison. Restore's native duration is 5 ticks; the period is an initiation interval, not a
wait after completion. Forty completed restores per match are not guaranteed.
""",
    'S2': """S2 one-shot chokepoint block `block-once-legal-v1`

Only at t=0, each owner selects the BlockTrafficZone listed by the compiler that protects its own
target, mapped as restricted_zone_[a/b]_subnet -> operational_zone_[a/b]_subnet. The slot must be
decidable and the action listed, otherwise the opportunity is skipped. The team Sleeps for the
rest of the match.

Frozen as a single attempt: no retry of a failed execution, no delayed resubmission, no Restore,
no Allow. It must never be silently turned into "block until it succeeds". Tests must confirm
the legal submission and the actual completion separately, not only interface acceptance.
Business impact is reported separately.
""",
    'A0': """A0 optional rule attacker `legal-rule-red-v1`

A deterministic, observation-limited scan -> exploit -> escalate state machine. It reuses the
compiler and native action body with a small rule brain. It is NOT the native `fsm` red: the
native FSM can select DiscoverDeception / Impact / DegradeServices and recognise decoys from raw
Processes and alter the exploit selector, which the model-driven agent cannot; filtering action
names alone does not remove that information gap.

It maintains only knowledge derivable from its own completion log: successfully scanned IPs,
explored subnets, its own legal hostname/IP map, its current sessions and roots, and the
submit/complete/fail times of every (unit, action, target). A failure never promotes state; a
lost session revokes the holding mark. Current holdings take precedence over "once succeeded".
Units are processed in name order; inside a unit one action is chosen from the current menu by
priority:

1. PrivilegeEscalate on a listed hostname that is not in the team's current `holdings.root`. The
   compiler menu already requires a session on the target; campaign-only per-unit root data is
   not read.
2. ExploitRemoteService on an IP this slot scanned successfully that is not in the team's set
   of known held IPs. That set is derived only from `holdings.sessions` joined with the legal
   `holdings.discovered.host_to_ips`; a missing join means unknown, not proven absent.
3. AggressiveServiceDiscovery on a visible IP not yet scanned successfully.
4. DiscoverRemoteSystems on a visible subnet not yet explored successfully.
5. If nothing above is available, refresh only subnets whose last successful exploration is at
   least 20 ticks old; otherwise Sleep.

A failed (unit, action, target) cools down for 10 ticks from the failed completion tick; the
target is never assumed permanently unreachable. Ties inside a priority are broken by ascending
submission count, ascending last submission tick and lexical target order, with an unsubmitted
time of minus infinity. During a cooldown the search continues with the next candidate or
priority. Only scored targets that are already legally associated may be reported; there is no
extra target ordering. Units share nothing beyond the RED team context.

Hosts already rooted are not escalated or exploited again; the team keeps exploring other legal
targets and does not stop because a non-scored host was rooted. This version never Withdraws and
never uses StealthServiceDiscovery; it uses a subset of the common action set and adds no hidden
detection, simulated rollouts or payoff oracle. The script has no RNG of its own and consumes no
environment RNG; the attack actions keep their native randomness. It is a pre-registered simple
reference policy, not a claimed strong attacker.

Its information ceiling is the ordinary RED LegalContext; the campaign-only
`red_sessions_by_unit` field is disabled. Scan, exploration and cooldown evidence is an internal
ledger built from the existing log's unit/action/target/result/round; missing mappings are handled
by the unknown rule above, never filled in from hidden state. The script may keep only its own
historical evidence and never widens the information available to the model-driven agent.
""",
}

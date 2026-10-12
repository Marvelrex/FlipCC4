"""Resource discovery and live compromise/service readers for FlipThem-on-CC4.

The two scored resources are the operational-zone OT servers; every other zone stays present as
approach terrain. Compromise and service are read live from CybORG state each tick and kept separate
(spec the design): compromise = a red privileged (root) session on the server; service = its OT service
still running.
"""
from __future__ import annotations

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Shared.Enums import ProcessName

OPERATIONAL_RESOURCES = (
    "operational_zone_a_subnet_server_host_0",
    "operational_zone_b_subnet_server_host_0",
)


class _Phase1PinnedScenarioGenerator(EnterpriseScenarioGenerator):
    """Hold the whole episode in mission phase 0 ("Preplanning" = spec 'the first revision').

    CC4 advances the phase only in SimulationController.step via
    State.check_next_phase_on_update_step, which buckets the step count into
    scenario.mission_phases (default ~T/3 each) and only ever increases the phase.
    Making phase 0 span every step means the check never advances past 0, so the
    phase-gated _update_agents_allowed_subnets (green comms policy) never switches
    away from phase 0, and BlueRewardMachine.get_phase_rewards -- read live off
    state.mission_phase -- stays on the phase-0 map. mission_phase + reward map +
    green comms thus stay consistent at the first revision (spec the design). Reset-safe: the
    scenario is regenerated through this override on every reset. No CybORG edit."""

    def _generate_mission_phases(self, steps):
        # phase 0 spans all `steps`; phases 1/2 get a nominal size so the
        # 3-tuple contract holds. check_next_phase tolerates total > steps.
        return (steps + 2, 1, 1)


def build_cc4_env(seed: int, steps: int = 100, red_agent_class=None) -> CybORG:
    """Build a reset CC4 env. Blue and red slots default to SleepAgent (runner-driven); green is the
    native EnterpriseGreenAgent. Pass a red_agent_class to let CC4 self-drive red (native anchors)."""
    sg = _Phase1PinnedScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=red_agent_class or SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        steps=steps,
    )
    env = CybORG(sg, seed=seed)
    env.reset()
    return env


def discover_resources(env: CybORG) -> list[str]:
    hosts = env.environment_controller.state.hosts
    found = [r for r in OPERATIONAL_RESOURCES if r in hosts]
    assert len(found) == 2, f"expected 2 operational servers, found {found}"
    return found


def compromise_of(env: CybORG, resource: str) -> str:
    state = env.environment_controller.state
    for agent, sessions in state.sessions.items():
        if "red" not in agent:
            continue
        for sess in sessions.values():
            if sess.hostname == resource and sess.has_privileged_access():
                return "red"
    return "blue"


def service_state(env: CybORG, resource: str) -> str:
    services = env.environment_controller.state.hosts[resource].services
    svc = services.get(ProcessName.OTSERVICE)
    return "up" if (svc is not None and svc.active) else "stopped"

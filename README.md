# FlipCC4

A resource-control testbed built on CybORG CAGE Challenge 4. Red and Blue agents compete for privileged control of network resources. Payoffs combine holding value with native action costs charged at completion.

The testbed supports legal observation and action interfaces, configuration switching, and shared-prefix branching. The included smoke experiment uses deterministic offline clients in the real CC4 environment to check these mechanisms.

## Quick start

Tested with Python 3.10 on Windows.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python smoke.py
```

After installation, the smoke runs offline without API keys. It returns `0` when all checks pass and `1` on failure, and writes a JSON report to `receipts/`.

## Smoke coverage

| Area | Checks |
| --- | --- |
| Environment and actions | Scenario startup, local observations, legal actions, and invalid-action handling |
| Resource scoring | Control changes, holding payoffs, and costs charged once at action completion |
| Configuration switching | One B0-to-B1 team switch, with ongoing actions, state, and remaining budget preserved |
| Shared-prefix branching | Checkpoint creation, independent restores, branch continuation, and segment payoffs |
| Offline execution | Network blocking and deterministic-client replay checks |

The smoke checks testbed functionality using offline clients. Model-based evaluation is separate.

## Configuration and interfaces

Start with [`testbed/configs/smoke.json`](testbed/configs/smoke.json) to inspect the scenario, budgets, and switching rule.

```bash
python smoke.py --config testbed/configs/smoke.json --out receipts --keep-going
```

`--keep-going` runs the remaining checks after a failure. The runner sets `PYTHONHASHSEED=0` for cross-process checkpoint checks.

The configuration profiles are **B0** (context-assisted Blue), **B1** (tool-assisted Blue), **R0** (baseline Red), and **R1** (Red with task memory, planning, and execution feedback). Scripted policies and native CybORG agents are also supported.

| Entry point | Purpose |
| --- | --- |
| [`FlipThemCC4`](flipthem/wrapper.py) | Reset and step the environment, obtain team observations, and calculate payoffs |
| [`build_match`](testbed/config.py) | Build a match from a configuration and an injected client factory |
| [`SwitchingTeamPolicy`](flipthem/switching_policy.py) | Switch configurations while sharing execution state and budget |
| [`testbed/fork.py`](testbed/fork.py) | Save checkpoints, restore branches, and calculate segment payoffs |
| [`testbed/fake_client.py`](testbed/fake_client.py) | Offline clients for deterministic and scripted decisions |

The pinned dependencies cover the testbed and smoke entry point. Other tools bundled with the upstream simulator may require additional packages.

## License

The testbed is released under the [MIT License](LICENSE). The bundled CybORG simulator retains its upstream license and attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

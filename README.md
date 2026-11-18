# Decentralised RL feasibility model

An analytic roofline model of one reinforcement-learning step (rollout, verification, update, weight
broadcast), validated against published runs, plus a solver that asks the compute-governance
question: how much RL compute can an actor without a datacentre spend, on how many GPUs, at how many
sites, in how long. ERA:AI Fellowship project.

## Setup

```bash
pip install -r requirements.txt
```

The model core (`specs.py`, `model.py`, `penalties.py`, `feasibility/`) is stdlib-only. pandas and
matplotlib are for the figures, Streamlit for the explorer, pytest for the tests.

## Use

```bash
python -m analysis build
```

Writes `results/dashboard.html` (every figure and table, light/dark toggle), paper exports in
`results/figures/` (PDF/SVG/PNG, untitled) and the data behind each in `results/tables/`. Rebuilds
only redraw figures whose numbers or styling changed (`.cache/`).

```bash
python -m analysis explore
```

Interactive explorer: change the threat model in a sidebar and every chart recomputes.

| Command | |
|---|---|
| `python -m analysis list` | every figure and table name |
| `python -m analysis show NAME ...` | print a figure's data or a table in the terminal |
| `python -m analysis build NAME ...` | build only those |
| `--set FIELD=VALUE` | override any `FeasConfig` field, e.g. `--set wan_mbps=10000 --set model=Llama3-70B` |
| `--penalties` | apply Rahman's distributed-training penalties |
| `--batch-mode scaled\|isocompute\|flat` | batch size per RL budget |
| `--theme dark --font-scale 1.5` | slide or poster exports |
| `python -m validation.validate` | the model against every published run, in full |

## Tests

```bash
python -m pytest
```

Fast checks in a few seconds: solver invariants on a sample of configs, the golden headline numbers,
the CLI. Add `--full` for the whole 864-point invariant gate (with and without penalties) and a check
that every committed `results/tables/*.csv` is what the code produces now. After an intended change
to the headline numbers, re-record them with `python -m tests.invariants --write`.

## Layout

| Path | |
|---|---|
| `specs.py` | dataclasses describing one RL run: model, RL, algorithm, hardware, network, verifier, penalties |
| `model.py` | one RL step: compute, memory, rollout, verification, broadcast, `simulate()` |
| `penalties.py` | Rahman's distributed-training penalties (stragglers, goodput, sync quality), off by default |
| `feasibility/` | the solver: `hardware` (GPU families, export and power caps), `targets` (RL budgets, thresholds), `models`, `config` (`FeasConfig`), `solver` (min cluster, optimal split, max compute, required link) |
| `analysis/` | every figure and table: `scenarios` → `sweep` → `figures/` → `render` (matplotlib, ERA theme) → `build` (exports, dashboard); `app.py` is the explorer |
| `validation/` | published-run presets, inverse calibration, Monte-Carlo intervals; `REPORT.md` reads the results |
| `experiments/` | measurement scripts (response lengths, oversampling, agentic traces) and their write-ups |
| `results/` | build outputs only: dashboard, figures, tables (regenerate with `python -m analysis build`) |
| `data/` | inputs: `reasoning_models.csv` (published reasoning models, the provenance of the RL budgets) and `lengths/` (measured response-length summaries behind section 3 and E[R]) |
| `tests/` | pytest suite and the invariant gate (`invariants.py`, `golden.json`) |

## Docs

- `GUIDE.md`: how to run a scenario, add a model, GPU, budget, variant, figure or validation run, and test.
- `MODEL_CALCULATIONS.md`: every equation the code computes, with notation and code identifiers.
- `SIMULATOR_MAP.md`: spec-to-code map, deliberate deviations, and the known-gaps register.
- `validation/REPORT.md`: what the validation results mean.
- `experiments/*.md`: how each free parameter was measured.

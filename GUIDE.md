# How-to guide

Recipes for the common jobs: running a scenario, adding a model / GPU / budget / variant / figure /
validation run, and testing. `README.md` has the layout; `MODEL_CALCULATIONS.md` the equations.

## How a run is configured

Everything the solver needs is one `FeasConfig` (`feasibility/config.py`). Its defaults *are* the
threat model: Llama-3.1-405B on H100, 16-GPU inference nodes, 128-GPU trainer nodes at separate sites,
1 Gbps WAN, 16x weight broadcast, 150x gradients, E[R] 20k, ω 1.5, 1.5M-GPU stock.

Each figure builds its configs through `analysis.scenarios.scenario()`, which layers settings in a
fixed order (later wins):

```
FeasConfig defaults  <-  your overrides (--set / explorer sidebar)  <-  the chart's variant  <-  the target's batch  <-  the chart's own axis
```

So your overrides change everything except what a chart is *about*. A decentralised-vs-centralised
chart still sets its own link speed and node size; a cross-model chart still sweeps every model;
2t is dense 405B and 2u MoE by design. The 1x validation charts use each published run's own config.

## Run a scenario

**From the command line.** Any `FeasConfig` field can be set; repeat `--set` for several.

```bash
python -m analysis show frontier table_runs --set model=Llama4-402B-A17B
```

```bash
python -m analysis build --set gpu=B200 --set wan_mbps=10000 --out results/b200_10g
```

Values parse as numbers or `true`/`false`; `model` takes a key of `MODELS`: `Llama3-8B`, `Llama3-70B`,
`Llama3-405B`, `Qwen3-30B-A3B`, `Qwen3-235B-A22B`, `Kimi-K3`, `Llama4-402B-A17B`. `python -m analysis
list` names the figures and tables. Use a separate `--out` so you don't overwrite `results/`.

| Common knob | Field | Default |
|---|---|---|
| model / GPU family | `model`, `gpu` | `Llama3-405B`, `H100` |
| inference node size / trainer node size | `gpus_per_node`, `train_gpus_per_node` | 16, 128 |
| WAN link (Mbps), broadcast compression | `wan_mbps`, `compression` | 1000, 16 |
| distributed trainer, gradient compression | `train_distributed`, `grad_compression` | true, 150 |
| staleness k | `sync_interval` | 1 |
| response length, its CV, oversampling | `er`, `er_cv`, `omega` | 20000, 0.5, 1.5 |
| batch | `prompts_per_batch`, `responses_per_prompt` | 128, 512 (but see `--batch-mode`) |
| MFU | `mfu_train`, `mfu_inf` | 0.40, 0.30 |
| inference precision | `b_weights_inf`, `b_kv`, `inf_flop_mult` | 2, 2, 1 (4-bit: 0.5, 0.5, 4) |
| stock, $/GPU-h, site power cap (MW) | `stock_gpus`, `gpu_hr_usd`, `site_power_mw` | 1.5M, 2, 5 |

Plus two run-wide switches: `--penalties` (Rahman's distributed-training penalties) and
`--batch-mode scaled|isocompute|flat` (how the batch grows with the RL budget).

**Interactively.** `python -m analysis explore` exposes the main knobs in a sidebar. Restart it after
editing code: it caches figures per setting.

**From Python.** For a one-off question, skip the figure layer:

```python
from feasibility import FeasConfig, MODELS, evaluate, max_compute_operating_point, steps_and_c_rl, with_cfg

cfg = FeasConfig(target_c_rl=2.5e25, model=MODELS["Llama3-70B"], gpu="B200", wan_mbps=10_000)
ev = evaluate(cfg, 180 * 86400)              # min cluster that finishes in 180 days, else the full-stock floor
ev["feasible"], ev["gpus"], ev["wall_d"], ev["bottleneck"]

op = max_compute_operating_point(with_cfg(cfg, stock_gpus=100_000))
steps_and_c_rl(op, 180 * 86400)              # (steps, max C_RL) for that stock in 180 days
```

`with_cfg(cfg, **changes)` copies a config with fields changed. `simulate(cfg.build(n_train_gpus,
n_inf_gpus))` from `model.py` gives the full per-stage breakdown of one step.

## Add a model

`feasibility/models.py`, `MODELS`: one `ModelSpec` from the model's `config.json`.

```python
"Qwen3-32B": ModelSpec(name="Qwen3-32B", is_moe=False, p_total=32.8e9, p_active=31.2e9,
                       d_model=5120, n_layers=64, n_q_heads=64, n_kv_heads=8, head_dim=128, vocab=151936),
```

`p_active` is decoder-layer parameters only (no embedding or head); for MoE, the active experts plus
shared layers. For multi-head latent attention, add `attention="mla"` and its four widths
(DeepSeek-V3: `kv_lora_rank=512, qk_rope_dim=64, qk_nope_dim=128, v_head_dim=128`; see the
`"DeepSeek-V3"` entry). MLA caches only the latent and the RoPE key, decodes absorbed and trains naive. It then appears in the explorer, `--set model=Qwen3-32B`, and the cross-model charts
(optionally a short axis label in `MSHORT`, `analysis/figures/feasibility.py`).

## Add a GPU family

`feasibility/hardware.py`, `GPUS`: dense BF16 FLOP/s, HBM GB, HBM TB/s, board watts. Add it to
`HIGH_END_GPUS` too if it should appear in the GPU-family charts (2e, 2f) and the explorer.

## Add or change an RL budget

`feasibility/targets.py`, `MODEL_TARGETS`: `name: (C_RL, provenance note)`, with the source in
`data/reasoning_models.csv`. Every per-target chart and table picks it up. Batches are assigned by rank in budget: x4 per rung from 256 in `scaled` mode, along
`ISO_N_BY_RUNG` (saturating at the last rung) in `isocompute` mode.

## Add a named variant

`analysis/scenarios.py`, `VARIANTS`: a dict of `FeasConfig` overrides a chart applies on top of
yours, e.g. `"fast_link": dict(wan_mbps=10_000)`. Use it as `scenario(opts, "fast_link", target=t)`.

## Add a figure or table

Write a function in the right `analysis/figures/*.py` module and register it. A figure is a tidy
DataFrame from `sweep()` plus a `Fig` spec; the renderer handles hollow bars, labels and both
themes.

```python
@register("fast_link_gpus", "2")                         # name, dashboard section
def fast_link_gpus(opts):
    data = sweep({"target": list(TARGETS), "wan_mbps": [1_000, 10_000]},
                 lambda p: min_gpus(scenario(opts, target=p["target"], wan_mbps=p["wan_mbps"]), 180))
    return Fig("2v", "Minimum GPUs at 1 vs 10 Gbps (180 days)", "bars", data, x="target", series="wan_mbps",
               ylabel="GPUs to finish in 180 days", x_fmt=target_label, series_fmt=lambda m: f"{m / 1000:g} Gbps")
```

- Metrics (`analysis/sweep.py`): `min_gpus(cfg, window_d)`, `max_compute(cfg, stock, window_d)`,
  `required_link(cfg, window_d)`. Each returns the plotting columns (`value`, hollow `fallback`, `note`)
  plus details for tables, and is cached on disk.
- Kinds: `bars` (grouped; `log`, `refs` for dashed reference lines, `inner` for text in the bars),
  `heatmap`, `tornado`, `hbars`, `multiples`.
- A table returns `Table(id, title, dataframe, caption=...)` instead.

Then `python -m analysis build fast_link_gpus`, and `python -m analysis build` to refresh the
dashboard. Commit the regenerated `results/` with the code (the `--full` tests check they match).

## Add a validation run

`validation/presets.py`: a function returning a `Scenario` with the run's logged config and
`published=dict(t_step=...)`, then one line in `published_runs()`. That list feeds both
`python -m validation.validate` and charts 1a-1c / table A4 (runs without a published step time are
reported but not charted).

## Add measured response lengths

Put the summary JSON in `data/lengths/` (`lengths_io.py` reads every schema the measurement scripts
write) and add it to the model lists at the top of `analysis/figures/lengths.py`. Raw traces stay in
`experiments/lengths/`, which is not tracked.

## Test

```bash
python -m pytest
```

Seconds: solver invariants on a sample of configs, the 13 golden headline numbers, the CLI.

```bash
python -m pytest --full
```

About 5 minutes: the 864-point invariant gate with penalties off, async and relay, a check that
every `results/tables/*.csv` matches what the code produces now, the explorer run headlessly
(default settings and a model that misses most windows), and every figure rendered under five
contrasting configs with no overlapping or off-figure text (`tests/test_render.py`; add a config there
when a new option could change chart layout).

- **A golden number moved on purpose?** Re-record with `python -m tests.invariants --write`, and say
  why in the commit.
- **`test_results_reproduce` failed?** The code changed a result: rebuild with
  `python -m analysis build`, check the diff in `results/tables/`, commit both.
- **New solver behaviour?** Add the property to `check()` in `tests/invariants.py`: it runs at every
  grid point. For a one-off case, add a plain test to `tests/test_solver.py`.

`python -m validation.validate` is the full model-vs-published report; its output should only change
when you mean the model to.

## Caches

`.cache/` holds solver results (keyed by a hash of the model code, so editing the model invalidates
them) and a fingerprint of every drawn figure (so unchanged figures are skipped). Deleting it is
always safe; the next build just takes ~1 minute.

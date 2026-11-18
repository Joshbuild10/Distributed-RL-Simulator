# SIMULATOR_MAP — spec ↔ code reconciliation

> **DRAFT, auto-assembled from the code on 2026-08-17; refreshed 2026-11-10 (ω in waves, A-iii, A-x closed).** Intended to be redrafted by hand.
> Purpose: satisfy PROJECT_CONTEXT §4 ("inventory actual modules/functions vs the spec, flag
> discrepancies rather than silently fixing them"). Numbers here are reproduced by
> `python -m validation.validate`.

## Repo layout

| Path | Role |
|---|---|
| `penalties.py` | Rahman's distributed-training penalties (straggler tails, goodput, η factors), ported verbatim; toggled by `PenaltySpec` |
| `MODEL_CALCULATIONS.md` | Every equation the code computes, with notation and code identifiers (report Appendix B) |
| `specs.py` | Dataclasses: `ModelSpec`, `RLSpec`, `AlgoSpec`, `HWSpec`, `NetSpec`, `VerifySpec`, `Scenario` |
| `model.py` | Analytic core: `training_compute`, `training_time`, `training_memory`, `rollout_terms`, `verify_time`, `broadcast_time`, `n_steps`, `simulate` |
| `validation/presets.py` | Published run scenarios (INTELLECT-2/3, prime-rl ×2, AReaL ×4) + hardware/model constants |
| `validation/solvers.py` | Inverse calibration (`solve_for_response_len`, `solve_for_capacity`, `solve_for_mfu`) |
| `validation/reporting.py` | `report()`, `CALIBRATION_METRICS`, `sweep_response_len()` |
| `validation/validate.py` | Entry point: report every scenario + inverse-calibration summary |
| `feasibility_core.py` | `FeasConfig`, `evaluate`, `min_gpus_for_tsb`, `min_cluster`, `optimal_split`, model/GPU/target registries, formatters |
| `feasibility.py` | Sweeps, tables, CSV writers, CLI |
| `analysis/` | Every figure and table: `scenarios` (threat model + variants) -> `sweep` (cached evaluations -> DataFrames) -> `figures/` (one module per section) -> `render` (matplotlib, ERA theme) -> paper exports, `results/dashboard.html`, and the Streamlit explorer. `python -m analysis [build\|explore\|list]` |
| `results/` | Build outputs: `dashboard.html`, `figures/` (PDF/SVG/PNG), `tables/` (CSV); `lengths/` (measured response-length summaries the figures read) |
| `lengths_io.py`, `svgkit.py` | Length-file reader (used by `analysis`); legacy SVG primitives (now only the gitignored experiment plots) |
| `experiments/` | Measurement campaign, invariant gate `check_invariants.py` (gitignored for now) |

## Spec → implementation

| Spec item (working doc §Appendix B) | Where | Status |
|---|---|---|
| `T_step = max(T_rollout+T_verify, T_update, T_broadcast)` async | `model.simulate` | ✅ when `algo.sync_interval >= 1` |
| serial (sum) when on-policy | `model.simulate` | ✅ when `sync_interval = 0`; solver partitions the budget to match |
| `N_steps = ceil(N_prompts/batch) · epochs` | `model.n_steps` | ✅ (`n_steps_override` wins) |
| `C_update = (3+recomp_act+recomp_old)(C_layers + C_attn)` | `model.training_compute` | ✅ |
| `C_layers = 2·P_active·tok_batch` | `model.training_compute` | ✅ |
| Causal attention coefficient 2 (not 4) | `AlgoSpec.causal_coef` | ✅ |
| `T_update = C_update/(FLOPs·MFU·Penalty_para·nodes)` | `model.training_time` | ✅ |
| Memory: weights/grads/optimiser/activations, ZeRO-3 | `model.training_memory` | ✅ computed **and enforced** — `total_time` rejects `not mem.fits` |
| KV/token `= b_kv·2·L·K·H` | `model.rollout_terms` | ✅ |
| `seq_par = floor(free_mem / KV_provision)` | `model.rollout_terms` | ✅ + `kv_provisioning ∈ {peak,expected}` |
| `T_decode = max(comp, mem)` roofline | `model.rollout_terms` | ✅ |
| `T_prefill`, `T_inf_attn`, `T_env` | `model.rollout_terms` | ✅ |
| `T_rollout = (…)·n_microbatches·ω` | `model.rollout_terms` | ✅ (see deviation 1); ω enters as `ω·B` sequences in the waves, so oversampling fills spare groups in parallel rather than multiplying time |
| `T_verify` rule- and model-based | `model.verify_time` | ✅ |
| `T_broadcast = vol/bw + latency` | `model.broadcast_time` | ✅ one-directional (Rahman's has ×2) |
| Policy age (= `sync_interval`) | `model.simulate` | ✅ reported as `policy_age`; **not wired into `N_steps`** |
| Mode: target time/FLOP → required nodes | `feasibility_core.min_cluster` | ✅ |
| Mode: optimal trainer:inference split | `feasibility_core.optimal_split` | ✅ derived exactly (no preset fraction grid) |
| Inference tensor parallelism across nodes | `model.rollout_terms`, `HWSpec.inf_tp` | ✅ serving group of `inf_tp` nodes, one weight copy |

## Deliberate deviations from the spec (do not "fix")

1. **Per-wave mean fill.** Rollout terms charge `seqs_per_wave = ω·size_batch/(active_groups·waves)`, not a
   full `seq_par` per microbatch — the `ceil()` overshoot double-charged the last, mostly-empty
   wave and got worse as `seq_par` rose. This alone took INTELLECT-3 from +35% → +5% at the time.
2. **MoE decode bandwidth.** `mem_weights_decode = b·(p_active + p_head)`, clamped at
   `b·p_total`, is what `t_dec_mem` reads; HBM *capacity* still uses `p_total` (all experts
   resident) and KV traffic stays dense. Dense models are unaffected (`active+head ≈ total`).
3. **Two throughput conventions.** Stage-local (`tok_s_train`, `tok_s_inf`) vs whole-step
   (`tok_s_inf_achieved`). Both are reported; `CALIBRATION_METRICS` documents which one each
   published figure is compared against and why.
4. **Trainer node = one full replica.** Each update node holds the whole model (weights + grads +
   optimiser + the resident activation), sharded only *within* the node (`n_shard` = the node's own
   GPUs). Multiple trainer nodes are data-parallel replicas, so there is no cross-node model
   parallelism and no inter-node sharding cost. Nodes are up to `train_gpus_per_node` (128) GPUs; a
   SINGLE trainer node is right-sized DOWN in `GPU_STEP` (16) chunks to the minimum that both fits
   the replica and keeps the update off the critical path. `act_site_parallel=True` shards the
   activation within the node (so it does not grow with node size); the data-parallel per-GPU path
   (`False`) is kept for the calibrated validation presets. Not modelled, and would only tighten
   the fit: the largest gathered FSDP unit, collective/workspace buffers, master-copy dtype.
5. **`oversample_override` precedence.** The override wins over `success_rate` and defaults to 1,
   so the derived zero-advantage path only runs when explicitly enabled — measured ω is preferred
   to a derived one, and the single-`p` form understates ω under bimodal difficulty.

## Known gaps (carried, not silently patched)

| # | Gap | Consequence |
|---|---|---|
| A-i | Single-turn only | Multi-turn agentic effective context (measured ~2k–123k median across four environments) not represented |
| A-ii | `penalty_para = 1.0`; no intra-group FSDP communication | Trainer optimistic at high FSDP shard degree — the residual in the AReaL 14B/32B rows. Rahman's penalties (`PenaltySpec`) do **not** close it: they charge sync *between* replicas and pipeline handoffs, not FSDP traffic *within* one shard group |
| A-iii | No Rahman η / DiLoCo outer loop | `sync_interval` k ≥ 2 amortises the broadcast over k steps (per-step cost ÷ k), but its convergence cost is not charged (see A-vi), so large k reads as a free win. At the default k = 1 weights are broadcast every step |
| A-iv | Inference limited to ONE serving group | A group is `inf_tp` tensor-parallel nodes (`inf_tp_max=1` ⇒ a single node). TP collectives (per-layer AllReduce; AllToAll once KV shards past `n_kv_heads`) are **not** charged — `inf_tp_max` stands in for what the fabric sustains |
| A-v | Trainer all-reduce is a plain ring over one link | With `train_distributed` (default) the 128-GPU trainer nodes sit at separate sites and all-reduce gradients every step (H = 1) over the WAN at `grad_compression` (150x, Rahman): 87 s at 1 Gbps, below the broadcast; it would bind below ~32x (810 s at 16x). No hierarchical/topology-aware reduction, gradient sparsity beyond `compression`, or overlap with backward beyond `max(T_comp, T_sync)`. Under-cap (16-GPU, pipelined) trainers are representable but not admitted by the solver |
| A-vi | Staleness quality penalty is a diagnostic | `policy_age` reports the configured `k`; its sample-efficiency cost is not wired into `N_steps` (would create a fixed point) |
| A-vii | Stage times use the MEAN response length; heterogeneous hardware only via lumped `capacity_mult` | Assumes continuous batching keeps every slot full across step boundaries. In the single-wave regime (the decentralised ceiling) the last sequence to finish is near `R_max`, not `R`, so the rollout floor is optimistic by up to `R_max/R` (2x at the defaults). Hardware fail-slow tails are coded (`PenaltySpec.inf_mode`, async by default); the response-length tail is not |
| A-viii | Facility power factor is a **flat constant** | `FACILITY_PUE = 1.6` multiplies board TDP in `node_gpus_power_cap`/`sites_power`, so every power figure is draw at the meter. One number stands in for host/network/conversion/cooling across every GPU family and site design |
| A-ix | `stock_gpus` counts **physical chips** | Not renormalised to H100-equivalents when `gpu` changes, so cross-family sweeps compare fixed chip counts, not fixed compute |

## Unit conventions

Times **seconds**, memory **bytes**, compute **FLOP**, bandwidth **bytes/s** (`NetSpec` converts
from Mbps). FLOP/s and bandwidth are **per node**, where a "node" is one inference replica or one
FSDP shard group (or an `inf_tp`-node serving group) — `gpus_per_node` converts to GPU counts. Quadratic terms use
`E[R²] = E[R]²(1+cv²)`.

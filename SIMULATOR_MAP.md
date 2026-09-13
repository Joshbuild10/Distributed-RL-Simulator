# SIMULATOR_MAP — spec ↔ code reconciliation

> **DRAFT, auto-assembled from the code on 2026-08-17; refreshed 2026-09-09 (layout, A-viii, A-x).** Intended to be redrafted by hand.
> Purpose: satisfy PROJECT_CONTEXT §4 ("inventory actual modules/functions vs the spec, flag
> discrepancies rather than silently fixing them"). Numbers here are reproduced by
> `python -m validation.validate`.

## Repo layout

| Path | Role |
|---|---|
| `specs.py` | Dataclasses: `ModelSpec`, `RLSpec`, `AlgoSpec`, `HWSpec`, `NetSpec`, `VerifySpec`, `Scenario` |
| `model.py` | Analytic core: `training_compute`, `training_time`, `training_memory`, `rollout_terms`, `verify_time`, `broadcast_time`, `n_steps`, `simulate` |
| `validation/presets.py` | Published run scenarios (INTELLECT-2/3, prime-rl ×2, AReaL ×4) + hardware/model constants |
| `validation/solvers.py` | Inverse calibration (`solve_for_response_len`, `solve_for_capacity`, `solve_for_mfu`) |
| `validation/reporting.py` | `report()`, `CALIBRATION_METRICS`, `sweep_response_len()` |
| `validation/validate.py` | Entry point: report every scenario + inverse-calibration summary |
| `feasibility_core.py` | `FeasConfig`, `evaluate`, `min_gpus_for_tsb`, `min_cluster`, `optimal_split`, model/GPU/target registries, formatters |
| `feasibility.py` | Sweeps, tables, CSV writers, CLI |
| `results/plot_dashboard.py` | The whole `results/dashboard.html` figure set (inline SVG, no matplotlib) |
| `experiments/` | Measurement campaign, `svgkit.py` chart primitives, `lengths_io.py` (gitignored for now) |

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
| `T_rollout = (…)·n_microbatches·ω` | `model.rollout_terms` | ✅ (see deviation 1) |
| `T_verify` rule- and model-based | `model.verify_time` | ✅ |
| `T_broadcast = vol/bw + latency` | `model.broadcast_time` | ✅ one-directional (Rahman's has ×2) |
| Policy age (= `sync_interval`) | `model.simulate` | ✅ reported as `policy_age`; **not wired into `N_steps`** |
| Mode: target time/FLOP → required nodes | `feasibility_core.min_cluster` | ✅ |
| Mode: optimal trainer:inference split | `feasibility_core.optimal_split` | ✅ derived exactly (no preset fraction grid) |
| Inference tensor parallelism across nodes | `model.rollout_terms`, `HWSpec.inf_tp` | ✅ serving group of `inf_tp` nodes, one weight copy |

## Deliberate deviations from the spec (do not "fix")

1. **Per-wave mean fill.** Rollout terms charge `seqs_per_wave = size_batch/(n_nodes·waves)`, not a
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
| A-ii | `penalty_para = 1.0` | Trainer optimistic at high FSDP shard degree — the residual in the AReaL 14B/32B rows |
| A-iii | No Rahman η / DiLoCo outer loop; weights broadcast **every** step | WAN/broadcast bound is *pessimistic*; `compression` stands in for the sync interval |
| A-iv | Inference limited to ONE serving group | A group is `inf_tp` tensor-parallel nodes (`inf_tp_max=1` ⇒ a single node). TP collectives (per-layer AllReduce; AllToAll once KV shards past `n_kv_heads`) are **not** charged — `inf_tp_max` stands in for what the fabric sustains |
| A-v | No inter-learner communication term | Multiple trainer nodes are data-parallel replicas with no sync cost (`penalty_para = 1.0`); assumes a co-located learner. Would not hold for WAN-separated learner sites |
| A-vi | Staleness quality penalty is a diagnostic | `policy_age` reports the configured `k`; its sample-efficiency cost is not wired into `N_steps` (would create a fixed point) |
| A-vii | Heterogeneous hardware only via lumped `capacity_mult` | Straggler/length-variance tail penalty designed, not coded |
| A-viii | Facility power factor is a **flat constant** | `FACILITY_PUE = 1.6` multiplies board TDP in `node_gpus_power_cap`/`sites_power`, so every power figure is draw at the meter. One number stands in for host/network/conversion/cooling across every GPU family and site design |
| A-x | Inference sizing probes a `GPU_STEP` node first | `_min_gpus_at_tp`/`optimal_split` test `model_fits` at a 16-GPU node, so a model that fits only a FULL node (e.g. Kimi-K3 on a 72-GPU node) is reported `no-fit`. Latent for every published figure (Llama-405B/H100 fits at 16); see the review note |
| A-ix | `stock_gpus` counts **physical chips** | Not renormalised to H100-equivalents when `gpu` changes, so cross-family sweeps compare fixed chip counts, not fixed compute |

## Unit conventions

Times **seconds**, memory **bytes**, compute **FLOP**, bandwidth **bytes/s** (`NetSpec` converts
from Mbps). FLOP/s and bandwidth are **per node**, where a "node" is one inference replica or one
FSDP shard group (or an `inf_tp`-node serving group) — `gpus_per_node` converts to GPU counts. Quadratic terms use
`E[R²] = E[R]²(1+cv²)`.

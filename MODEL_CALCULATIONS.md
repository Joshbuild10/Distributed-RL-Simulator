# Distributed RL Roofline Model — Full Calculations

Replaces the report's *Appendix B: Full Model Calculations*. Every equation below is what the code
computes (`model.py` for one RL step, `feasibility_core.py` for the solver), with the code identifier
alongside so a reader can check one against the other. Equations are written out in plain text in the
same style as the working doc; `ceil`, `floor`, `max`, `min` are the usual functions.

Units throughout: time in **seconds**, memory and traffic in **bytes**, compute in **FLOP**, bandwidth in
**bytes/s**. Hardware rates (FLOP/s, HBM, HBM bandwidth) are **per node** unless subscripted `_gpu`.

## Notation

`T_x` is always a **time**. Sequence lengths are `P` (prompt) and `R` (response), never `T`.

| Symbol | Meaning | Code |
|---|---|---|
| `P_total`, `P_active` | Total params; params active per token (decoder layers only) | `p_total`, `p_active` |
| `P_head` | Unembedding params = `D * vocab` | `p_head` |
| `L`, `D` | Layers; hidden dimension | `n_layers`, `d_model` |
| `N_q`, `N_kv`, `H` | Query heads; KV heads; head dimension | `n_q_heads`, `n_kv_heads`, `head_dim` |
| `A` | Attention coefficient `N_q * H * L` | `attn_coef` |
| `P` | Mean prompt (prefill) length | `prompt_len` |
| `R` | Mean response length, `E[R]` | `response_len_mean` |
| `E[R^2]` | Second moment, `E[R]^2 * (1 + cv^2)` | `er2` |
| `G` | Responses per prompt (group size) | `responses_per_prompt` |
| `B` | Rollout batch = `prompts_per_batch * G` | `size_batch` |
| `ω` | Oversampling ratio (rollouts generated / rollouts trained on) | `oversample_ratio()` |
| `k` | Off-policy degree (staleness) | `sync_interval` |
| `b_x` | Bytes per element for `x` (weights, grads, optimiser, KV) | `b_*` |
| `n_train`, `n_inf` | Trainer / inference **node** counts | `train_hw.n_nodes`, `inf_hw.n_nodes` |
| `g_train`, `g_inf` | GPUs per trainer / inference node | `gpus_per_node` |
| `tp` | Inference nodes per tensor-parallel serving group | `inf_tp` |
| `F`, `M`, `W` | Node peak FLOP/s, HBM capacity, HBM bandwidth | `flops`, `node_hbm`, `bw` |
| `MFU_train`, `MFU_inf` | Achieved fraction of peak compute | `mfu` |
| `η_bw` | Achieved fraction of peak HBM bandwidth (0.85) | `bw_eff` |
| `C_RL` | Target RL compute budget for the whole run | `target_c_rl` |

## 1. Net

```
T_step  = max(T_rollout + T_verify, T_update, T_broadcast)        k >= 1  (off-policy, overlapped)
T_step  = T_rollout + T_verify + T_update + T_broadcast           k = 0   (on-policy, serial)
T_total = N_steps * T_step
```
- Bottleneck = the stage with the largest time. At `k = 0` it is still reported, but every stage adds.
- `N_steps` comes from the dataset when simulating a published run:
  `N_steps = ceil(N_prompts / prompts_per_batch) * N_epochs`. In the feasibility study it comes from
  the compute budget instead: `N_steps = C_RL / C_step` (§7).
- Policy age = `k` steps. Its sample-efficiency cost is **not** modelled, so a larger `k` only ever
  helps (gap A-vi).

## 2. Updates / training

**Compute.**
```
tok_batch      = B * (P + R)
C_layers       = 2 * P_active * tok_batch
C_dot-attention = c_causal * B * E[(P + R)^2] * A
    E[(P + R)^2] = P^2 + 2 * P * R + E[R^2]
C_update       = (3 + recomp_act + recomp_old) * (C_layers + C_dot-attention)
```
- `c_causal = 2` under causal masking (half the query–key pairs), 4 otherwise.
- The quadratic term uses the full second moment `E[R^2]`, not `R^2`, so length variance raises
  attention cost. Embedding/unembedding FLOPs are excluded.
- `recomp_act = 1`: full activation checkpointing (one extra forward). `recomp_old = 1`: old-policy
  log-probs recomputed on the trainer (INTELLECT-2 style). The feasibility study uses `recomp_act = 1`,
  `recomp_old = 0`, so an update costs **8** FLOP/param/token against rollout's 2 — a 4x ratio, not 3x.
- Attention passes 20% of layer FLOPs at `R ≈ 0.2 * P_active / A` (`attn_significance_T`).

**Time.**
```
T_comp   = C_update / (F_train * MFU_train * tput_mult_train * Penalty_para * n_train)
T_update = max(T_comp, T_sync)                          (T_sync = 0 for a co-located trainer, §5)
```
- `Penalty_para` (inter-node throughput loss) is held at 1: trainer nodes are co-located data-parallel
  replicas (gap A-ii, A-v).

**Memory per trainer node.** Each node holds a **full model replica**, sharded only across its own GPUs
(`N_shard = g_train`). The general form, for a ZeRO-3 group of `N_shard` GPUs:
```
Mem_weights   = b_weights_train * P_total * (g_train / N_shard)
Mem_gradients = b_grads         * P_total * (g_train / N_shard)
Mem_optimiser = b_optimiser     * P_total * (g_train / N_shard)
Mem_activation = 2 * c_act * seqs_per_micro * (P + R) * D * L        (site-parallel: one micro-batch per node)
               = that * g_train                                       (data-parallel: one per GPU)
    c_act = 1 with recomp_act = 1, else 20
Mem_train = Mem_weights + Mem_gradients + Mem_optimiser + Mem_activation
fits      = Mem_train <= M_train
```
- With `N_shard = g_train` the state term is the whole replica, independent of node size — which is
  why a 16-GPU node cannot train Llama-3.1-405B (3.3 TB needed, 1.28 TB available) and a 128-GPU node
  can (10.2 TB).
- `fits` is **enforced**: a configuration that fails it is inadmissible (it would need cross-node model
  parallelism, which the model excludes). Not counted, and would only tighten it: the largest gathered
  FSDP unit, collective/workspace buffers, a master-weight copy.
- `grad_accum = B / (n_train * seqs_resident_per_node) / opt_steps` is reported, not used.

## 3. Rollout / inference

**Serving group.** `tp` whole nodes pool HBM, bandwidth and FLOP/s around **one** copy of the weights.
`tp = 1` is a single node — the decentralised case, since tensor parallelism over a WAN is not viable.
```
Mem_weights-inf = b_weights_inf * P_total                      (all experts resident)
Mem_free        = tp * M_inf - Mem_weights-inf
model_fits      = Mem_free > 0
W_grp           = tp * W_inf * η_bw
F_grp           = tp * F_inf * MFU_inf * tput_mult_inf
```

**KV cache.**
```
KV_per-tok    = b_kv * 2 * L * N_kv * H
Mem_KV_peak   = KV_per-tok * R_max                             (R_max = max_response_len, else P + R)
Vol_KV_total  = KV_per-tok * (P * R + E[R^2] / 2)              (KV bytes read over one sequence's decode)
seq-parallel  = floor(Mem_free / Mem_KV_peak)                  (sequences one group holds at once)
```
- Provisioning is for a max-length sequence (`kv_provisioning = "peak"`), the conservative bound on
  concurrency. `"expected"` provisions `KV_per-tok * (P + R/2)` instead.
- When `max_response_len` is set the peak excludes the prompt — a ~`P / R_max` (2.5% at the defaults)
  underestimate of the peak.

**Waves.** A wave is one continuously-batched pass of up to `seq-parallel` sequences per group. The
pool generates `ω * B` sequences per step — the oversampled ones alongside the batch, not after it.
```
B_gen          = ω * B
n_groups       = floor(n_inf / tp)
active_groups  = min(n_groups, ceil(B_gen))
n_waves        = ceil(B_gen / (seq-parallel * active_groups))
seqs_per_wave  = B_gen / (active_groups * n_waves)             (mean fill, not seq-parallel)
```
- Spare groups run a wave at lower occupancy rather than idle; the floor is one sequence per group,
  because splitting a sequence across groups is `tp`'s job. So once `n_groups >= B_gen`, extra
  inference hardware does nothing.
- Charging the mean fill rather than a full `seq-parallel` per wave avoids double-counting the last,
  mostly-empty wave (deviation 1 in `SIMULATOR_MAP.md`).

**Compute per sequence.**
```
C_prefill = 2 * P_active * P + c_causal * P^2 * A              (divided by G under prefix caching)
C_decode  = 2 * P_active * R + 4 * A * (P * R + E[R^2] / 2)
```
- Decode attention sums `4 * A * s` over positions `s = P .. P + R`; the causal halving is already in
  that sum, so the coefficient is 4, not `c_causal`.

**Time per wave.**
```
Mem_weights-decode = b_weights_inf * min(P_active + P_head, P_total)
T_decode_mem  = Mem_weights-decode * R / W_grp                 (stream the weights once per token)
T_decode_comp = C_decode * seqs_per_wave / F_grp
T_decode      = max(T_decode_comp, T_decode_mem)
kv_shard      = min(tp, N_kv * max(1, seqs_per_wave))
T_inf-attention = Vol_KV_total * seqs_per_wave / (W_inf * η_bw * kv_shard)
T_prefill     = C_prefill * seqs_per_wave / F_grp
T_env         = environment latency per wave
```
- `T_decode_mem` is **independent of batch occupancy**: a group pays the full weight-streaming cost
  per token whether it serves one sequence or a hundred. It falls only with `tp` (more pooled
  bandwidth) — this is the term that sets the decentralised ceiling.
- Every time above is charged at the **mean** length `R`. That holds when continuous batching keeps
  slots full across step boundaries; in a single-wave step the last sequence to finish is nearer
  `R_max`, so the floor is optimistic by up to `R_max / R` (gap A-vii).
- MoE decode reads the active path plus the head, clamped at `P_total` so a dense model doesn't
  double-count its head; HBM capacity above still holds every expert, and KV traffic stays dense
  (deviation 2).
- KV shards over heads first (up to `N_kv`) then over the batch, so `tp` keeps helping past the head
  count. The AllReduce/AllToAll collectives this implies are **not** charged (gap A-iv).

**Rollout.**
```
T_rollout        = (T_decode + T_inf-attention + T_prefill + T_env) * n_waves
C_rollout_total  = (C_prefill + C_decode) * B * ω
```
- `ω` enters time through `B_gen`, so it costs time only once the extra sequences overflow the
  groups available: with spare groups it is absorbed in parallel; with no spare capacity it adds waves
  in proportion, as a serial multiplier would. Its compute is always paid in full.
- `ω` is an input (`oversample_override`, 1.5 in the study). If set to `None` it is derived from a
  task success rate `p` under zero-advantage filtering: `ω = 1 / (1 - p^G - (1 - p)^G)`. That
  single-`p` form understates `ω` for bimodal datasets; the measured `1 / E_p[f(p)]` is 1.5–4.7x.

## 4. Verifier

```
Rule-based:  T_verify = B * t_verify * ω / n_verifier_workers
Model-based: C_verify = 2 * P_verifier * tok_batch * ω
             T_verify = C_verify / (F_verifier * MFU_verifier)      (prefill only; short verdicts)
```

## 5. Communication

```
Vol_broadcast = b_weights_inf * P_total / compression
T_broadcast   = (Vol_broadcast / bandwidth_net + latency) / max(k, 1)
    bandwidth_net = link_Mbps * 1e6 / 8
```
- `k = 0` and `k = 1` both broadcast every step; `k >= 2` broadcasts once every `k` steps, so its
  per-step cost is `1/k` of a full sync.

**Trainer all-reduce** (`train_distributed`, the default). Trainer nodes sit at separate sites, so every
step the replicas average their gradients over the same link as the broadcast:
```
T_sync = 2 * b_grads * P_total / grad_compression / bandwidth_net + latency    (n_train > 1; else 0)
```
- `grad_compression = 150` (Rahman's default: 4-bit + top-k sparsification with error feedback;
  Covenant-72B ran 146x). The broadcast keeps its own `compression` (16x).
- Sent and received (the factor 2), and paid **every step**: `H = 1`, and unlike the broadcast it is
  not amortised by `k`. It binds only if `grad_compression < 2 * compression / k`: at 150x vs 16x it
  is ~0.21x the broadcast (87 s vs 406 s for Llama-405B at 1 Gbps), so the broadcast stays the wall.
  At 16x it would be 810 s, halving the ceiling.
- Overlapped with compute, hence `T_update = max(T_comp, T_sync)`. More trainer nodes cannot shorten
  it, so it is a per-step floor for any run needing more than one trainer node.
- `train_distributed = False` is the co-located trainer (datacentre fabric, sync free); the
  centralised comparator's 1 Tbps link makes it ~1 s either way.
- One-directional, single link, no topology (relay trees, fan-out) and no rollout upload.
- `compression` covers quantisation and sparsification together (16x in the study = int4 x 4x).

## 6. Diagnostics (reported, not used by the solver)

```
MFU_model  = (3 / (3 + recomp_act + recomp_old)) * C_update / (T_update * F_train * n_train)
MFU_hw     = C_update / (T_update * F_train * n_train)
tok/s_train = tok_batch / T_update
tok/s_inf   = B * R * ω / T_rollout                  (peak: while generating)
tok/s_inf,achieved = B * R * ω / T_step              (whole step: matches papers' total-gen / wall-clock)
flop_ratio    = C_rollout_total / C_update
gputime_ratio = (T_rollout * GPUs_inf) / (T_update * GPUs_train)
```
`MFU_model` / `MFU_hw` only see the trainer's own `T_update`, so they echo the `MFU_train` input. The
**achieved** whole-cluster MFU (§8) divides all FLOPs done by `T_step` times every allocated GPU's
peak, so a pool idling on the other stage drags it down.

## 7. Building a cluster from GPU counts (`FeasConfig.build`)

The solver works in **GPUs** and converts to nodes:

- **Trainer.** Nodes of up to `g_train,max = 128` GPUs, each a full replica (`N_shard = g_train`). If
  one node suffices it is right-sized **down** to the requested count; otherwise
  `n_train = ceil(GPUs_train / 128)` full nodes.
- **Inference.** At least `GPU_STEP = 16` GPUs. With `tp = 1`, nodes of up to `g_inf,max`
  (16 decentralised, 72 centralised), one node right-sized down when it suffices. With `tp > 1`, whole
  serving groups of `tp` full nodes.
- Fixed per-step settings: `R_max = 2 * R`, prefix caching on, `recomp_act = 1`, `η_bw = 0.85`,
  `latency = 0.5 s`, rule verifier at 0.5 s/rollout over 10,000 workers, site-parallel activations.

```
C_step  = C_update + C_rollout_total                  (node-count independent)
N_steps = C_RL / C_step
T_run(GPUs_train, GPUs_inf, tp) = N_steps * T_step,   or ∞ if !model_fits or !fits
```

## 8. Solver (`feasibility_core.py`)

One routine sizes everything, so the two solvers built on it cannot disagree.

**`min_gpus_for_tsb(T_sb)`** — smallest cluster whose step fits a per-step time budget `T_sb`:
```
k >= 1:  T_broadcast <= T_sb,     budget_rollout = budget_update = T_sb
k = 0:   rem = T_sb - T_broadcast,  (budget_rollout, budget_update) = (a * rem, (1 - a) * rem)
         for a in {0.2, 0.3, ..., 0.8}

GPUs_inf   = smallest multiple of 16 with  T_rollout + T_verify <= budget_rollout      (bisection)
GPUs_train = max( ceil16(Mem_train / M_gpu),
                  ceil16(C_update / (F_gpu * MFU_train * budget_update)) )
```
- Tried for every `tp` in `{1, 2, 4, ..., inf_tp_max}` and every split; the cheapest wins. So `tp`
  rises only when a larger group buys hardware back.
- With `tp = 1` the inference floor starts at 16 GPUs and walks up to the smallest node that holds
  the weights; with `tp > 1` the floor is one full group.
  The bisection assumes rollout time is non-increasing in inference GPUs.
- Rejected unless the **built** cluster (after rounding to whole nodes/groups) has finite time,
  passes both memory checks, and fits the stock.
- At `k = 0` the split grid cannot land on an exact stage boundary, so `optimal_split` doubles its
  starting budget (up to 8 times) until the routine can reproduce it, then bisects back down.

```
min_cluster(window)   = min_gpus_for_tsb(window / N_steps)            min GPUs to finish in the window
optimal_split(budget) = min_gpus_for_tsb(T_sb*), with T_sb* the smallest budget that still fits
                        `budget` GPUs (bisection to 1e-4 relative)    fastest run on a fixed fleet
```
- `evaluate(window)` returns the minimum cluster if feasible, else the **full-stock optimal split**: the
  fastest achievable wall-clock and the GPUs that reach it. `no-fit` means the weights exceed one
  serving group or the replica exceeds one trainer node.
- In by-stock sweeps a node is capped at half the stock (`min(g, floor(stock / 2))`), so a small
  fleet still forms one trainer and one inference node.

**Maximum achievable compute** on a fixed stock and window:
```
C_RL,max(window) = C_step * window / T_step*                          (T_step* from optimal_split(stock))
MFU_achieved     = C_step / (T_step* * Σ_allocated F)
```

**Required link** — the slowest `link * compression` that keeps the broadcast hidden:
```
solve the operating point with the link effectively free (1e9 Mbps), then
budget   = k * max(T_update, T_rollout + T_verify) - latency
Mbps_1x  = 8 * b_weights_inf * P_total / budget / 1e6
```
Solving at a free link stops the answer depending on the link it is measuring; quoted at 1x, so a real
compression factor divides it.

**Power and export caps.**
```
GPUs_per_site = site_MW * 1e6 / (W_board * PUE)            PUE = 1.6 (facility draw at the meter)
TPP_chip      = F_gpu,bf16 * 16                            (= FLOP-equivalents at a uniform BF16 rating)
g_cap         = min( floor(16 * TPP_H100 / TPP_chip), floor(16 * 80 GB / M_gpu) )
```

## 9. Distributed-training penalties (optional, `PenaltySpec.enabled`)

Ported from Rahman's simulator (`penalties.py`, functions identical to `evasion_calculator.py` @ 7b94d81).
Off by default; with it off every number above is unchanged. `f_tail(n)` is his slowest-of-`n` tail:
```
f_tail(n) = 1 + k_mode * 0.05 * log2(n)          k_mode: 1 synchronous/none, 0.4 relay (default),
                                                         0.3 backup_workers, 0 threshold/async
```

**Trainer.** Replicas are groups of `S` pipeline stages (`S = ceil(Mem_train / M_train)`, 1 when a
replica fits a node); `n_grp = floor(n_train / S)`.
```
T_rep  = C_update / n_grp / (F_train * MFU_train)                  one replica's update on one stage slot
T_comp = T_rep                                                     S = 1
       = (M + S - 1) * (T_rep / (S * M) + (T_handoff + lat_pp) * f_tail(S))     S > 1 (GPipe, M = 8)
    T_handoff = 2 * (tok_batch / n_grp) * D * b_weights_train / C_act / M / bw_pp
T_sync = (§5 trainer all-reduce) * f_tail(n_grp)                                n_grp > 1
H      = 1 (synchronous DDP; see below). Unset: ceil(T_sync / T_comp), Rahman's DiLoCo, for parity only
T_update = max(T_comp, T_sync / H)                                 streaming; else T_comp + T_sync / H
```
- The sync uses the trainer link (§5): the WAN for a distributed trainer, else the co-located fabric
  (`g_train * 50 GB/s`, 400 Gb/s IB per GPU, 10 µs). Pipeline handoffs default to the fabric
  (`pp_bw`). WAN/regional values with `h_inner = None` reproduce his flat and PP-group DiLoCo.
- `mem.fits` still gates the solver, so pipelined trainers (`S > 1`) only arise in scenarios built
  directly (e.g. testing 16-GPU trainer nodes).
- `H = 1` because local steps have no clean RL meaning: between syncs each replica holds a different
  policy, so inference must either sample from one replica (off-policy for the rest), wait for the
  average (staleness `>= H`), or run as `n_grp` separate RL runs merged every `H` steps. His `η_H` is
  calibrated on pre-training DiLoCo with a fixed data stream; in RL the data depends on the policy, so
  replica drift also shifts what is sampled. At `H = 1`, `η_H = 1` and the replica loss is exactly 1.

**Inference.** No gradients move, so no quality terms apply; only throughput:
```
T_rollout   *= f_tail(active_groups)       each wave ends when its slowest serving group does
T_broadcast *= f_tail(n_inf)               the fan-out ends when the slowest receiver does
```
Tails use `inf_mode`, **async by default**: streaming generation never waits on a slow node, so both
tails are 1. `relay` gives `1 + 0.02 * log2(n)` (x1.25 at ~6k nodes) as a sensitivity. Not modelled:
pipelining one model across inference nodes (still no-fit) and tensor-parallel collectives (gap A-iv).

**Goodput** stretches the run, `T_total = N_steps * T_step / u`, with `u = u_train * u_inf`:
`u_train` is his `reliability()` over the trainer GPUs (checkpoint/restart semantics by mode), and
`u_inf = exp(-λ * t_rec)` — a failed inference node loses only its in-flight sequences, nothing stalls.

**Quality** discounts what the compute is worth and never changes time:
```
η = η_H * η_comp * η_act * η_mit * η_replica
    η_H    = max(0.4, 1 - α(P) * log10(H)),  α(P) = 0.08 / (1 + log10(P / 1e9) / 5)    n_grp > 1
    η_comp = his pseudo-gradient table at the gradient compression (EF, "expected")  n_grp > 1
    η_act  = q(C_act) ^ (2 * (S - 1))                                                  S > 1
    η_mit  = 0.85 threshold, 0.97 async, else 1                                        n_grp > 1
```
His replica penalty is a loss multiplier `M^β(P,H)` that he turns into FLOP through the Chinchilla
pre-training law. Here the multiplier is reported (`pen.replica_loss`) and the Chinchilla conversion
(`penalties.replica_flop_penalty`) is for separate reporting only — it asks what from-scratch
pre-training compute buys back the loss, which has no RL analogue. Both are 1 at `H = 1`.

**In the solver.** `T_run = N_steps * T_step / u`, and `min_cluster` shrinks the per-step budget by `u`.
`η` is carried on every result (`evaluate`, the max-compute point) but **never sizes anything**:
quality-adjusted compute, `C_RL * η`, is a separate presentation. Because the broadcast tail grows with
the inference node count the solver picks, the solver now also checks that the composed step fits its
budget. Toggle: `FeasConfig.penalty`, `feasibility.py --penalties`, `PENALTIES=on` for the dashboard.

### What this means at the threat-model operating point

Llama-3.1-405B, H100, 16-GPU inference nodes, distributed 128-GPU trainer nodes, 1 Gbps, 16x,
`R = 20k`, `B = 65,536`, `ω = 1.5`, 1.5M-GPU stock:

| Term | Value |
|---|---|
| `T_broadcast` | **406 s — the binding stage** |
| `T_sync` | 87 s: 2 × 810 GB / 150 over 1 Gbps, every step (hidden) |
| `T_update` | 405 s on 31,232 trainer GPUs (244 nodes) |
| `B_gen` | 98,304 sequences over 6,443 inference nodes: 1 wave at ~15 per node (`seq-parallel` = 22) |
| `T_decode_mem` | 354 s — streaming 806 GB of decode weights (`P_active + P_head`) 20,000 times through one node |
| `T_rollout` = 354 + 47 (decode is memory-bound, so `T_decode_comp` hides) | 401 s |
| GPUs allocated | 134,320 of 1,500,000: more hardware cannot shorten a step below the broadcast |
| `C_RL,max` | 2.6e26 (180 d), 1.0e27 (720 d) |

So at the stock frontier the decentralised ceiling is the weight broadcast — `link * compression`
(1 Gbps x 16) and `k` — with weight streaming (354 s) just below it. The trainer's all-reduce only
takes over if gradients are compressed less than ~32x (810 s at 16x, halving the ceiling).

### Headline numbers at these inputs

Reproduced from the code on 2026-11-10 (`python results/plot_dashboard.py`, `python feasibility.py`).
Per-target rows use the budget-scaled batch (256 → 65,536); the by-stock rows use the flat 65,536.

Minimum GPUs to finish (decentralised), as `total GPUs (trainer GPUs)`; `X` = misses the window:

| Target `C_RL` | 30 d | 90 d | 180 d | 720 d |
|---|---|---|---|---|
| DeepSeek-R1-Zero 2e22 | 96 (48) | 64 (48) | 64 (48) | 64 (48) |
| o1 1e23 | 272 (80) | 112 (48) | 80 (48) | 64 (48) |
| o3 1e24 | 3,008 (768) | 896 (256) | 432 (128) | 128 (48) |
| Grok-3 2.5e25 | X (floor 69 d) | 23,904 (6,016) | 12,016 (3,072) | 2,560 (768) |
| Grok-4 2.5e26 | X (floor 173 d) | X (floor 173 d) | 105,152 (29,952) | 25,440 (7,552) |

Maximum `C_RL` in 180 d on a fixed stock, decentralised vs the centralised comparator:

| Stock | Decentralised | GPUs it can use | Bound by | Centralised | Ratio |
|---|---|---|---|---|---|
| 128 | 2.8e23 | 128 | rollout | 5.4e23 | 1.9x |
| 1,024 | 2.2e24 | 1,024 | rollout | 4.3e24 | 1.9x |
| 10,000 | 2.4e25 | 9,968 | update | 4.5e25 | 1.9x |
| 100,000 | 2.4e26 | 99,920 | update | 4.5e26 | 1.9x |
| 300,000 | 2.6e26 | 134,320 | **broadcast** | 1.3e27 | 5.0x |
| 1,500,000 | 2.6e26 | 134,320 | **broadcast** | 3.6e27 | 13.9x |

- The decentralised ceiling is a **rate** — 2.6e26 per 180 days (1.45e24 FLOP/day) — reached at
  134,320 GPUs, of which 31,232 (244 nodes of 128, ~35 MW at the meter) are trainers.
- Doubling inference HBM bandwidth cuts the Grok-4 / 180 d cluster from 105,152 to 67,568 GPUs.
- Link needed to keep the broadcast hidden for Grok-3 in 180 d (Mbps at 1x compression): Llama-3.1-8B
  1,334; 70B 1,423; 405B 1,537; Qwen3-235B-A22B 9,646; Llama-4 402B-A17B 29,499.

## Inputs used in the feasibility study (`FeasConfig` defaults)

| Input | Value |
|---|---|
| Model | Llama-3.1-405B: `P_total` 405B, `P_active` 401B, `L` 126, `D` 16,384, `N_q` 128, `N_kv` 8, `H` 128 |
| Batch | 128 prompts x 512 responses = 65,536 (scaled per budget in the dashboard's per-target charts: 256 → 65,536 in x4 rungs) |
| `P`, `R`, `cv` | 1,000; 20,000; 0.5 |
| `ω`, `k` | 1.5; 1 |
| Precision | BF16 weights/grads/KV; optimiser 4 B/param (Muon) |
| `MFU_train`, `MFU_inf` | 0.40, 0.30 |
| GPU | H100: 989 TFLOP/s BF16, 80 GB, 3.35 TB/s, 700 W |
| Nodes | 128-GPU trainer at separate sites, 16-GPU inference, `tp = 1` (centralised comparator: 72-GPU, `tp <= 4`, 1 Tbps) |
| Network | 1 Gbps, 0.5 s latency for both the broadcast (16x) and the trainer all-reduce (150x gradients) |
| Stock | 1.5M GPUs; 5 MW sites |

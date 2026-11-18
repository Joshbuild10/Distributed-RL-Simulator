"""
feasibility -- Config and solver for the distributed-RL feasibility study. No printing lives here: the analysis/ package
turns these into every figure and table.

The question: for an RL run of a given COMPUTE SCALE (anchored to published
models), what is the minimum cluster -- node count + trainer:inference split -- that finishes
it inside a target wall-clock window, on a given GPU family and internet speed, subject to a
finite compute stock and a per-site power (detectability) cap, and what does it cost. Every input
is a knob: see FeasConfig.

RESULT MODEL: every table goes through evaluate(cfg, window), which ALWAYS returns one operating
point, so a scenario that misses the window is never just "infeasible":
    feasible   -> the minimum cluster that hits the window
    infeasible -> the full stock at optimal split = the fastest achievable wall-clock ("days to
                  expend the FLOPs") and the trainer:rollout ratio at that floor
    no-fit     -> the weights exceed one serving group, or the replica exceeds one trainer node;
                  either needs parallelism this model does not represent, so it is unrepresentable
                  rather than slow.

Modelling assumptions and their known biases are catalogued ONCE in SIMULATOR_MAP.md ("Deliberate
deviations" and "Known gaps" A-i..A-x) -- read that before quoting an absolute number. The two
that shape this file's solver: each trainer node holds a FULL model replica (intra-node sharding
only; multiple nodes are data-parallel replicas), and inference weights must fit ONE serving group
of inf_tp nodes. Both are enforced, not assumed -- total_time() rejects a config that violates
either. Time windows are ours; anchor them against Epoch's distributed-training feasibility model
when writing.
"""
from .hardware import (GB, TB, GPUS, HIGH_END_GPUS, FACILITY_PUE, GPU_STEP, node_hw, tpp_per_chip,
                       node_gpus_tpp_cap, node_gpus_power_cap, effective_node_gpus, sites_power)
from .targets import (MODEL_TARGETS, POLICY_THRESHOLDS, model_targets, TARGET_BATCH_BASE, target_batch,
                      ISO_N_BY_RUNG, target_batch_iso, HIGH_BATCH, BASELINE_BATCH, BATCH_SWEEP, batch_split,
                      HBM_MULTS)
from .models import MODELS, DEFAULT_MODEL, smallest_fitting_gpu
from .config import FeasConfig, with_cfg, config_lines
from .solver import (per_step_flop, n_steps, gpus_of, total_time, min_gpus_for_tsb, optimal_split, min_cluster,
                     min_wall_full_stock, evaluate, required_bcast_mbps, max_compute_operating_point,
                     steps_and_c_rl)

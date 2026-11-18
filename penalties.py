"""
penalties.py -- Distributed-training efficiency penalties, ported from Rahman's simulator
(github.com/robirahman/miri-decentralized-training-report, evasion_calculator.py @ 7b94d81;
arXiv 2605.29359, Simulator_Documentation.md sections 4-5).

Pure functions with his constants and functional forms unchanged, so results can be checked
against his code line for line. How they attach to an RL step is model.py's job (see
trainer_penalties / the PenaltySpec toggle); nothing here knows about RL.

Two kinds of penalty:
  throughput -- straggler_factor (sync tail), reliability (goodput), GPipe bubble + activation
                transfer (model.trainer_penalties). These lengthen wall-clock.
  quality    -- eta_h, compression_quality, activation_compression_quality, replica loss.
                These discount the compute a run is worth; they never change time.
"""
import math

# --- Straggler / reliability defaults (evasion_calculator.py L125-144) ---------------------------
FAILURE_RATE_PER_GPU_HOUR = 2.0e-5          # 1 failure / 50,000 GPU-hours (Llama 3; Epoch AI)
RECOVERY_TIME_S = 600.0                     # detect + restart + reload a failed node
CHECKPOINT_MODE = "async"
LOCAL_CHECKPOINT_BW_BPS = 3e9 * 8           # ~3 GB/s NVMe, bits/s
SLOW_NODE_FRACTION = 0.10                   # fail-slow nodes (ByteDance trace)
SLOW_NODE_SEVERITY = 0.60                   # ...running at this fraction of normal speed
BACKUP_FRACTION = 0.05
THRESHOLD_QUALITY_PENALTY = 0.85            # dropped-token staleness
ASYNC_STALENESS_PENALTY = 0.97
ASYNC_GOODPUT_FLOOR = 0.88                  # Decoupled DiLoCo under high failure
CKPT_REPLICAS = 4
_TAIL_COEF_BASE = 0.05 / (0.10 * (1 / 0.60 - 1))   # ~0.75: defaults reproduce 1 + 0.05*log2(n)
_RELAY_TAIL_FACTOR = 0.40
_BACKUP_TAIL_FACTOR = 0.30
MODES = ("none", "synchronous", "checkpoint_elastic", "relay", "threshold", "async", "backup_workers")

# --- Quality tables (L277-296, L560-568) ----------------------------------------------------------
COMPRESSION_QUALITY_NO_EF = {
    1:   {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    4:   {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    16:  {"optimistic": 1.00, "expected": 0.99, "conservative": 0.95},
    100: {"optimistic": 1.00, "expected": 0.98, "conservative": 0.90},
    150: {"optimistic": 0.99, "expected": 0.96, "conservative": 0.85},
    500: {"optimistic": 0.99, "expected": 0.95, "conservative": 0.75},
}
COMPRESSION_QUALITY_EF = {
    1:   {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    4:   {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    16:  {"optimistic": 1.00, "expected": 1.00, "conservative": 0.97},
    100: {"optimistic": 1.00, "expected": 0.99, "conservative": 0.93},
    150: {"optimistic": 1.00, "expected": 0.99, "conservative": 0.91},
    500: {"optimistic": 0.99, "expected": 0.96, "conservative": 0.80},
}
ACTIVATION_COMPRESSION_QUALITY = {
    1:  {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    2:  {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    4:  {"optimistic": 1.00, "expected": 0.995, "conservative": 0.96},
    10: {"optimistic": 0.995, "expected": 0.98, "conservative": 0.90},
}
REPLICA_H_REF = 30                          # Charles et al. (2025) calibration H

# Chinchilla (Besiroglu et al. 2024 refit), used only to turn a replica LOSS multiplier into a FLOP
# penalty. That conversion is a pre-training law; see replica_flop_penalty.
CHINCHILLA_TOKENS_PER_PARAM = 25.6
_CHIN_E, _CHIN_A, _CHIN_ALPHA, _CHIN_B, _CHIN_BETA = 1.8172, 482.01, 0.3478, 2085.43, 0.3658


def straggler_factor(n, mode="relay", slow_fraction=SLOW_NODE_FRACTION,
                     slow_severity=SLOW_NODE_SEVERITY):
    """Slowest-of-n sync-tail multiplier (expected max of n fail-slow workers ~ log2 n)."""
    if n <= 1:
        return 1.0
    base = _TAIL_COEF_BASE * slow_fraction * (1.0 / max(1e-6, slow_severity) - 1.0) * math.log2(n)
    if mode in ("threshold", "async"):
        return 1.0                                     # non-blocking: nobody waits on the tail
    if mode == "relay":
        return 1.0 + _RELAY_TAIL_FACTOR * base
    if mode == "backup_workers":
        return 1.0 + _BACKUP_TAIL_FACTOR * base
    return 1.0 + base                                  # none / synchronous / checkpoint_elastic


def reliability(gpus, time_s, state_bytes, mode="relay", failure_rate=FAILURE_RATE_PER_GPU_HOUR,
                recovery_s=RECOVERY_TIME_S, checkpoint_mode=CHECKPOINT_MODE,
                gpu_mem_feasible=False):
    """(goodput u, eta_mit) for a cluster of `gpus` running `time_s`. Rahman's reliability_model
    minus the sync tail (straggler_factor) and cost multiplier, which callers apply themselves.
    gpu_mem_feasible: free HBM exists for peer-GPU checkpoints (his gpu_memory checkpoint mode)."""
    rate = max(1, gpus) * max(0.0, failure_rate)       # cluster failures per hour
    t_h, rec_h = time_s / 3600.0, recovery_s / 3600.0
    p_down = 1.0 - math.exp(-failure_rate * rec_h)
    c_raw_h = state_bytes * 8.0 / LOCAL_CHECKPOINT_BW_BPS / 3600.0
    if checkpoint_mode == "async":
        c_h = c_raw_h * 0.02
    elif checkpoint_mode == "gpu_memory" and gpu_mem_feasible:
        c_h = c_raw_h * 0.005
    else:
        c_h = c_raw_h

    def young_daly(full_recovery):
        if rate <= 0:
            return 0.0
        t_star = math.sqrt(2.0 * c_h / rate) if c_h > 0 else float("inf")
        if t_star == float("inf"):
            return rate * rec_h
        return c_h / t_star + rate * (t_star / 2.0 + (rec_h if full_recovery else 0.0))

    if mode == "none":
        u = 1.0 / (1.0 + rate * t_h / 2.0)             # no checkpoints: restart from scratch
    elif mode == "synchronous":
        u = max(0.0, 1.0 - young_daly(True))
    elif mode == "checkpoint_elastic":
        u = max(0.0, 1.0 - young_daly(False) - p_down)
    elif mode == "backup_workers":
        u = 1.0 - max(0.0, p_down - BACKUP_FRACTION)
    elif mode == "async":
        u = 1.0 - p_down - (1.0 - ASYNC_GOODPUT_FLOOR) * (1.0 - math.exp(-rate * rec_h))
    else:                                              # relay / threshold
        u = 1.0 - p_down
    eta_mit = {"threshold": THRESHOLD_QUALITY_PENALTY, "async": ASYNC_STALENESS_PENALTY}.get(mode, 1.0)
    return u, eta_mit


def alpha(params):
    """Sync-interval sensitivity, shrinking with model size. params as a raw count."""
    return 0.08 / (1.0 + math.log10(params / 1e9) / 5.0)


def eta_h(h, params):
    """Quality left after H local steps between syncs, floored at 0.4."""
    return max(0.4, 1.0 - alpha(params) * math.log10(max(1.0, h)))


def _interpolate_quality(table, ratio, scenario):
    """Log-linear interpolation in a quality table keyed by compression ratio."""
    if ratio <= 1:
        return 1.0
    if ratio in table:
        return table[ratio][scenario]
    keys = sorted(table)
    for lo, hi in zip(keys, keys[1:]):
        if lo <= ratio <= hi:
            lo_log = 0 if lo <= 1 else math.log10(lo)
            t = (math.log10(ratio) - lo_log) / (math.log10(hi) - lo_log)
            return table[lo][scenario] + t * (table[hi][scenario] - table[lo][scenario])
    return table[keys[-1]][scenario]


def compression_quality(ratio, scenario="expected", error_feedback=True):
    """Quality factor for pseudo-gradient compression at `ratio`."""
    return _interpolate_quality(COMPRESSION_QUALITY_EF if error_feedback else COMPRESSION_QUALITY_NO_EF,
                                ratio, scenario)


def activation_compression_quality(ratio, stages, scenario="expected"):
    """Per-boundary activation-compression quality compounded over 2(S-1) crossings (fwd + bwd)."""
    if ratio <= 1 or stages <= 1:
        return 1.0
    return _interpolate_quality(ACTIVATION_COMPRESSION_QUALITY, ratio, scenario) ** (2 * (stages - 1))


def replica_loss_multiplier(n_replicas, params, h):
    """Loss multiplier from averaging n replicas every h steps: M^beta, beta = 1.0923 N^-0.2342
    * ln h / ln 30 (fit to Charles et al. 2025, N <= 10B). h = 1 is plain data parallel: 1.0."""
    if n_replicas <= 1 or h <= 1:
        return 1.0
    beta = 1.0923 * params ** (-0.2342) * math.log(h) / math.log(REPLICA_H_REF)
    return n_replicas ** beta


def chinchilla_loss(params, tokens):
    if params <= 0 or tokens <= 0:
        return float("inf")
    return _CHIN_E + _CHIN_A * params ** (-_CHIN_ALPHA) + _CHIN_B * tokens ** (-_CHIN_BETA)


def _chinchilla_efficiency(params, tokens, c_flop, loss_multiplier):
    """Fraction of c_flop a Chinchilla-optimal run would need to reach the same (inflated) loss."""
    target = chinchilla_loss(params, tokens) * loss_multiplier
    lo, hi = 1e10, c_flop * 10
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        n = math.sqrt(mid / (6 * CHINCHILLA_TOKENS_PER_PARAM))
        if chinchilla_loss(n, CHINCHILLA_TOKENS_PER_PARAM * n) > target:
            lo = mid
        else:
            hi = mid
    return min(1.0, math.sqrt(lo * hi) / c_flop)


def replica_flop_penalty(loss_multiplier, params, tokens):
    """Rahman's eta_replica = chi(with loss mult) / chi(without), via the Chinchilla law at
    (params, tokens). A PRE-TRAINING conversion: it asks what extra from-scratch compute buys back
    the lost loss. There is no RL analogue, so it is for separate reporting only, never folded into eta."""
    if loss_multiplier <= 1.0:
        return 1.0
    c = 6 * params * tokens
    chi = _chinchilla_efficiency(params, tokens, c, 1.0)
    return _chinchilla_efficiency(params, tokens, c, loss_multiplier) / chi if chi > 0 else 0.0

"""
targets.py -- What a run is asked to do: RL compute budgets anchored to published models, governance
FLOP thresholds, and the batch heuristics and sweep axes the figures use.
"""
import math

# ---------------------------------------------------------------------------
# Target compute-scale presets  (total RL FLOP; see data/reasoning_models.csv for provenance)
# ---------------------------------------------------------------------------
# Names carry NO "(est)" suffix: every figure here except DeepSeek-R1-Zero's is an estimate, the
# per-entry provenance note below is the record of how each was derived, and the writeup
# disclaims the estimation method in prose. Repeating "(est)" in the key just made every chart
# category and table row 6 characters longer for a governance audience that is being told the
# provenance anyway.
MODEL_TARGETS = {          # (C_RL, provenance-note)
    "DeepSeek-R1-Zero":     (2e22, "FLOP calculations"),
    "o1":       (1.0e23, "ESTIMATE: rough bracket, absolute n/d"),
    "o3":       (1.0e24, "ESTIMATE: 10x o1 disclosed ratio, absolute n/d"),
    "Grok-3-Reasoning":   (2.5e25, "SPECULATIVE ESTIMATE, inferred from published chart"),
    "Grok-4":   (2.5e26, "SPECULATIVE ESTIMATE, inferred from published chart"),
}


# Named governance thresholds = total-training-run FLOP reporting/registration bars. Point for the
# writeup: the disclosed RL stages (MODEL_TARGETS, 2e22 up) sit below these, so an RL run reproducing a frontier
# reasoning model is invisible to FLOP-threshold governance -- the threshold binds on pre-training.
POLICY_THRESHOLDS = {
    "MIRI-monitored":   1e22,
    "EU GPAI":          1e23,   # EU AI Act GPAI baseline
    "MIRI-strict":      1e24,
    "EU systemic":      1e25,   # EU AI Act systemic-risk tier
    "SB53 / dual-use":  1e26,   # CA SB 53 covered-model / US dual-use foundation model
}


def model_targets(top=None):
    """Just the model-anchored targets as {label: C_RL} -- MODEL_TARGETS itself stores
    (C_RL, provenance) tuples, which every caller was unpacking by hand. `top=n` keeps only the
    n largest by C_RL (the latency-wall/batch tables only make their point at large targets)."""
    t = {k: v[0] for k, v in MODEL_TARGETS.items()}
    if top is None:
        return t
    return dict(sorted(t.items(), key=lambda kv: kv[1])[-top:])


# --- Experiment: per-target batch that scales with compute budget --------------------------------
# One FIXED batch across all targets leaves the smallest budgets finishing in ~3 steps, where serial
# step latency -- the whole point of the study -- cannot bind. Scaling the batch with the budget
# (TARGET_BATCH_BASE, x4 per rung: 256 -> 65,536 across the five targets) gives every target a
# sizeable step count, and is defensible anyway since critical batch size grows with scale.
# Callers opt in explicitly; model_targets() and the solver are unchanged.
TARGET_BATCH_BASE = 256


def target_batch(name):
    """Total rollout batch assigned to model target `name`, scaling x4 with each rung up the budget
    ladder (TARGET_BATCH_BASE for the smallest). None for an unknown name. Split into a square
    prompts x responses via batch_split() at the call site."""
    order = [n for n, _ in sorted(MODEL_TARGETS.items(), key=lambda kv: kv[1][0])]
    return TARGET_BATCH_BASE * 4 ** order.index(name) if name in order else None


# --- IsoCompute law (arXiv 2603.12151), an alternative to the x4 heuristic above ------------------
# That paper decomposes sampling compute as C = Bp * n * M and finds Bp mainly a stability knob
# (hold it fixed) while n is the compute-sensitive lever, rising along a log-log sigmoid that
# saturates (~512 easy set, ~128-256 hard). So: fix Bp, walk n up a saturating ladder.
# ISO_N_BY_RUNG is a STYLIZED read of their Figure 7 (powers of two, saturating at 512), not their
# fitted curve -- the exact n*(C) values were not extractable.
ISO_BP = 128                                 # fixed problems/batch: mid of the paper's {32..1024} sweep, "stable"


ISO_N_BY_RUNG = (8, 32, 128, 256, 512)       # rollouts/problem up the budget ladder; saturating, cap 512 (easy set)


def target_batch_iso(name):
    """(prompts_per_batch, responses_per_prompt) for model target `name` under the IsoCompute law:
    fixed Bp = ISO_BP, rollouts-per-problem n from the saturating ISO_N_BY_RUNG ladder. Total Bp*n
    stays <= 65,536 (the paper's hardware cap). None for an unknown name."""
    order = [n for n, _ in sorted(MODEL_TARGETS.items(), key=lambda kv: kv[1][0])]
    return (ISO_BP, ISO_N_BY_RUNG[min(order.index(name), len(ISO_N_BY_RUNG) - 1)]) if name in order else None


# Batch levers. 128 prompts x G=512 = 65,536 total rollout batch.
HIGH_BATCH = dict(prompts_per_batch=128, responses_per_prompt=512)


BASELINE_BATCH = dict(prompts_per_batch=32, responses_per_prompt=16)


# Total rollout batch (B questions x n samples/question), x4 per step so six points span three
# orders of magnitude. All exponents are even, so batch_split() gives clean square B x n splits.
BATCH_SWEEP = (256, 1024, 4096, 16_384, 65_536, 262_144)


def batch_split(total):
    """Split a power-of-two total rollout batch into (B questions, n samples per question) as
    evenly as powers of two allow: B = 2**ceil(e/2), n = 2**floor(e/2) for total = 2**e.

    Even-ish because at fixed total batch the two axes are near-substitutes for wall-clock here:
    both cut the serial step count, and n is only marginally cheaper via shared prefill. That is
    why a full B x n grid is redundant along its diagonals and the presentation sweeps the PRODUCT.
    They stop being substitutes in learning dynamics, which this runtime model does not cover."""
    e = round(math.log2(total))
    return 2 ** ((e + 1) // 2), 2 ** (e // 2)


# HBM-bandwidth multipliers for the dedicated memory-bandwidth sweep (CLI table + dashboard panel).
HBM_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)

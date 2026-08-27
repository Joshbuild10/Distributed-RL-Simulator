#!/usr/bin/env python3
"""
uncertainty.py -- Monte-Carlo predictive intervals on the validation anchors' step time.

WHY THIS EXISTS. The point predictions in validate.py / dashboard chart 1a answer "how far off
are we", which is the wrong question on its own: several anchors' inputs are genuinely uncertain
(nobody published their MFU, and E[R] is measured on a proxy checkpoint), so a residual of -40%
might be a modelling gap or might be entirely input uncertainty. Sampling the uncertain inputs
and reporting a predictive INTERVAL converts each anchor from "off by X%" into "the published
number does / does not fall inside what our input uncertainty can explain" -- which is the claim
the writeup actually needs to make about the FSDP and MoE/multi-turn residuals.

model.py stays deterministic on purpose; all the distribution machinery lives here.

WHAT IS VARIED (independently -- see NOTE ON INDEPENDENCE):
  train_hw.mfu  ~ per-config Triangular, mode = the preset's own trainer MFU (see mfu_train_tri).
                  Default; `mfu_prior="uniform"` swaps the flat U(MFU_TRAIN_RANGE) for comparison.
                  The preset MFU is itself the data-grounded value (Ultra-Scale Playbook node-
                  matched RL-like frontier x 0.85), so the deterministic point prediction and this
                  prior's centre are the SAME number. prime-rl is pinned (PINNED_MFU_TRAIN): its
                  MFU is the run's own published, throughput-corroborated value, not an estimate.
  inf_hw.mfu    ~ U(0.10, 0.40)     inference MFU, never published by any anchor
  rl.response_len_mean ~ Triangular(E[R]*(1-ER_PCT), E[R], E[R]*(1+ER_PCT)), clamped to the cap --
                  a simple "the measured E[R] could be +-ER_PCT off the run's true E[R]" band, since
                  E[R] is measured on a proxy model/checkpoint (and RL lengthens responses). prime-rl
                  is pinned (PINNED_ER): its E[R] was inverse-solved from its own throughput.

WHAT IS NOT VARIED: response_len_cv. The within-run length spread already enters simulate() EXACTLY
via er2 = Var + E[R]^2 = E[R]^2(1+cv^2), the full second moment (tail included -- not a 1-SD/66%
slice), so its effect on the mean step time needs no resampling. (An earlier build also sampled the
cv PARAMETER; dropped as unnecessary -- it moved short-E[R] anchors ~4% and changed no verdict.)

NOTE ON INDEPENDENCE: trainer and inference MFU are physically correlated (same engineering
team, same stack). Sampling them independently therefore widens the interval relative to truth --
the conservative direction for a claim of the form "even allowing for input uncertainty, this
residual survives". Stated rather than silently assumed.

NOTE ON THE INFERENCE PRIOR: U(0.10, 0.40) has mean 0.25 while the presets carry mfu_inf=0.25-0.40,
so for a ROLLOUT-BOUND anchor the MC mean can sit BELOW the deterministic point. Intended (the old
0.40 inference default was optimistic); do not read the MC-mean vs point gap as a bug.
"""
import hashlib
import math
import random
import statistics as st
from dataclasses import replace

from model import simulate

MFU_TRAIN_RANGE = (0.10, 0.50) # flat prior, for the mfu_prior="uniform" head-to-head comparison
MFU_INF_RANGE = (0.10, 0.40)

# E[R] uncertainty: the measured E[R] is a proxy (different model/checkpoint/dataset than the run's
# own, and RL lengthens responses over training), so the run's true E[R] could be some percent
# above or below it. Modelled as a symmetric Triangular band about the measured value,
# E[R] ~ Tri(er*(1-ER_PCT), er, er*(1+ER_PCT)). +-30% is a generous "we don't really
# know" band; set to 0.20 for a tighter band. prime-rl is exempt (PINNED_ER): its E[R] was inverse-solved from the
# run's own published throughput and corroborated (7,536 vs 7,470), so varying it would score the
# model against a number it was fitted to.
ER_PCT = 0.30
PINNED_ER = {
    "prime-rl reference run (R1-Distill-Qwen-32B, 24x H200)":
        "E[R] inverse-solved from this run's own throughput (corroborated 7,536 vs 7,470) -- pinned",
}

# Trainer MFU is a data-grounded ESTIMATE (0.85 x nanotron frontier) for every anchor except
# prime-rl, where 38.46% is the anchor's own PUBLISHED, throughput-corroborated MFU -- not an
# inferred haircut. Applying the same generic +/- spread to a known number overstates uncertainty
# on the best-anchored run, so it is pinned exactly like its E[R].
PINNED_MFU_TRAIN = {
    "prime-rl reference run (R1-Distill-Qwen-32B, 24x H200)":
        "trainer MFU is this run's own published, throughput-corroborated value (38.46%) -- pinned",
}


def er_band(sc):
    """(pct_or_None, basis). None => do not vary E[R] (pinned). Else E[R] ~ Tri(+-pct*E[R])."""
    if sc.name in PINNED_ER:
        return None, PINNED_ER[sc.name]
    return ER_PCT, f"measured E[R]={sc.rl.response_len_mean:,.0f}; +-{ER_PCT:.0%} band (proxy model/checkpoint)"


def mfu_train_tri(sc):
    """Per-config trainer-MFU triangular (lo, mode, hi). The MODE is the preset's own trainer MFU
    (validation/presets.py) -- the single source of truth, so the deterministic point prediction
    and this prior's centre are the SAME number. Those preset values are the data-derived expected
    MFU: mode = 0.85 x the Ultra-Scale Playbook (huggingface/nanotron) node-matched RL-realistic
    (pp=1, tp<=4, ZeRO DP-sharded / FSDP-like) best MFU at each anchor's (model size, trainer node
    count). 0.85 is the clean-well-run-RL haircut, corroborated independently by prime-rl
    (38.46/45) AND AReaL-7B (implied 34.1 / node-matched 40.0). Big anchors (32B/106B) read the
    70-80B frontier (a 32B is 60% log-size toward 70B and FSDP param-gather scales with size);
    INTELLECT-3 also takes a x0.6 MoE+forced-TP penalty.

      hi = 0.30 (MoE) else 0.50  -- FP8 / better-kernel upside over nanotron's bf16 pretraining.
      lo = max(0.10, 0.6*mode) = max(0.10, 0.5*best) -- "underperforming but not broken" floor
           (RL frameworks are less tuned than nanotron; the competent-config spread bottoms near
           half-of-best). Since mode = 0.85*best, 0.6*mode ~= 0.5*best.

    CRUCIAL: anchored on nanotron (INDEPENDENT data), NOT on the step-time-implied MFU from
    solvers.solve_for_mfu -- centering there would be circular and would let the prior launder the
    FSDP residual. So it sits ABOVE the implied MFU for the high-shard anchors, and that gap is the
    residual the MC flags. Triangular (not a bell) because the nanotron top-configs are exactly
    that shape: a hard ceiling with a tail down.

    EXCEPTION: PINNED_MFU_TRAIN anchors (prime-rl) skip all of the above -- their mode IS the
    published MFU, not an estimate, so the triple degenerates to (mode, mode, mode) and
    random.triangular returns that constant every draw (low==high short-circuits before the
    low/high spread is computed)."""
    mode = sc.train_hw.mfu
    if sc.name in PINNED_MFU_TRAIN:
        return (mode, mode, mode)
    hi = 0.30 if sc.model.is_moe else 0.50
    lo = max(0.10, round(0.5 * mode, 2))
    return (lo, mode, hi)

# Draws per anchor. Chosen from measured convergence, not habit: at n=2,000 the p95 estimate
# wobbled 5.2% across seeds, which is too loose to publish an interval from; n=8,000 brings
# that to 0.6% (p5 to 2%) for 190ms/anchor, ~1.5s for all eight. simulate() is 19us, so the
# precision is nearly free -- there is no reason to sub-sample and then caveat the tails.
MC_DRAWS = 8_000

# See its use in mc_step_time: absorbs point-match rounding noise on a fully-pinned, degenerate
# (zero-width) interval without being wide enough to touch any real residual in this table.
INSIDE_TOL = 0.01

def _seed_for(name, seed):
    """Stable per-anchor seed. Python salts str hashing per process, so hashing the name directly
    would make the dashboard non-deterministic across runs -- and the HTML is regenerated and
    diffed constantly, so churn there is a real cost."""
    d = hashlib.md5(f"{name}|{seed}".encode()).hexdigest()[:8]
    return int(d, 16)


def mc_step_time(sc, n=MC_DRAWS, seed=0, mfu_prior="tri"):
    """Predictive distribution of simulate(sc).t_step under the input priors above.

    Returns dict(mean, p5, p50, p95, published, inside, sigma_log, basis, n, varied) where
    `inside` is whether the published step time falls in [p5, p95] -- the headline number: it
    says whether the residual is explainable by input uncertainty alone. `published`/`inside` are
    None when the preset carries no published t_step.

    n=MC_DRAWS is set from measured convergence, not habit -- see that constant.
    """
    rng = random.Random(_seed_for(sc.name, seed))
    pct, basis = er_band(sc)
    er0 = sc.rl.response_len_mean
    cap = sc.rl.max_response_len
    lo_m, mode_m, hi_m = mfu_train_tri(sc)
    draws, bottlenecks = [], []
    for _ in range(n):
        er = er0
        if pct:
            # symmetric +-pct band about the measured E[R]; clamp to the generation cap.
            er = rng.triangular(er0 * (1 - pct), er0 * (1 + pct), er0)
            if cap:
                er = min(er, float(cap))
        # trainer MFU: per-config triangular (mode = preset MFU) if mfu_prior="tri", else the flat
        # MFU_TRAIN_RANGE (for the head-to-head). random.triangular takes (low, high, mode).
        mfu_tr = rng.triangular(lo_m, hi_m, mode_m) if mfu_prior == "tri" else rng.uniform(*MFU_TRAIN_RANGE)
        cand = replace(
            sc,
            train_hw=replace(sc.train_hw, mfu=mfu_tr),
            inf_hw=replace(sc.inf_hw, mfu=rng.uniform(*MFU_INF_RANGE)),
            rl=replace(sc.rl, response_len_mean=er),
        )
        r = simulate(cand)
        if math.isfinite(r.t_step):
            draws.append(r.t_step)
            bottlenecks.append(r.bottleneck)
    if not draws:
        return dict(mean=None, p5=None, p50=None, p95=None, published=None, inside=None,
                    er_pct=pct, basis=basis, n=0, varied=[], bottleneck=None, spread=None)
    draws.sort()
    q = lambda p: draws[min(len(draws) - 1, max(0, int(round(p * (len(draws) - 1)))))]
    pub = (sc.published or {}).get("t_step")
    p5, p95 = q(0.05), q(0.95)
    varied = (["mfu_train"] if sc.name not in PINNED_MFU_TRAIN else []) + ["mfu_inf"] + (["er"] if pct else [])
    # Modal bottleneck across draws. Reported because a near-ZERO-WIDTH interval is a result, not
    # a glitch: a broadcast-bound anchor's step time depends on weight volume and link speed, and
    # neither MFU nor E[R] touches those -- so no amount of the uncertainty we are modelling can
    # move its prediction, and its residual cannot be explained away by these priors.
    modal = max(set(bottlenecks), key=bottlenecks.count)
    # INSIDE_TOL: a 1% relative slack on containment. Needed now that prime-rl pins BOTH mfu_train
    # and E[R] and is update-bound (mfu_inf doesn't touch update time) -- its interval is a true
    # single point, so a strict p5<=pub<=p95 fails on sub-percent model/publication rounding alone
    # (e.g. 1,377 vs 1,374) and would misreport an exact match as "no real residual". 1% is far
    # below every genuine residual in this table (smallest is AReaL-14B at ~8%), so it cannot
    # launder a real gap -- it only absorbs point-match noise on fully-pinned anchors.
    inside = None if not pub else bool(p5 * (1 - INSIDE_TOL) <= pub <= p95 * (1 + INSIDE_TOL))
    return dict(mean=st.fmean(draws), p5=p5, p50=q(0.50), p95=p95, published=pub,
                inside=inside,
                er_pct=pct, basis=basis, n=len(draws), varied=varied,
                bottleneck=modal, spread=(p95 / p5 if p5 > 0 else float("inf")))


def mc_table(scenarios, n=MC_DRAWS, seed=0, mfu_prior="tri"):
    """[(label, mc_dict)] for a list of (label, scenario) pairs."""
    return [(lab, mc_step_time(sc, n=n, seed=seed, mfu_prior=mfu_prior)) for lab, sc in scenarios]


def print_mc_report(scenarios, n=MC_DRAWS, seed=0, width=126, mfu_prior="tri"):
    """Text counterpart of dashboard chart 1a's error bars."""
    prior_desc = ("mfu_train ~ per-config Triangular (mode = preset MFU, nanotron-grounded)"
                  if mfu_prior == "tri" else f"mfu_train ~ U{MFU_TRAIN_RANGE}")
    print(f"\n{'=' * width}\n  MONTE-CARLO PREDICTIVE INTERVAL on step time "
          f"(n={n:,}/anchor, seed={seed})\n{'=' * width}")
    print(f"  priors: {prior_desc}; mfu_inf ~ U{MFU_INF_RANGE} (independent); "
          f"E[R] ~ Tri(+-{ER_PCT:.0%}) about the measured value (prime-rl pinned)")
    print(f"  {'anchor':<20}{'published':>10}{'point':>9}{'MC mean':>9}{'p5':>9}{'p95':>9}"
          f"{'p95/p5':>8}{'E[R] band':>10}{'bottleneck':>16}{'pub in 90% band?':>20}")
    rows = mc_table(scenarios, n=n, seed=seed, mfu_prior=mfu_prior)
    for (lab, mc), (_, sc) in zip(rows, scenarios):
        point = simulate(sc).t_step
        erb = "pinned" if mc["er_pct"] is None else f"+-{mc['er_pct']:.0%}"
        verdict = "-" if mc["inside"] is None else ("YES explainable" if mc["inside"]
                                                   else "NO real residual")
        pub = mc["published"]
        print(f"  {lab:<20}{(f'{pub:,.0f}' if pub else '--'):>10}{point:>9,.0f}"
              f"{mc['mean']:>9,.0f}{mc['p5']:>9,.0f}{mc['p95']:>9,.0f}{mc['spread']:>8.2f}"
              f"{erb:>10}{mc['bottleneck']:>16}{verdict:>20}")
    print("  'pub in 90% band' = does input uncertainty ALONE explain the gap? NO means a structural")
    print("  residual (FSDP shard penalty, MoE/multi-turn re-prefill) survives the uncertainty.")
    print("  p95/p5 ~ 1.0 is a RESULT, not a bug: a broadcast-bound anchor's step time is set by")
    print("  weight volume / link speed, which neither MFU nor E[R] touches -- so these priors")
    print("  cannot move it, and its residual cannot be attributed to them.")
    print("  'point' now uses each preset's data-grounded per-config trainer MFU (not a flat 0.40),")
    print("  so it and the MC mode agree; MC mean can still sit below it via the independent mfu_inf.")
    if mfu_prior == "tri":
        print("\n  trainer-MFU prior (lo, mode=preset, hi) -- mode = 0.85 x Ultra-Scale-Playbook")
        print("  node-matched RL-like best MFU (corroborated by prime-rl AND AReaL-7B); anchored on")
        print("  nanotron NOT the step-time-implied MFU (circular), so it sits ABOVE implied for the")
        print("  high-shard anchors and the gap is the FSDP residual the 'NO' verdicts flag:")
        for lab, sc in scenarios:
            lo, mode, hi = mfu_train_tri(sc)
            tag = "  <- pinned (published MFU)" if sc.name in PINNED_MFU_TRAIN else ""
            print(f"    {lab:<20} ({lo:.2f}, {mode:.2f}, {hi:.2f}){tag}")
    return rows

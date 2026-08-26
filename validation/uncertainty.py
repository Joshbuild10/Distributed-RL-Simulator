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
  train_hw.mfu  ~ U(0.30, 0.50)     trainer MFU, never published by any anchor
  inf_hw.mfu    ~ U(0.10, 0.40)     inference MFU, likewise
  rl.response_len_mean ~ LogNormal(log(preset E[R]), sigma_log), clamped to max_response_len

WHAT IS NOT VARIED, AND WHY IT WOULD BE DOUBLE-COUNTING:
`rl.response_len_cv` (0.3-1.1 across these presets) is the spread of INDIVIDUAL responses within
a run, and model.py already consumes it -- it feeds E[R^2] in the quadratic attention/KV terms.
The quantity that belongs in an error bar is the uncertainty in the MEAN, which is a different
and much smaller number. Sampling the within-run cv again would inflate the interval by an effect
the deterministic model has already accounted for.

E[R] PRIOR WIDTH: three provenance tiers, because these anchors are not equally well pinned.
  "pinned"   -- E[R] was inverse-solved FROM the published step time we are scoring against.
                Varying it manufactures error that the run does not have. prime-rl only; and its
                7,470 is independently corroborated at 7,536 by primerl_skywork_lengths.json
                (n=256), so treating it as known is a measurement claim, not just a dodge.
  "measured" -- E[R] measured on (approximately) this run's own model and task. sigma_log =
                MEASURED_FLOOR, which is set by checkpoint drift; measurement sampling error is
                an order of magnitude smaller and so never binds (see that constant).
  "proxy"    -- E[R] carried over from a different model on the same task. sigma_log = the
                CROSS-MODEL spread of measured mean E[R] in that domain, computed live from
                experiments/lengths/ by domain_sigma_log(). That population spread is exactly the
                "we don't know which model's length profile applies" uncertainty.

NOTE ON INDEPENDENCE: trainer and inference MFU are physically correlated (same engineering
team, same stack). Sampling them independently therefore widens the interval relative to truth --
the conservative direction for a claim of the form "even allowing for input uncertainty, this
residual survives". Stated rather than silently assumed.

NOTE ON THE INFERENCE PRIOR: U(0.10, 0.40) has mean 0.25 while every preset carries mfu=0.40, so
for a ROLLOUT-BOUND anchor the MC mean sits BELOW the deterministic point prediction by design.
That is intended (the 0.40 inference default was optimistic) but it means MC mean and point
prediction are answering different questions -- do not read a gap between them as a bug.
"""
import glob
import hashlib
import math
import os
import random
import statistics as st
from dataclasses import dataclass, replace

from model import simulate

# sigma_log for a "measured" anchor -- set by CHECKPOINT DRIFT, not measurement noise.
#
# Sampling error of the measured mean (cv/sqrt(n)) was computed for every measured anchor and is
# 0.018-0.051 (QwQ-32B math n=450 cv=1.09 -> 0.051; glm45air code n=256 cv=0.47 -> 0.029;
# R1-Distill-7B math n=512 cv=0.96 -> 0.042; areal14b n=512 cv=0.45 -> 0.020; areal32b n=256
# cv=0.28 -> 0.018). Using those would claim we know each run's E[R] to a few percent, which is
# false for a reason sampling error cannot see: RL lengthens responses over training and we
# measure a STATIC checkpoint. So the floor below dominates in every case and the SE term is not
# computed -- carrying it would imply a precision the provenance does not support.
#
# Empirical anchor for the floor: the two independent measurements of R1-Distill-7B on math in
# this campaign (8,714 and 10,613 tokens) differ by x1.22 despite being the same model and task,
# implying sigma_log >~ ln(1.22)/2 ~ 0.10 per measurement. Rounded up to 0.15 for drift on top.
# This is the one hand-set number in this module.
MEASURED_FLOOR = 0.15

LENGTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", "lengths")

# Series whose measured mean is unusable as a population data point. qwen3-32b hit the 65,536-token
# generation cap on 38% (math) / 42% (code) of samples through a repetition/no-stop pathology, so
# its mean is a censored lower bound that happens to read ~2x every other model. Including it
# inflated the math domain's cross-model sigma_log from 0.155 to 0.325 -- i.e. a known-bad series
# would have doubled the error bars on every proxy-tier anchor.
LENGTH_EXCLUDE = ("qwen3-32b",)
MAX_TRUNCATED_FRAC = 0.10        # drop any series censored beyond this


@dataclass(frozen=True)
class ErPrior:
    """How well this anchor's E[R] is known. sigma_log=None means 'do not vary'."""
    tier: str
    sigma_log: float | None
    basis: str


# Keyed by Scenario.name. Every entry states its provenance so the choice is auditable rather
# than buried; `basis` is printed by the CLI report and carried into the dashboard caption.
ER_PRIORS = {
    "prime-rl reference run (R1-Distill-Qwen-32B, 24x H200)": ErPrior(
        "pinned", None,
        "E[R] 7,470 inverse-solved from this run's own published throughput; independently "
        "corroborated at 7,536 (primerl_skywork, n=256). Varying it would score the model "
        "against a number it was fitted to."),
    "INTELLECT-2 (TARGET-SHORT)": ErPrior(
        "measured", None, "QwQ-32B math measured in-campaign (n=450)."),
    "INTELLECT-2 (TARGET-LONG)": ErPrior(
        "measured", None, "QwQ-32B math measured in-campaign; long-target config of the same run."),
    "INTELLECT-3 (prime-rl, 512x H200)": ErPrior(
        "measured", None,
        "GLM-4.5-Air code measured in-campaign (n=256, 22,961 tok). CAVEAT: measurement is "
        "single-turn, the run is multi-turn agentic, so this understates the real spread."),
    "AReaL R1-Distill-Qwen-1.5B (math, 16 H800 nodes)": ErPrior(
        "proxy", None, "no 1.5B measurement in-campaign; math-domain cross-model spread used."),
    "AReaL R1-Distill-Qwen-7B (math, 24 H800 nodes)": ErPrior(
        "measured", None, "R1-Distill-7B math measured live in-campaign (n=512 and n=256)."),
    "AReaL R1-Distill-Qwen-14B (code, 32 H800 nodes)": ErPrior(
        "measured", None, "areal14b code measured in-campaign (n=512, 10,376 tok)."),
    "AReaL R1-Distill-Qwen-32B (code, 48 H800 nodes)": ErPrior(
        "measured", None, "areal32b code measured in-campaign (n=256, 8,137 tok)."),
}

# Which length-campaign domain each anchor's task sits in, for the proxy tier's population spread.
ANCHOR_DOMAIN = {
    "INTELLECT-2 (TARGET-SHORT)": "math",
    "INTELLECT-2 (TARGET-LONG)": "math",
    "INTELLECT-3 (prime-rl, 512x H200)": "code",
    "AReaL R1-Distill-Qwen-1.5B (math, 16 H800 nodes)": "math",
    "AReaL R1-Distill-Qwen-7B (math, 24 H800 nodes)": "math",
    "AReaL R1-Distill-Qwen-14B (code, 32 H800 nodes)": "code",
    "AReaL R1-Distill-Qwen-32B (code, 48 H800 nodes)": "code",
}

MFU_TRAIN_RANGE = (0.30, 0.50)
MFU_INF_RANGE = (0.10, 0.40)

# Draws per anchor. Chosen from measured convergence, not habit: at n=2,000 the p95 estimate
# wobbled 5.2% across seeds, which is too loose to publish an interval from; n=8,000 brings
# that to 0.6% (p5 to 2%) for 190ms/anchor, ~1.5s for all eight. simulate() is 19us, so the
# precision is nearly free -- there is no reason to sub-sample and then caveat the tails.
MC_DRAWS = 8_000

_domain_cache: dict[str, float] = {}


def domain_sigma_log(domain):
    """Cross-model spread (SD of log mean E[R]) of every usable series measured in `domain`.

    This is the proxy tier's prior width: if we don't know which model's length profile applies to
    an anchor, the population of measured models IS the uncertainty. Computed live from
    experiments/lengths/ so it tracks the campaign rather than freezing a number in a comment.
    Returns None when fewer than 3 usable series exist (logic/science currently have 2 -- too few
    for a spread estimate, and quietly returning a 2-point stdev would look authoritative)."""
    if domain in _domain_cache:
        return _domain_cache[domain]
    try:
        import sys
        exp_dir = os.path.dirname(LENGTH_DIR)          # .../experiments
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)
        import lengths_io
    except ImportError:
        return None
    means = []
    for path in sorted(glob.glob(os.path.join(LENGTH_DIR, f"*_{domain}_lengths.json"))):
        stem = os.path.basename(path).replace(f"_{domain}_lengths.json", "")
        if any(x in stem for x in LENGTH_EXCLUDE):
            continue
        trunc = (lengths_io.meta(path) or (0, None, None))[0] or 0.0
        if trunc > MAX_TRUNCATED_FRAC:
            continue
        for s in lengths_io.load_stats(path).values():
            if s.get("n", 0) >= 8 and s.get("mean"):
                means.append(s["mean"])
    out = st.stdev([math.log(m) for m in means]) if len(means) >= 3 else None
    _domain_cache[domain] = out
    return out


def er_sigma_log(sc):
    """(sigma_log_or_None, basis_text) for this scenario's E[R] prior.

    RAISES on an unregistered scenario name rather than falling back. An earlier version returned
    a silent None here, which is the worst possible failure: a one-character typo in an ER_PRIORS
    key (`16 H800 nodes` vs the preset's `24`) made AReaL-7B report as "pinned -- explainable",
    i.e. it silently disabled that anchor's E[R] uncertainty AND relabelled its provenance, in a
    table whose entire purpose is auditing provenance. Loud is correct: adding or renaming a
    preset should force a decision about how well its E[R] is known."""
    prior = ER_PRIORS.get(sc.name)
    if prior is None:
        raise KeyError(
            f"no E[R] prior registered for scenario {sc.name!r}. Add an ER_PRIORS entry (and an "
            f"ANCHOR_DOMAIN entry if tier='proxy') in validation/uncertainty.py -- choose "
            f"'pinned' (E[R] inverse-solved from the number being validated), 'measured' (E[R] "
            f"measured on this run's own model+task), or 'proxy' (carried from another model). "
            f"Known: {sorted(ER_PRIORS)}")
    if prior.tier == "pinned":
        return None, prior.basis
    if prior.sigma_log is not None:
        return prior.sigma_log, prior.basis
    if prior.tier == "measured":
        return MEASURED_FLOOR, prior.basis
    # proxy: the cross-model population spread for this anchor's domain.
    dom = ANCHOR_DOMAIN.get(sc.name)
    s = domain_sigma_log(dom) if dom else None
    return (s if s is not None else MEASURED_FLOOR), prior.basis


def _seed_for(name, seed):
    """Stable per-anchor seed. Python salts str hashing per process, so hashing the name directly
    would make the dashboard non-deterministic across runs -- and the HTML is regenerated and
    diffed constantly, so churn there is a real cost."""
    d = hashlib.md5(f"{name}|{seed}".encode()).hexdigest()[:8]
    return int(d, 16)


def mc_step_time(sc, n=MC_DRAWS, seed=0):
    """Predictive distribution of simulate(sc).t_step under the input priors above.

    Returns dict(mean, p5, p50, p95, published, inside, sigma_log, basis, n, varied) where
    `inside` is whether the published step time falls in [p5, p95] -- the headline number: it
    says whether the residual is explainable by input uncertainty alone. `published`/`inside` are
    None when the preset carries no published t_step.

    n=MC_DRAWS is set from measured convergence, not habit -- see that constant.
    """
    rng = random.Random(_seed_for(sc.name, seed))
    sigma, basis = er_sigma_log(sc)
    er0 = sc.rl.response_len_mean
    cap = sc.rl.max_response_len
    draws, bottlenecks = [], []
    for _ in range(n):
        er = er0
        if sigma:
            er = er0 * math.exp(rng.gauss(0.0, sigma))
            if cap:
                # The mean cannot exceed the generation cap. With these sigmas this clips well
                # under 1% of draws, so a clamp is cheaper than rejection sampling and does not
                # meaningfully distort the interval.
                er = min(er, float(cap))
        cand = replace(
            sc,
            train_hw=replace(sc.train_hw, mfu=rng.uniform(*MFU_TRAIN_RANGE)),
            inf_hw=replace(sc.inf_hw, mfu=rng.uniform(*MFU_INF_RANGE)),
            rl=replace(sc.rl, response_len_mean=er),
        )
        r = simulate(cand)
        if math.isfinite(r.t_step):
            draws.append(r.t_step)
            bottlenecks.append(r.bottleneck)
    if not draws:
        return dict(mean=None, p5=None, p50=None, p95=None, published=None, inside=None,
                    sigma_log=sigma, basis=basis, n=0, varied=[], bottleneck=None, spread=None)
    draws.sort()
    q = lambda p: draws[min(len(draws) - 1, max(0, int(round(p * (len(draws) - 1)))))]
    pub = (sc.published or {}).get("t_step")
    p5, p95 = q(0.05), q(0.95)
    varied = ["mfu_train", "mfu_inf"] + (["er"] if sigma else [])
    # Modal bottleneck across draws. Reported because a near-ZERO-WIDTH interval is a result, not
    # a glitch: a broadcast-bound anchor's step time depends on weight volume and link speed, and
    # neither MFU nor E[R] touches those -- so no amount of the uncertainty we are modelling can
    # move its prediction, and its residual cannot be explained away by these priors.
    modal = max(set(bottlenecks), key=bottlenecks.count)
    return dict(mean=st.fmean(draws), p5=p5, p50=q(0.50), p95=p95, published=pub,
                inside=(None if not pub else bool(p5 <= pub <= p95)),
                sigma_log=sigma, basis=basis, n=len(draws), varied=varied,
                bottleneck=modal, spread=(p95 / p5 if p5 > 0 else float("inf")))


def mc_table(scenarios, n=MC_DRAWS, seed=0):
    """[(label, mc_dict)] for a list of (label, scenario) pairs."""
    return [(lab, mc_step_time(sc, n=n, seed=seed)) for lab, sc in scenarios]


def print_mc_report(scenarios, n=MC_DRAWS, seed=0, width=126):
    """Text counterpart of dashboard chart 1a's error bars."""
    print(f"\n{'=' * width}\n  MONTE-CARLO PREDICTIVE INTERVAL on step time "
          f"(n={n:,}/anchor, seed={seed})\n{'=' * width}")
    print(f"  priors: mfu_train ~ U{MFU_TRAIN_RANGE}, mfu_inf ~ U{MFU_INF_RANGE} (independent); "
          f"E[R] ~ LogNormal(log mu, sigma) by provenance tier")
    print(f"  {'anchor':<20}{'published':>10}{'point':>9}{'MC mean':>9}{'p5':>9}{'p95':>9}"
          f"{'p95/p5':>8}{'sigmaE[R]':>10}{'bottleneck':>16}{'pub in 90% band?':>20}")
    rows = mc_table(scenarios, n=n, seed=seed)
    for (lab, mc), (_, sc) in zip(rows, scenarios):
        point = simulate(sc).t_step
        sig = "pinned" if mc["sigma_log"] is None else f"{mc['sigma_log']:.3f}"
        verdict = "-" if mc["inside"] is None else ("YES explainable" if mc["inside"]
                                                   else "NO real residual")
        pub = mc["published"]
        print(f"  {lab:<20}{(f'{pub:,.0f}' if pub else '--'):>10}{point:>9,.0f}"
              f"{mc['mean']:>9,.0f}{mc['p5']:>9,.0f}{mc['p95']:>9,.0f}{mc['spread']:>8.2f}"
              f"{sig:>10}{mc['bottleneck']:>16}{verdict:>20}")
    print("  'pub in 90% band' = does input uncertainty ALONE explain the gap? NO means a structural")
    print("  residual (FSDP shard penalty, MoE/multi-turn re-prefill) survives the uncertainty.")
    print("  p95/p5 ~ 1.0 is a RESULT, not a bug: a broadcast-bound anchor's step time is set by")
    print("  weight volume / link speed, which neither MFU nor E[R] touches -- so these priors")
    print("  cannot move it, and its residual cannot be attributed to them.")
    print("  MC mean sits below the point prediction for inference-heavy anchors BY DESIGN: the")
    print("  inference prior is centred at 0.25 while the presets carry mfu_inf=0.40.")
    print(f"\n  E[R] prior provenance (sigma_log; 'measured' floors at {MEASURED_FLOOR} for checkpoint drift):")
    for lab, mc in rows:
        sig = "pinned" if mc["sigma_log"] is None else f"{mc['sigma_log']:.3f}"
        print(f"    {lab:<20} [{sig:>6}] {mc['basis']}")
    return rows

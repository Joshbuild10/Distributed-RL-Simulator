"""
reporting.py -- Formatting and printing of simulate() results.
"""

from typing import Any, Dict, List

from specs import Scenario
from model import SimResult, simulate, with_response_len


def fmt(x: float, unit: str = "") -> str:
    if x == float("inf"):
        return "inf"
    if unit == "s":
        if x < 90:
            return f"{x:,.1f} s"
        if x < 5400:
            return f"{x:,.0f} s ({x/60:.1f} min)"
        return f"{x/3600:,.2f} h"
    if unit == "B":
        for d, u in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
            if abs(x) >= d:
                return f"{x/d:,.1f} {u}"
        return f"{x:,.0f} B"
    if unit == "F":
        return f"{x:.3e} FLOP"
    return f"{x:,.4g}"


# Each `published` metric maps to how to pull the matching model value out of a
# SimResult and how to render it. Adding a new comparable published metric is
# then a new entry here, not a new branch in report()'s printing logic.
CALIBRATION_METRICS: Dict[str, Dict[str, Any]] = {
    "t_step": dict(label="step time", model=lambda r: r.t_step, fmt=lambda x: fmt(x, "s")),
    "t_broadcast": dict(label="broadcast", model=lambda r: r.t_bc, fmt=lambda x: fmt(x, "s")),
    "vol_broadcast": dict(label="bcast volume", model=lambda r: r.vol_bc, fmt=lambda x: fmt(x, "B")),
    # trainer: published figures are stage-local (batch tokens / trainer active time) -- prime-rl's
    # 11.3K matches tok_s_train directly.
    "tok_s_train": dict(label="trainer tok/s", model=lambda r: r.tok_s_train, fmt=lambda x: f"{x:,.0f}"),
    # inference: DELIBERATELY compared against ACHIEVED (gen / t_step), not the stage-local peak.
    # prime-rl's published 14.4K is total-generated / wall-clock: vs achieved it lands -3%, vs
    # stage-local peak +117%. Rollout-bound runs (INTELLECT-3) have peak == achieved so the choice
    # is only visible on update-bound runs. Flip to r.tok_s_inf only if a source is known to quote
    # the flat-out generation rate instead.
    "tok_s_inf": dict(label="infer tok/s", model=lambda r: r.tok_s_inf_achieved, fmt=lambda x: f"{x:,.0f}"),
    "mfu_train": dict(label="trainer MFU", model=lambda r: r.mfu_model, fmt=lambda x: f"{x*100:.1f}%"),
    "t_total": dict(label="total runtime", model=lambda r: r.t_total, fmt=lambda x: fmt(x, "s")),
    "inf_train_gputime_ratio": dict(label="inf:train GPU-time", model=lambda r: r.gputime_ratio,
                                     fmt=lambda x: f"{x:.2f}x", pub_fmt=lambda v: f"{v}x", pct=False),
}
# Unrecognized published keys (e.g. "node_split", a plain annotation with no
# model-computed counterpart) fall through to a raw "key: value" line below.


def _print_calibration_line(r: SimResult, k: str, v: Any) -> None:
    spec = CALIBRATION_METRICS.get(k)
    if spec is None:
        print(f"  {k}: {v}")
        return
    model_val = spec["model"](r)
    model_str = spec["fmt"](model_val)
    if spec.get("pct", True):
        err = (model_val - v) / v * 100
        pub_str = spec["fmt"](v)
        print(f"  {spec['label']:<14} model {model_str:>18} | published {pub_str:>18} | {err:+.0f}%")
    else:
        pub_str = spec.get("pub_fmt", spec["fmt"])(v)
        print(f"  {spec['label']}: model {model_str} | published {pub_str}")


def report(r: SimResult) -> None:
    s, tc, ro, mem = r.scenario, r.tc, r.ro, r.mem
    m, rl = s.model, s.rl
    W = 78
    print("=" * W)
    print(f"  {s.name}")
    print(f"  {m.name}   |   {rl.prompts_per_batch} prompts x {rl.responses_per_prompt} "
          f"= {rl.size_batch} rollouts   |   ctx ~{rl.context_len:,.0f} tok")
    print("=" * W)

    print("\n-- MODEL / ATTENTION SIGNIFICANCE " + "-" * 44)
    print(f"  P_total {m.p_total/1e9:.1f}B | P_active(layers) {m.p_active/1e9:.1f}B "
          f"| head {m.p_head/1e9:.2f}B | embed {m.p_embed/1e9:.2f}B")
    print(f"  attention hits 20% of layer FLOPs at T ~ {m.attn_significance_T(0.20):,.0f} tok;"
          f" parity at T ~ {m.attn_significance_T(1.0):,.0f}")
    print(f"  at this context, attention = {tc.attn_frac*100:.1f}% of training layer FLOPs")

    print("\n-- ROLLOUT (inference) " + "-" * 55)
    print(f"  KV/token {fmt(ro.kv_tok,'B')} | KV @peak ctx {fmt(ro.kv_peak,'B')}/seq "
          f"| @expected {fmt(ro.kv_expected,'B')}/seq | weights(inf) {fmt(ro.mem_weights_inf,'B')}")
    # capacity (all experts resident) vs decode bandwidth (active path + head) -- equal for dense,
    # ~p_total/p_active smaller for MoE, which is what sets t_decode's memory floor.
    print(f"  weights read per decode token {fmt(ro.mem_weights_decode,'B')} "
          f"({ro.mem_weights_decode/max(ro.mem_weights_inf,1)*100:.0f}% of resident weights)")
    if not ro.model_fits:
        print("  !! model weights exceed one node's HBM -- sharding penalty NOT modelled")
    if ro.model_fits and not ro.seq_fits:
        print("  !! a single max-length sequence does NOT fit -- generation may not complete")
    print(f"  provisioning = {s.algo.kv_provisioning} ; concurrency per node (seq_par) = {ro.seq_par} "
          f"; mean fill/wave = {ro.seqs_per_wave:,.0f} ; waves = {ro.n_waves}")
    print(f"  oversample_ratio = {ro.oversample_ratio:.2f}")
    print(f"  t_decode = {fmt(ro.t_decode,'s')}  [{ro.decode_bound}-bound: "
          f"comp {fmt(ro.t_dec_comp,'s')} vs mem {fmt(ro.t_dec_mem,'s')}]")
    print(f"  t_inf_attention (KV traffic) = {fmt(ro.t_inf_attn,'s')}")
    print(f"  t_prefill = {fmt(ro.t_prefill,'s')} | t_env = {fmt(ro.t_env,'s')}")
    print(f"  T_ROLLOUT = {fmt(ro.t_rollout,'s')}")

    print("\n-- UPDATE (training) " + "-" * 57)
    print(f"  tokens/batch {tc.tok_batch:.3e} | C_layers {fmt(tc.c_layers,'F')} "
          f"| C_attn {fmt(tc.c_attn,'F')}")
    print(f"  coefficient (3+recomp_act+recomp_old) = "
          f"{3+s.algo.recomp_act+s.algo.recomp_old}")
    print(f"  C_UPDATE = {fmt(tc.c_update,'F')}   ->   T_UPDATE = {fmt(r.t_update,'s')}")
    print(f"  per-node train memory: weights {fmt(mem.weights,'B')} | grads "
          f"{fmt(mem.grads,'B')} | optim {fmt(mem.optimiser,'B')} | act "
          f"{fmt(mem.activations,'B')} ({fmt(mem.act_per_gpu,'B')}/GPU)")
    print(f"  grad-accum micro-steps implied = {mem.grad_accum:.0f}")
    print(f"  total {fmt(mem.total,'B')} vs node HBM {fmt(s.train_hw.node_hbm,'B')}"
          f"  -> {'FITS' if mem.fits else 'DOES NOT FIT'}"
          f" (headroom {fmt(mem.headroom,'B')})")

    print("\n-- COMMUNICATION / VERIFY " + "-" * 52)
    print(f"  broadcast volume {fmt(r.vol_bc,'B')} -> T_BROADCAST {fmt(r.t_bc,'s')}")
    print(f"  T_VERIFY = {fmt(r.t_verify,'s')}")

    print("\n-- STEP / TOTALS " + "-" * 61)
    print(f"  mode: {r.mode}")
    for k, v in r.stages.items():
        bar = "#" * int(round(30 * r.efficiency[k]))
        star = "  <== BOTTLENECK" if k == r.bottleneck else ""
        print(f"    {k:<16} {fmt(v,'s'):>20}  {bar:<30}{star}")
    # Both stages reported STAGE-LOCAL first (rate while that stage is actually running), with the
    # whole-step "achieved" rate second. Papers differ on which they quote; see CALIBRATION_METRICS.
    print(f"  throughput: trainer {r.tok_s_train:,.0f} tok/s (stage-local) | inference "
          f"{r.tok_s_inf:,.0f} tok/s stage-local, {r.tok_s_inf_achieved:,.0f} achieved over the step "
          f"| implied MFU {r.mfu_hw*100:.1f}%")
    print(f"  T_STEP = {fmt(r.t_step,'s')}   staleness = {r.staleness} step(s)")
    print(f"  N_STEPS = {r.n_steps}   ->   T_TOTAL = {fmt(r.t_total,'s')}")
    print(f"  ratio inference:training  --  true FLOPs {r.flop_ratio:.2f}x  |"
          f"  GPU-time {r.gputime_ratio:.2f}x")

    if s.published:
        print("\n-- CALIBRATION vs PUBLISHED " + "-" * 50)
        for k, v in s.published.items():
            _print_calibration_line(r, k, v)

    if s.notes:
        print(f"\n  NOTE: {s.notes}")
    print()


def sweep_response_len(s: Scenario, lengths: List[float]) -> None:
    print(f"\n  Response-length sensitivity -- {s.name}")
    print(f"  {'E[R]':>9} {'T_step':>14} {'bottleneck':>16} {'seq_par':>8} "
          f"{'attn%':>7} {'FLOP i:t':>9}")
    for R in lengths:
        r = simulate(with_response_len(s, R))
        print(f"  {R:>9,.0f} {fmt(r.t_step,'s'):>14} {r.bottleneck:>16} "
              f"{r.ro.seq_par:>8} {r.tc.attn_frac*100:>6.1f}% "
              f"{r.flop_ratio:>8.2f}x")

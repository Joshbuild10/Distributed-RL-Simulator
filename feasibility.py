#!/usr/bin/env python3
"""
feasibility.py -- Sweeps, tables and CSV writers for the distributed-RL feasibility study,
plus the CLI. The config/solver primitives live in feasibility_core.py.

Run:
    python feasibility.py                          # headline tables (A1, policy, sensitivity)
    python feasibility.py --mode thorough          # everything, + CSVs
    python feasibility.py --mode models            # cross-model sweep + batch levers
    python feasibility.py --er 15000 --omega 2.5   # override any FeasConfig field
    python feasibility.py --model Llama3-405B --gpu B200 --batch 128x512
    python feasibility.py --targets DeepSeek-R1-Zero,o3 --windows 60,120
    python feasibility.py --mode thorough --fine-split   # trainer:inf split at 1% resolution

The trainer:inference split search defaults to a coarse 13-point grid (fast; every table uses
it). Pass --fine-split to search every integer percent instead (~13x more simulate() calls) when
the split fraction or GPU count itself is the number you're about to quote -- the coarse grid can
overstate the required cluster size by double digits of percent at some targets/windows.

Every table routes through feasibility_core.evaluate(), so a scenario that misses the window
still reports the fastest achievable wall-clock (days to expend the FLOPs) and the optimal
trainer:rollout split -- never a bare "infeasible". See feasibility_core for the result model
and the documented first-cut biases.
"""
import argparse
import math

from feasibility_core import (
    GPUS, MODELS, DEFAULT_MODEL, MODEL_TARGETS, POLICY_THRESHOLDS,
    HIGH_BATCH, BASELINE_BATCH, BATCH_CAP, BATCH_SWEEP, HBM_MULTS, HIGH_END_GPUS, FeasConfig,
    batch_split,
    all_targets, cell_text, config_lines, evaluate, fmt_num, fmt_ratio, fmt_time,
    min_wall_full_stock, model_targets, n_steps, optimal_split, per_step_flop, fmt_rate,
    required_bcast_mbps, sites_power, smallest_fitting_gpu, split_for, total_time, with_cfg, GB,
    tpp_per_chip, node_gpus_tpp_cap, node_gpus_power_cap,
)


# ---------------------------------------------------------------------------
# Generic renderers
# ---------------------------------------------------------------------------
def _rule(width=104):
    return "=" * width


def render_matrix(title, rows, cols, cellfn, row_label=str, col_label=str,
                  note=None, cw=23, rw=16, width=104):
    """Print a rows x cols matrix of strings. `cellfn(row, col) -> str`.

    Used by every genuinely matrix-shaped sweep (GPU family, total batch, HBM scaling,
    required bandwidth, batch-size x target). Tables with extra leading info columns or paired
    sub-columns keep their own printer -- folding those in would cost information, not save it.
    """
    print(f"\n{_rule(width)}\n  {title}\n{_rule(width)}")
    print(f"  {'':<{rw}}" + "".join(f"{col_label(c):>{cw}}" for c in cols))
    for r in rows:
        print(f"  {row_label(r):<{rw}}" + "".join(f"{cellfn(r, c):>{cw}}" for c in cols))
    if note:
        for line in note.split("\n"):
            print(f"  {line}")


def _cfg_hdr(base):
    return (f"GPU={base.gpu}, WAN={base.wan_mbps:.0f}Mbps, comp={base.compression:.0f}x, "
            f"E[R]={base.er:,.0f}, batch={base.prompts_per_batch}x{base.responses_per_prompt}, "
            f"omega={base.omega:g}, stock={fmt_num(base.stock_gpus)} GPUs, "
            f"split={'fine(1%)' if base.fine_split else 'coarse(13pt)'}")


def target_label(name, tc):
    """'<name> (<C_RL>)' -- for tables that reference a named RL target WITHOUT an adjacent
    explicit C_RL column (e.g. gpu_target_matrix's row labels), so the estimated compute a
    benchmark point like DeepSeek-R1-Zero corresponds to is visible right next to its name.
    Tables that already print C_RL as its own column (feasibility_table, policy_framing,
    batch_compare_table) don't need this -- it would just repeat the adjacent column."""
    return f"{name} ({tc:.1e})"


def target_label_width(targets):
    """Column width that fits the longest target_label() in `targets` -- computed from the
    actual data so a longer/renamed target never truncates or misaligns the table."""
    return max((len(target_label(n, tc)) for n, tc in targets.items()), default=14) + 2


def print_config_block(cfg, title="CONFIGURATION"):
    """Full config dump at the top of CLI output, so a copy-pasted terminal log is self-contained
    (which settings produced these numbers) without needing to cross-reference the invocation."""
    print(f"\n{_rule()}\n  {title}\n{_rule()}")
    for line in config_lines(cfg):
        print(f"  {line}")


def write_config_comment(fh, cfg):
    """Write the FeasConfig as '#'-prefixed comment lines at the top of a CSV, before the header
    row -- pandas/csv.reader skip these by default (comment='#'), so every CSV this module writes
    carries its own provenance without breaking the tabular structure for downstream tools."""
    for line in config_lines(cfg):
        fh.write(f"# {line}\n")


FLOOR_NOTE = ("* = infeasible in the window: GPUs = full stock, wall = FASTEST achievable (days to\n"
              "  expend the FLOPs), tr:inf = optimal trainer:rollout split at that floor.")
CELL_NOTE = "cell = GPUs/wall/tr:inf (feasible) or wall!/tr:inf (full-stock floor); no-fit = weights > one node"


# ---------------------------------------------------------------------------
# A1 / headline tables
# ---------------------------------------------------------------------------
def feasibility_table(base, targets, windows_d=(30, 90, 180, 720), title="A1 FEASIBILITY"):
    print(f"\n{_rule(150)}\n  {title}: min cluster to finish in the window ({_cfg_hdr(base)})\n{_rule(150)}")
    print(f"  {'target':<16}{'C_RL':>11}{'win':>6}{'feas':>6}{'GPUs':>11}{'%stock':>8}"
          f"{'tr:inf':>8}{'steps':>9}{'wall':>9}{'cost':>10}{'bottleneck':>15}"
          f"{'T_step':>9}{'T_update':>10}{'T_rollout':>11}{'T_bcast':>9}")
    for name, t in targets.items():
        tc = t if isinstance(t, (int, float)) else t[0]
        cfg = with_cfg(base, target_c_rl=tc)
        for d in windows_d:
            ev = evaluate(cfg, d * 86400)
            if not ev["runnable"]:
                print(f"  {name:<16}{tc:>11.1e}{d:>5}d{'no-fit':>6}"
                      f"{'weights exceed single node HBM (needs inference TP)':>63}")
                continue
            feas = "yes" if ev["feasible"] else "NO"
            gs = f"{ev['gpus']:,}" + ("" if ev["feasible"] else "*")
            cost_str = ("$" + fmt_num(ev["cost"])) if ev["feasible"] else "--"
            r = ev["r"]
            print(f"  {name:<16}{tc:>11.1e}{d:>5}d{feas:>6}{gs:>11}{ev['pct_stock']:>7.1f}%"
                  f"{fmt_ratio(ev['frac']):>8}{fmt_num(ev['steps']):>9}{fmt_time(ev['wall_s']):>9}"
                  f"{cost_str:>10}{ev['bottleneck']:>15}"
                  f"{fmt_time(r.t_step):>9}{fmt_time(r.t_update):>10}"
                  f"{fmt_time(r.stages['rollout+verify']):>11}{fmt_time(r.t_bc):>9}")
    print("  steps = training/RL steps to spend C_RL at this batch size (batch-dependent, not")
    print("  window-dependent -- see the total-rollout-batch table to trade steps for cluster size).")
    print("  T_step/T_update/T_rollout(+verify)/T_bcast = the single-STEP time breakdown at this")
    print("  operating point -- distinct from 'wall', which is the FULL run (steps x T_step).")
    print("  " + FLOOR_NOTE.replace("\n  ", "\n  "))


def centralised_vs_decentralised(base, target_c_rl, window_d=90):
    print(f"\n{_rule()}\n  CENTRALISED vs DECENTRALISED (target {target_c_rl:.1e}, {window_d}d window)\n{_rule()}")
    print(f"  {'setting':<30}{'feas':>6}{'GPUs':>11}{'wall':>9}{'tr:inf':>8}{'cost':>10}{'bottleneck':>16}")
    variants = [("decentralised (WAN 200Mbps)", dict(wan_mbps=200, compression=base.compression)),
                ("decentralised (int4 only 4x)", dict(wan_mbps=200, compression=4)),
                ("centralised (IB 400Gbps)",     dict(wan_mbps=400_000, compression=1))]
    for label, ov in variants:
        ev = evaluate(with_cfg(base, target_c_rl=target_c_rl, **ov), window_d * 86400)
        if not ev["runnable"]:
            print(f"  {label:<30}{'no-fit':>6}"); continue
        gs = f"{ev['gpus']:,}" + ("" if ev["feasible"] else "*")
        cost_str = ("$" + fmt_num(ev["cost"])) if ev["feasible"] else "--"
        print(f"  {label:<30}{('yes' if ev['feasible'] else 'NO'):>6}{gs:>11}{fmt_time(ev['wall_s']):>9}"
              f"{fmt_ratio(ev['frac']):>8}{cost_str:>10}{ev['bottleneck']:>16}")
    print("  (* = infeasible in window; wall = full-stock floor. int4-only-4x hitting the WAN wall is")
    print("   an A-iii artifact -- broadcast-every-step; a DiLoCo sync interval would relax it.)")


def policy_framing(base):
    print(f"\n{_rule()}\n  RL vs FLOP-THRESHOLD GOVERNANCE: where each RL target sits vs reporting "
          f"bars\n{_rule()}")
    print("  (an RL run's TOTAL FLOP; 'OVER' = covered by that bar)")
    bars = sorted(POLICY_THRESHOLDS.items(), key=lambda kv: kv[1])
    print(f"  {'RL target':<16}{'C_RL':>11}   " + "".join(f"{n:>16}" for n, _ in bars))
    for name, (tc, _) in MODEL_TARGETS.items():
        cells = "".join(f"{('OVER' if tc >= v else 'under'):>16}" for _, v in bars)
        print(f"  {name:<16}{tc:>11.1e}   {cells}")
    print("  -> disclosed RL stages sit UNDER even EU-systemic (1e25); the threshold binds on")
    print("     pre-training, not the RL stage. Core governance gap.")


def max_achievable_split(base):
    """The (T_step, nt, ni, r, frac) at the FULL stock that minimises step time. Reused by both
    the printed table and the CSV writer -- one optimal_split() call answers both, since
    per_step_flop is node-count-independent and total_time is ~monotone non-increasing in node
    count (the same property min_cluster's bisection relies on): more nodes never hurt, so the
    max-throughput operating point is always "use the whole stock, split it optimally", never a
    smaller subset."""
    n_total = base.stock_gpus // base.gpus_per_node
    return n_total, optimal_split(base, n_total)


def max_compute_operating_point(cfg, stock=None):
    """Everything about the max-throughput operating point that does NOT depend on the time
    window: split, T_step (+ the T_update/T_rollout+verify/T_broadcast breakdown), the ACHIEVED
    overall MFU of the whole allocated cluster, bottleneck, per-step FLOP. Computed ONCE per
    (cfg, stock) and shared across every window a caller loops over, so sweeping windows never
    re-runs optimal_split(). `stock` overrides cfg.stock_gpus (for the by-stock sweep); None uses
    cfg's own stock_gpus. Returns None if the model doesn't fit a single node at all.

    MFU HERE IS GENUINELY DERIVED, NOT AN ECHOED INPUT: r.mfu_model/r.mfu_hw (model.py) reduce
    algebraically to (mfu_train input) x (a fixed recompute-coefficient constant) -- they are
    IDENTICAL at every stock size because they only ever look at the trainer's OWN t_update, never
    at whether the trainer or inference pool sits idle waiting on the other. The overall MFU below
    instead divides total ACTUAL FLOPs (train + rollout, both pools) by (T_step x the COMBINED
    peak FLOPs/s of every GPU allocated, train+inference) -- T_step is the max/sum across stages,
    so a pool that finishes early and idles while the OTHER stage is the bottleneck correctly
    drags this number down. It genuinely varies with the split/bottleneck (e.g. craters when
    broadcast dominates at very large stock), unlike the flat mfu_model/mfu_hw echo."""
    c = cfg if stock is None else with_cfg(cfg, stock_gpus=stock)
    n_total, sol = max_achievable_split(c)
    if sol is None or not math.isfinite(sol[0]) or sol[3] is None:
        return None
    tt, nt, ni, r, frac = sol
    # BUG (fixed): `tt` is optimal_split's total_time() -- the wall-clock to finish cfg's OWN
    # target_c_rl at this split (n_steps(cfg) * r.t_step) -- NOT the per-step time. Using it here
    # made every "T_step" in this table off by a constant factor of n_steps(cfg) for whatever
    # target_c_rl happened to be set on cfg (e.g. 954x too large at the default 6e23), which made
    # every derived steps/max-C_RL number ~954x too SMALL. The real per-step time is r.t_step.
    pf = per_step_flop(cfg)   # node-count-independent, unaffected by stock; == r.tc.c_update + r.ro.c_rollout_total
    s = r.scenario
    train_peak = s.train_hw.n_nodes * s.train_hw.flops   # FLOPs/s, hardware peak (no MFU derating)
    inf_peak = s.inf_hw.n_nodes * s.inf_hw.flops
    total_peak = train_peak + inf_peak
    # model-FLOPs convention: strip the recompute coefficient off the training term only (inference
    # is already forward-only, no recompute distinction) -- mirrors model.py's own model_flops calc.
    model_flops_step = (3.0 / (3 + s.algo.recomp_act + s.algo.recomp_old)) * r.tc.c_update + r.ro.c_rollout_total
    return dict(gpus=n_total * cfg.gpus_per_node, n_total=n_total, nt=nt, ni=ni, frac=frac,
                t_step=r.t_step, t_update=r.t_update, t_rollout=r.stages["rollout+verify"],
                t_broadcast=r.t_bc, bottleneck=r.bottleneck,
                mfu_model=model_flops_step / (r.t_step * total_peak),
                mfu_hw=pf / (r.t_step * total_peak),
                per_step_flop=pf)


def steps_and_c_rl(op, window_s):
    """Window-dependent numbers from an operating point: (steps, max achievable C_RL). Pure
    arithmetic -- no simulate() calls -- so sweeping many windows over one op is free."""
    steps = window_s / op["t_step"]
    return steps, op["per_step_flop"] * steps


def max_compute_table(base, windows_d=(30, 90, 180, 720)):
    """THE INVERSE of the A1 table: instead of "given a target C_RL, what's the min cluster",
    this asks "given the FULL STOCK (+ model + RL/algo settings), what's the MOST RL compute
    achievable, at the optimal trainer:inference split, in each window". Throughput = per-step
    FLOP (fixed by the RL/model config, not by hardware) / T_step -- so maximising throughput is
    exactly minimising T_step, which optimal_split() already does; this table just multiplies by
    each window's length and reports which named targets that much compute would cover."""
    print(f"\n{_rule()}\n  MAX ACHIEVABLE RL COMPUTE (inverse of A1): full stock, optimal split, "
          f"{_cfg_hdr(base)}\n{_rule()}")
    op = max_compute_operating_point(base)
    if op is None:
        gpus = (base.stock_gpus // base.gpus_per_node) * base.gpus_per_node
        print(f"  no-fit: weights exceed a single node's HBM at {gpus:,} GPUs -- needs inference TP")
        return
    print(f"  stock: {op['gpus']:,} GPUs ({op['n_total']:,} nodes) | optimal split tr:inf "
          f"{fmt_ratio(op['frac'])} ({op['nt']:,} train / {op['ni']:,} inf nodes)")
    print(f"  per-step FLOP {op['per_step_flop']:.3e} | T_step {fmt_time(op['t_step'])} | "
          f"bottleneck {op['bottleneck']}")
    print(f"    breakdown: T_update {fmt_time(op['t_update'])} | T_rollout+verify "
          f"{fmt_time(op['t_rollout'])} | T_broadcast {fmt_time(op['t_broadcast'])}")
    print(f"  MFU (achieved, whole allocated cluster, hw-FLOPs): {op['mfu_hw']*100:.1f}%")
    print("    (divides actual FLOPs done by T_step x combined train+inference peak FLOPs/s -- a")
    print("     pool idling on the OTHER stage's bottleneck correctly drags this down; NOT an echo")
    print("     of the mfu_train/mfu_inf config inputs. hw-FLOPs (8N, includes recompute) is the")
    print("     convention published RL-training MFU figures use, e.g. prime-rl's 38%; the model-")
    print("     FLOPs (6N, pretraining-style) figure is also computed -- see mfu_model in the CSV --")
    print("     but dropped from this printout as a second number that rarely changes the story.)")
    thr = op["per_step_flop"] / op["t_step"]
    print(f"  throughput: {thr:.3e} FLOP/s  =  {thr*86400:.3e} FLOP/day")
    print(f"\n  {'window':>8}{'steps':>12}{'max C_RL':>14}   covers")
    targets = all_targets()
    for d in windows_d:
        steps, c_rl = steps_and_c_rl(op, d * 86400)
        covered = [target_label(name, tc) for name, tc in targets.items() if tc <= c_rl]
        print(f"  {d:>6}d{fmt_num(steps):>12}{c_rl:>14.2e}   {', '.join(covered) or '(none)'}")
    print("  (max C_RL = per-step FLOP x window / T_step, at the full-stock optimal split; 'covers'")
    print("   lists named/threshold targets at or below that ceiling.)")


def write_max_compute_csv(base, path="experiments/max_compute.csv", windows_d=(30, 90, 180, 720)):
    import csv, os
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    op = max_compute_operating_point(base)
    with open(path, "w", newline="") as fh:
        write_config_comment(fh, base)
        w = csv.writer(fh)
        w.writerow(["model", "gpus", "nodes", "runnable", "train_frac", "t_step_s", "t_update_s",
                    "t_rollout_s", "t_broadcast_s", "mfu_model_achieved", "mfu_hw_achieved",
                    "per_step_flop", "bottleneck", "window_days", "steps", "max_c_rl", "covers"])
        if op is None:
            gpus = (base.stock_gpus // base.gpus_per_node) * base.gpus_per_node
            w.writerow([base.model.name, gpus, base.stock_gpus // base.gpus_per_node, 0,
                        "", "", "", "", "", "", "", "", "no-fit", "", "", "", ""])
        else:
            targets = all_targets()
            for d in windows_d:
                steps, c_rl = steps_and_c_rl(op, d * 86400)
                covered = [target_label(name, tc) for name, tc in targets.items() if tc <= c_rl]
                w.writerow([base.model.name, op["gpus"], op["n_total"], 1, round(op["frac"], 3),
                            round(op["t_step"], 2), round(op["t_update"], 2), round(op["t_rollout"], 2),
                            round(op["t_broadcast"], 2), round(op["mfu_model"], 4), round(op["mfu_hw"], 4),
                            f"{op['per_step_flop']:.3e}", op["bottleneck"], d, round(steps),
                            f"{c_rl:.3e}", "|".join(covered)])
    print(f"\n  wrote {path}")


# ---------------------------------------------------------------------------
# Matrix sweeps (all via render_matrix)
# ---------------------------------------------------------------------------
def gpu_target_matrix(base, targets, gpus_l=HIGH_END_GPUS, window_d=90, gpus_per_node=None):
    """min cluster per (target, GPU family). gpus_per_node=None uses the config's node size;
    pass a dict {gpu: n} (e.g. from an export cap) to size each family's node individually."""
    def cell(t, g):
        gpn = base.gpus_per_node if gpus_per_node is None else gpus_per_node[g]
        return cell_text(evaluate(with_cfg(base, gpu=g, gpus_per_node=gpn,
                                           target_c_rl=targets[t]), window_d * 86400))
    render_matrix(
        f"GPU FAMILY x TARGET ({window_d}d) -- HBM-bandwidth story. {CELL_NOTE}",
        list(targets), list(gpus_l), cell,
        row_label=lambda t: target_label(t, targets[t]),
        col_label=lambda g: f"{g} {GPUS[g]['bw_tbs']}TB/s", rw=target_label_width(targets))


def export_cap_table(base, target_c_rl, target_name="target", gpus_l=HIGH_END_GPUS,
                     h100_equiv=16, window_d=180):
    """Cross-GPU feasibility under a BIS-style per-node export cap: a node may not exceed
    `h100_equiv` H100s of TPP (compute) OR of HBM (capacity). For each GPU family this caps the
    node size, which is the governance-relevant twist -- high-FLOP/high-HBM parts get forced into
    SMALL nodes (a B300 node caps at 4 GPUs vs H100's 16), and a small node may no longer hold a
    large model's weights for single-node inference at all ('no-fit'), independent of how many
    nodes you have. Shows the cap breakdown then min-cluster feasibility at the capped node size,
    with the uncapped (config node) result alongside so the cap's effect is visible.

    h100_equiv is the knob: the cap is N H100-equivalents (default 16 = 15,824 TFLOP/s /
    253,184 TFLOP-bit/s TPP / 1,280 GB). See feasibility_core.node_gpus_tpp_cap."""
    tc = target_c_rl
    tname = target_name
    print(f"\n{_rule(150)}\n  EXPORT-CAP NODE SIZING x FEASIBILITY: node <= {h100_equiv} H100-equiv "
          f"of TPP and HBM (target {target_label(tname, tc)}, {window_d}d)\n{_rule(150)}")
    print(f"  {'GPU':<9}{'TF/s':>7}{'TPP/chip':>11}{'HBM GB':>8}{'n:TPP':>7}{'n:HBM':>7}"
          f"{'node':>6}{'cap by':>8}{'fit?':>6}{'min GPUs@cap':>14}{'wall':>8}"
          f"{'vs node='+str(base.gpus_per_node):>16}")
    for g in gpus_l:
        n_tpp = node_gpus_tpp_cap(g, h100_equiv, cap_by_memory=False)
        n_hbm = int(h100_equiv * GPUS["H100"]["hbm_gb"] // GPUS[g]["hbm_gb"])
        node = node_gpus_tpp_cap(g, h100_equiv)
        capby = "TPP" if node == n_tpp and n_tpp <= n_hbm else "HBM"
        ev = evaluate(with_cfg(base, gpu=g, gpus_per_node=node, target_c_rl=tc), window_d * 86400)
        std = evaluate(with_cfg(base, gpu=g, target_c_rl=tc), window_d * 86400)
        fit = "no" if not ev["runnable"] else "yes"
        cap_cell = "no-fit" if not ev["runnable"] else (
            (f"{ev['gpus']:,}" + ("" if ev["feasible"] else "*")))
        wall = "--" if not ev["runnable"] else fmt_time(ev["wall_s"])
        std_cell = "no-fit" if not std["runnable"] else (
            (f"{std['gpus']:,}" + ("" if std["feasible"] else "*")))
        print(f"  {g:<9}{GPUS[g]['flops']/1e12:>7.0f}{tpp_per_chip(g)/1e12:>9.0f}TB"
              f"{GPUS[g]['hbm_gb']:>8}{n_tpp:>7}{n_hbm:>7}{node:>6}{capby:>8}{fit:>6}"
              f"{cap_cell:>14}{wall:>8}{std_cell:>16}")
    print(f"  TPP = flops x {16} bit (BF16 rating); cap = {h100_equiv} x H100 (15,824 TFLOP/s, 1,280 GB).")
    print("  node = min(TPP-limited, HBM-limited) chips. 'min GPUs@cap' = min cluster at the capped")
    print(f"  node size; last col = same target at the config's {base.gpus_per_node}-GPU node, for contrast.")
    print("  " + FLOOR_NOTE.replace("\n  ", "\n  "))


def power_node_table(gpus_l=HIGH_END_GPUS, caps_kw=(10, 40, 132)):
    """Max chips per node under a per-node POWER budget, from board TDP -- a second physical node
    cap (a rack PDU / cooling limit) independent of the TPP/HBM export cap. Standalone reference
    table: caps_kw defaults span a single high-density server (~10kW), a small rack (~40kW), and a
    full GB200 NVL72-class rack (~132kW)."""
    print(f"\n{_rule()}\n  MAX NODE SIZE UNDER PER-NODE POWER CAP (GPUs/node = floor(cap / board W))"
          f"\n{_rule()}")
    print(f"  {'GPU':<9}{'board W':>9}" + "".join(f"{f'{c} kW':>9}" for c in caps_kw))
    for g in gpus_l:
        cells = "".join(f"{node_gpus_power_cap(g, c):>9}" for c in caps_kw)
        print(f"  {g:<9}{GPUS[g]['watts']:>9}" + cells)
    print("  (board W = per-GPU TDP; excludes host/NIC/cooling overhead, so real nodes fit fewer.)")


def power_envelope_table(gpus_l=HIGH_END_GPUS, caps_mw=(1, 5)):
    """Max TOTAL GPUs a SITE can run under a power envelope (1 MW / 5 MW), from board TDP. A
    site/detectability cap on GPU COUNT -- distinct from the per-node power table (rack limit) and
    from the TPP/HBM export cap (which caps NODE size). floor(MW x 1e6 / board_W)."""
    print(f"\n{_rule()}\n  MAX GPUs UNDER SITE POWER ENVELOPE (total GPUs = floor(MW / board W))"
          f"\n{_rule()}")
    print(f"  {'GPU':<9}{'board W':>9}" + "".join(f"{f'{c} MW':>12}" for c in caps_mw))
    for g in gpus_l:
        cells = "".join(f"{node_gpus_power_cap(g, c * 1000):>12,}" for c in caps_mw)
        print(f"  {g:<9}{GPUS[g]['watts']:>9}" + cells)
    print("  (board W = per-GPU TDP; real sites also draw host/NIC/network/cooling, so fewer fit.)")


def bandwidth_requirement_table(base, targets=None, window_d=180):
    """What LINK does the weight broadcast need, per model, so the network is never the
    bottleneck? Replaces the old WAN x compression grid, which mostly restated A1's feasibility
    answer at 16 (bandwidth, compression) pairs; this asks the question a monitor or an evader
    actually asks, and answers it in one number per (model, compute budget).

    Reported at 1x compression, i.e. the raw uncompressed weight volume over the time budget.
    That keeps the figure a property of model size and window alone -- stable when the compression
    knob is retuned -- and any real compression factor simply divides it (int4-at-4x needs a
    quarter of the quoted rate). Evaluated at each (model, target)'s own min-cluster operating
    point, since that is the cluster an evader would actually be running.
    """
    targets = targets or model_targets()
    rows = [n for n, m in MODELS.items() if smallest_fitting_gpu(m, base.gpus_per_node)]
    def cell(name, tname):
        gpu = smallest_fitting_gpu(MODELS[name], base.gpus_per_node)
        cfg = with_cfg(base, model=MODELS[name], gpu=gpu, target_c_rl=targets[tname])
        mbps, ev = required_bcast_mbps(cfg, window_d * 86400)
        if mbps is None:
            return "no-fit" if not ev["runnable"] else "unbounded"
        return f"{fmt_rate(mbps)}Mbps@{fmt_num(ev['gpus'])}g"
    render_matrix(
        f"REQUIRED BROADCAST LINK ({window_d}d): RAW (1x-compression) link Mbps for weight sync\n"
        f"  to stop being the bottleneck, at each model's min-cluster operating point",
        rows, list(targets),
        cell, row_label=lambda n: f"{n} ({2.0*MODELS[n].p_total/GB:,.0f}GB wt)",
        col_label=lambda t: target_label(t, targets[t]),
        cw=target_label_width(targets), rw=30, width=32 + len(targets) * target_label_width(targets),
        note=("cell = required RAW (1x) link Mbps @ the GPU count it was measured at. Divide by your\n"
              f"compression factor for the wire rate (config is currently {base.compression:g}x).\n"
              "'unbounded' = the compute stage is already faster than the network's fixed latency, so\n"
              "no bandwidth suffices. Bigger models need MORE bandwidth (weights scale) but also spend\n"
              "longer computing per step (more time to hide the sync in) -- the ratio is the story."))


def batch_group_table(base, targets=None, batches=BATCH_SWEEP, window_d=180):
    """Latency-wall relief vs TOTAL rollout batch, one row per batch size.

    Replaces a B x n grid. At fixed total batch the two axes are near-substitutes here (both
    only cut the serial step count; n is marginally cheaper via shared prefill), so the grid's
    diagonals were largely redundant -- the product is the variable that moves the answer. B and n
    are still shown per row, split even-ish by batch_split(), so the config behind each row is
    explicit rather than implied."""
    targets = targets or model_targets()
    render_matrix(
        f"TOTAL ROLLOUT BATCH ({window_d}d): min cluster vs batch size, B x n split even-ish\n"
        f"  (both axes only cut serial steps in this model, so the PRODUCT is what moves it)",
        list(batches), list(targets),
        lambda t, tn: cell_text(evaluate(
            with_cfg(base, target_c_rl=targets[tn], prompts_per_batch=batch_split(t)[0],
                     responses_per_prompt=batch_split(t)[1]), window_d * 86400)),
        row_label=lambda t: f"{fmt_num(t)} ({batch_split(t)[0]}x{batch_split(t)[1]})",
        col_label=lambda tn: target_label(tn, targets[tn]),
        cw=target_label_width(targets), rw=18,
        width=20 + len(targets) * target_label_width(targets),
        note=(f"batch = questions B x samples/question n. {BATCH_CAP:,} is the largest EMPIRICALLY\n"
              "TESTED batch (arXiv 2603.12151); rows beyond it are extrapolation past this model's\n"
              "calibration. Bigger batch -> fewer serial steps -> beats the latency wall; C_RL fixed.\n"
              + CELL_NOTE))


def hbm_scaling_table(base, targets=None, mults=HBM_MULTS, window_d=180):
    """HBM bandwidth on its own axis, not just a x0.5/x2 tornado bar. Two reasons it earns a
    dedicated sweep: it is the binding constraint in this model's rollout-bound regime (it beats
    every network lever in the A3 tornado), and it is the lever export controls actually move --
    memory bandwidth is what the H20-style cut-down parts sacrifice. So "how much does min-cluster
    size move per unit of memory bandwidth" is a directly policy-relevant elasticity."""
    targets = targets or model_targets()
    render_matrix(
        f"HBM BANDWIDTH SCALING ({window_d}d): min cluster vs memory bandwidth multiplier",
        list(mults), list(targets),
        lambda k, t: cell_text(evaluate(
            with_cfg(base, target_c_rl=targets[t], inf_bw_mult=base.inf_bw_mult * k),
            window_d * 86400)),
        row_label=lambda k: f"HBM x{k:g}", col_label=lambda t: target_label(t, targets[t]),
        cw=target_label_width(targets), rw=12, width=14 + len(targets) * target_label_width(targets),
        note=("x1 = the config's own GPU. Rollout is HBM-bound in this regime, so bandwidth buys\n"
              "cluster size almost linearly until another stage takes over as bottleneck.\n" + CELL_NOTE))


def batch_feasibility_table(base, targets, batches=(256, 1024, 4096), window_d=180):
    """Feasibility vs prompts/batch (G fixed at the config's value)."""
    G = base.responses_per_prompt
    render_matrix(
        f"BATCH-SIZE x TARGET ({window_d}d) as the rollout batch grows (G={G})",
        list(targets), list(batches),
        lambda t, b: cell_text(evaluate(
            with_cfg(base, target_c_rl=targets[t], prompts_per_batch=b), window_d * 86400)),
        col_label=lambda b: f"{b*G//1000}k batch", rw=14,
        note="bigger batch -> fewer serial steps -> beats the latency wall; C_RL unchanged.\n" + CELL_NOTE)


# ---------------------------------------------------------------------------
# Bespoke-shape tables (extra leading columns / paired sub-columns)
# ---------------------------------------------------------------------------
def gpu_family_check(base, target_c_rl=2.5e25, window_d=90,
                     gpus_l=("H100", "H200", "B200", "A100", "RTX4090")):
    print(f"\n  GPU-family check ({target_c_rl:.1e}, {window_d}d) -- GPUs / wall / cost / sites@power ----")
    for gpu in gpus_l:
        ev = evaluate(with_cfg(base, gpu=gpu, target_c_rl=target_c_rl), window_d * 86400)
        if not ev["runnable"]:
            print(f"    {gpu:<8} no-fit"); continue
        ns, mw = sites_power(with_cfg(base, gpu=gpu), ev["gpus"])
        flag = "" if ev["feasible"] else "  (infeasible: floor at full stock)"
        print(f"    {gpu:<8} {ev['gpus']:>9,} GPUs | {fmt_time(ev['wall_s']):>7} | ${fmt_num(ev['cost']):>6} | "
              f"tr:inf {fmt_ratio(ev['frac'])} | {ns:>5.0f} sites @ {base.site_power_mw:.0f}MW ({mw:.1f} MW){flag}")


def power_sites_table(base, targets, window_d=90, caps_mw=(1, 10, 100)):
    print(f"\n{_rule()}\n  NODE/SITE RESTRICTION: sites needed at each per-site power cap "
          f"({window_d}d)\n{_rule()}")
    print("  (a site must stay under the cap to dodge the detection tier; more sites = more exposure.")
    print("   For infeasible targets the GPU count shown is the full stock at the wall-clock floor.)")
    tw = target_label_width(targets)
    hdr = "".join(f"{f'{c}MW':>14}" for c in caps_mw)
    print(f"  {'target':<{tw}}{'feas':>6}{'GPUs':>11}{'wall':>9}{'total MW':>10}{hdr}")
    w = GPUS[base.gpu]["watts"]
    for name, tc in targets.items():
        lab = target_label(name, tc)
        ev = evaluate(with_cfg(base, target_c_rl=tc), window_d * 86400)
        if not ev["runnable"]:
            print(f"  {lab:<{tw}}{'no-fit':>6}"); continue
        gpus = ev["gpus"]
        cells = "".join(f"{gpus/(c*1e6/w):>12,.0f} st" for c in caps_mw)
        print(f"  {lab:<{tw}}{('yes' if ev['feasible'] else 'NO'):>6}{gpus:>11,}{fmt_time(ev['wall_s']):>9}"
              f"{gpus*w/1e6:>9.1f}M{cells}")


def frontier_diagnosis(base):
    print(f"\n{_rule()}\n  LATENCY WALL: min wall-clock throwing the WHOLE stock at one run (optimal "
          f"split); batch scaling cuts serial steps\n{_rule()}")
    print(f"  {'target':<14}{'C_RL':>10}{'setting':<22}{'steps':>9}{'per-step floor':>15}"
          f"{'tr:inf':>8}{'min wall @stock':>16}")
    # Labels computed from the actual (prompts_per_batch, base.responses_per_prompt) rather than
    # hardcoded text: a fixed "(256x16)"/"batch 4k" string silently goes stale the moment
    # base.prompts_per_batch or responses_per_prompt is retuned (as just happened -- base is now
    # 128x512=65,536, not 256x16=4,096, so the old "4k/16k/64k" labels were describing numbers
    # ~16x smaller than what was actually being computed).
    cases = [{}, dict(prompts_per_batch=base.prompts_per_batch * 4),
             dict(prompts_per_batch=base.prompts_per_batch * 16)]
    # Take the three LARGEST real model targets rather than hardcoded (name, C_RL) pairs: the old
    # literals had drifted to values that matched no target in the catalogue any more (1e24/3e24/
    # 1e25 vs the catalogue's o3 1e24 / Grok-3 2.5e25 / Grok-4 2.5e26), so the table was labelling
    # rows with model names whose printed C_RL was not that model's estimate. The latency wall is
    # a big-target phenomenon, so the top three are also the right three to show.
    for name, tc in model_targets(top=3).items():
        for ov in cases:
            p = ov.get("prompts_per_batch", base.prompts_per_batch)
            label = f"batch {fmt_num(p * base.responses_per_prompt)} ({p}x{base.responses_per_prompt})"
            cfg = with_cfg(base, target_c_rl=tc, **ov)
            sol = min_wall_full_stock(cfg)
            floor = fmt_time(sol["r"].t_step) if sol["r"] else "no-fit"
            print(f"  {name:<14}{tc:>10.1e}{label:<22}{n_steps(cfg):>9,.0f}{floor:>15}"
                  f"{fmt_ratio(sol['frac']):>8}{fmt_time(sol['tt']):>16}")
        print()


def batch_compare_table(base, targets=None, windows_d=(30, 90, 180, 720)):
    """A1 per target, baseline batch vs high-end batch, side by side per window.
    '*' after a high-batch cell = it flips feasible where baseline was not.

    Defaults to model_targets() (real published runs) rather than all_targets(): the batch-size
    lever is an engineering choice a specific lab made for a specific run, so pairing it with
    "C=1e24"-style governance reporting thresholds compared a decision against a regulation."""
    targets = targets or model_targets()
    print(f"\n{_rule(118)}\n  BATCH COMPARISON: baseline ({BASELINE_BATCH['prompts_per_batch']}x"
          f"{BASELINE_BATCH['responses_per_prompt']}) vs high-end ({HIGH_BATCH['prompts_per_batch']}x"
          f"{HIGH_BATCH['responses_per_prompt']}) rollout batch, per window\n{_rule(118)}")
    print(f"  {'target':<14}{'C_RL':>11}{'base steps':>12}{'high steps':>12}  "
          + "".join(f"{f'{d}d base':>17}{f'{d}d high':>18}" for d in windows_d))
    for name, tc in targets.items():
        cb = with_cfg(base, target_c_rl=tc, **BASELINE_BATCH)
        ch = with_cfg(base, target_c_rl=tc, **HIGH_BATCH)
        cells = []
        for d in windows_d:
            eb, eh = evaluate(cb, d * 86400), evaluate(ch, d * 86400)
            flip = "*" if (eh["feasible"] and not eb["feasible"]) else ""
            cells.append(f"{cell_text(eb):>17}{cell_text(eh) + flip:>18}")
        print(f"  {name:<14}{tc:>11.1e}{fmt_num(n_steps(cb)):>12}{fmt_num(n_steps(ch)):>12}  "
              + "".join(cells))
    print("  (steps = training steps at that batch size, fixed across windows; " + CELL_NOTE + ";")
    print("   * = high-batch flips feasible. high-end = 128 prompts x G=512.)")


def model_sweep(base, target_c_rl=2.5e25, windows_d=(90, 180, 720)):
    """A1 across models at fixed C_RL and E[R] (isolates the model's effect). Each model runs on
    the smallest GPU family whose node holds its weights, so cost is NOT cross-comparable -- the
    fit tier and bottleneck are the story."""
    print(f"\n{_rule(118)}\n  CROSS-MODEL FEASIBILITY: C_RL={target_c_rl:.1e} (E[R]={base.er:,.0f} fixed, "
          f"{base.wan_mbps:.0f}Mbps, {base.compression:.0f}x)\n{_rule(118)}")
    print(f"  {'model':<18}{'tot/act B':>13}{'wt GB':>8}{'node':>6}{'steps':>9}"
          + "".join(f"{f'{d}d (GPU/wall/tr:inf)':>27}" for d in windows_d))
    for name, m in MODELS.items():
        wt_gb = 2.0 * m.p_total / GB
        gpu = smallest_fitting_gpu(m, base.gpus_per_node)
        tot_act = f"{m.p_total/1e9:.0f}/{m.p_active_layers/1e9:.0f}"
        if gpu is None:
            print(f"  {name:<18}{tot_act:>13}{wt_gb:>7,.0f}{'--':>6}   weights exceed an 8xB200 node "
                  f"-> multi-node inference TP (not modelled)")
            continue
        cfg = with_cfg(base, model=m, gpu=gpu, target_c_rl=target_c_rl)
        steps = target_c_rl / per_step_flop(cfg)
        cells = "".join(f"{cell_text(evaluate(cfg, d*86400)):>27}" for d in windows_d)
        print(f"  {name:<18}{tot_act:>13}{wt_gb:>7,.0f}{gpu:>6}{steps:>9,.0f}" + cells)
    print("  (node = smallest GPU family whose 8-GPU node holds the weights; " + CELL_NOTE + ".")
    print("   Cost/GPUs not cross-comparable across node tiers.)")


def max_compute_by_model(base, windows_d=(90, 180, 720)):
    """Cross-model variant of max_compute_table: at the SAME fixed stock/hardware/RL settings,
    which model choice gets the most RL compute done? Each model runs on the smallest GPU family
    whose node holds its weights (mirrors model_sweep), so this also implicitly compares node
    tiers, not just the model's own compute/HBM profile."""
    print(f"\n{_rule(150)}\n  MAX ACHIEVABLE RL COMPUTE BY MODEL: same stock ({fmt_num(base.stock_gpus)} GPUs), "
          f"E[R]={base.er:,.0f} fixed\n{_rule(150)}")
    print(f"  {'model':<18}{'node':>6}{'tr:inf':>8}{'T_step':>9}{'T_update':>10}{'T_rollout':>11}"
          f"{'T_bcast':>9}{'MFU':>6}{'bottleneck':>16}" + "".join(f"{f'{d}d max C_RL':>16}" for d in windows_d))
    for name, m in MODELS.items():
        gpu = smallest_fitting_gpu(m, base.gpus_per_node)
        if gpu is None:
            print(f"  {name:<18}  weights exceed an 8xB200 node -> multi-node inference TP (not modelled)")
            continue
        op = max_compute_operating_point(with_cfg(base, model=m, gpu=gpu))
        if op is None:
            print(f"  {name:<18}{gpu:>6}  no-fit at this stock")
            continue
        cells = "".join(f"{steps_and_c_rl(op, d * 86400)[1]:>16.2e}" for d in windows_d)
        print(f"  {name:<18}{gpu:>6}{fmt_ratio(op['frac']):>8}{fmt_time(op['t_step']):>9}"
              f"{fmt_time(op['t_update']):>10}{fmt_time(op['t_rollout']):>11}"
              f"{fmt_time(op['t_broadcast']):>9}{op['mfu_hw']*100:>5.0f}%"
              f"{op['bottleneck']:>16}" + cells)
    print("  (max C_RL = per-step FLOP x window / T_step at the full-stock optimal split, per model;")
    print("   MFU is the ACHIEVED overall-cluster figure (actual FLOPs / T_step x combined peak,")
    print("   hw-FLOPs/8N convention -- matches published RL-training MFU figures e.g. prime-rl's")
    print("   38%; not an echo of the mfu_train/mfu_inf config inputs). Different models sit on")
    print("   different node tiers so GPU counts aren't comparable, but achievable C_RL per window")
    print("   is the direct \"which model gets furthest\" comparison.)")


# Default GPU-count sweep for max_compute_by_stock / write_max_compute_by_stock_csv / the dashboard
# panel. Edit this tuple (here, or pass stocks= explicitly) to change which stock sizes are swept.
STOCK_SWEEP = (16, 128, 1024, 10_000, 100_000, 1_000_000)


def max_compute_by_stock(base, stocks=STOCK_SWEEP, windows_d=(30, 90, 180, 720)):
    """max_compute_table(), swept across GPU stock sizes instead of just base.stock_gpus --
    "at N GPUs (for a range of N), how far could RL training get". One row per stock with the
    full operating point (split, T_step, MFU, bottleneck -- window-independent, computed once per
    stock) plus a compact steps/max-C_RL cell per window. Bespoke printer (not render_matrix):
    the per-row operating-point columns don't fit a pure rows x cols cell shape."""
    print(f"\n{_rule(160)}\n  MAX ACHIEVABLE RL COMPUTE x GPU STOCK: {base.model.name}, optimal "
          f"split per stock size ({_cfg_hdr(base)})\n{_rule(160)}")
    print(f"  {'stock':<12}{'tr:inf':>8}{'T_step':>9}{'T_update':>10}{'T_rollout':>11}"
          f"{'T_bcast':>9}{'MFU':>6}{'bottleneck':>16}  " + "".join(f"{f'{d}d steps/max C_RL':>26}" for d in windows_d))
    for stock in stocks:
        op = max_compute_operating_point(base, stock)
        if op is None:
            print(f"  {fmt_num(stock)+' GPUs':<12}no-fit: weights exceed a single node at this gpus_per_node")
            continue
        cells = []
        for d in windows_d:
            steps, c_rl = steps_and_c_rl(op, d * 86400)
            cells.append(f"{fmt_num(steps) + '/' + f'{c_rl:.2e}':>26}")
        print(f"  {fmt_num(stock)+' GPUs':<12}{fmt_ratio(op['frac']):>8}{fmt_time(op['t_step']):>9}"
              f"{fmt_time(op['t_update']):>10}{fmt_time(op['t_rollout']):>11}"
              f"{fmt_time(op['t_broadcast']):>9}{op['mfu_hw']*100:>5.0f}%"
              f"{op['bottleneck']:>16}  " + "".join(cells))
    print("  (MFU is the ACHIEVED overall-cluster figure -- actual FLOPs done (train+rollout) over")
    print("   T_step x combined train+inference peak FLOPs/s, so a pool idling on the OTHER stage's")
    print("   bottleneck drags it down; it is NOT an echo of the mfu_train/mfu_inf config inputs.")
    print("   hw-FLOPs/8N convention shown (matches published RL-training MFU figures, e.g. prime-")
    print("   rl's 38%; the model-FLOPs/6N figure is also computed -- see mfu_model in the CSV).")
    print("   Cell = training steps / max achievable C_RL (FLOP) at that stock+window, at the")
    print("   full-stock-for-that-row optimal split.)")


def write_max_compute_by_stock_csv(base, path="experiments/max_compute_by_stock.csv",
                                   stocks=STOCK_SWEEP, windows_d=(30, 90, 180, 720)):
    import csv, os
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        write_config_comment(fh, base)
        w = csv.writer(fh)
        w.writerow(["model", "stock_gpus", "nodes", "runnable", "train_frac", "t_step_s",
                    "t_update_s", "t_rollout_s", "t_broadcast_s", "mfu_model_achieved",
                    "mfu_hw_achieved", "bottleneck", "window_days", "steps", "max_c_rl"])
        for stock in stocks:
            op = max_compute_operating_point(base, stock)
            if op is None:
                for d in windows_d:
                    w.writerow([base.model.name, stock, "", 0, "", "", "", "", "", "", "",
                                "no-fit", d, "", ""])
                continue
            for d in windows_d:
                steps, c_rl = steps_and_c_rl(op, d * 86400)
                w.writerow([base.model.name, stock, op["n_total"], 1, round(op["frac"], 3),
                            round(op["t_step"], 2), round(op["t_update"], 2), round(op["t_rollout"], 2),
                            round(op["t_broadcast"], 2), round(op["mfu_model"], 4), round(op["mfu_hw"], 4),
                            op["bottleneck"], d, round(steps), f"{c_rl:.3e}"])
    print(f"\n  wrote {path}")


def sensitivity(base, target_c_rl, window_d=90):
    """A3: at the operating point for the target/window (min cluster if feasible, else the
    full-stock floor), perturb each lever x0.5 and x2 -> binding constraint (biggest swing)."""
    cfg = with_cfg(base, target_c_rl=target_c_rl)
    ev = evaluate(cfg, window_d * 86400)
    if not ev["runnable"]:
        print("\n  (sensitivity: base model does not fit a node)"); return
    N, base_t = ev["N"], ev["wall_s"]
    tag = "min cluster" if ev["feasible"] else "full-stock floor (infeasible in window)"
    print(f"\n{_rule()}\n  SENSITIVITY (A3): {target_c_rl:.1e} at its {window_d}d {tag} "
          f"({ev['gpus']:,} GPUs, tr:inf {fmt_ratio(ev['frac'])}); wall-clock elasticity per lever\n{_rule()}")
    print(f"  base wall-clock {fmt_time(base_t)}, bottleneck {ev['bottleneck']}")
    print(f"  {'lever':<22}{'x0.5':>12}{'x2':>12}{'swing':>9}")
    levers = {
        "WAN bandwidth":   lambda c, k: with_cfg(c, wan_mbps=c.wan_mbps * k),
        "compression":     lambda c, k: with_cfg(c, compression=c.compression * k),
        "HBM bandwidth":   lambda c, k: with_cfg(c, inf_bw_mult=c.inf_bw_mult * k),
        "HBM capacity":    lambda c, k: with_cfg(c, hbm_mult=c.hbm_mult * k),
        "inference FLOP":  lambda c, k: with_cfg(c, mfu_inf=c.mfu_inf * k),
        "training FLOP":   lambda c, k: with_cfg(c, mfu_train=c.mfu_train * k),
        "E[R] length":     lambda c, k: with_cfg(c, er=c.er * k),
        "node count":      None,   # handled specially (scale the operating-point split)
    }
    rows = []
    for name, fn in levers.items():
        if name == "node count":
            lo = total_time(cfg, max(1, int(ev["nt"] * 0.5)), max(1, int(ev["ni"] * 0.5)))[0]
            hi = total_time(cfg, ev["nt"] * 2, ev["ni"] * 2)[0]
        else:
            lo = total_time(_c := fn(cfg, 0.5), *split_for(_c, N))[0]
            hi = total_time(_c := fn(cfg, 2.0), *split_for(_c, N))[0]
        rows.append((name, lo / base_t, hi / base_t, max(lo, hi) / min(lo, hi)))
    for name, lo, hi, swing in sorted(rows, key=lambda x: -x[3]):
        print(f"  {name:<22}{lo:>11.2f}x{hi:>11.2f}x{swing:>8.1f}x")
    print("  (biggest swing = the binding constraint -> the monitoring/countermeasure lever)")


def er_table(base, targets, ers=(1_000, 10_000, 50_000, 100_000, 500_000), window_d=180):
    """Effect of mean response length E[R] on feasibility, grouped by RL compute TARGET rather
    than swept at one fixed target: for each named target (DeepSeek-R1-Zero, o3, Grok-3, ...),
    how does the min cluster/split/step-time breakdown change across representative response
    lengths from short (1k, single-turn math-ish) to long (500k, deep multi-turn agentic
    territory)? Longer responses inflate the quadratic rollout/verify attention+KV terms but also
    raise per-step FLOP (more tokens generated per step), so steps needed FALLS as E[R] rises even
    as each step gets more expensive -- the two effects compete, and which one wins the wall-clock
    race can flip depending on how much compute the target itself demands (a small target has
    room to absorb longer responses in wall-clock; a huge one is already latency-wall-bound and
    every extra token of rollout time bites harder). Standardised on a single 180d window
    (matching the dashboard's 2c/2d/2e/2g panels) rather than a multi-window sweep -- this table
    already has two swept axes (target x E[R]); a third (window) would be unreadable in text."""
    print(f"\n{_rule(150)}\n  E[R] SENSITIVITY: min cluster by response length x target "
          f"({window_d}d, {_cfg_hdr(base)})\n{_rule(150)}")
    print(f"  {'target':<16}{'C_RL':>11}{'E[R]':>8}{'feas':>6}{'GPUs':>11}{'%stock':>8}"
          f"{'tr:inf':>8}{'steps':>9}{'wall':>9}{'cost':>10}{'bottleneck':>15}"
          f"{'T_step':>9}{'T_update':>10}{'T_rollout':>11}{'T_bcast':>9}")
    for name, t in targets.items():
        tc = t if isinstance(t, (int, float)) else t[0]
        for er in ers:
            cfg = with_cfg(base, target_c_rl=tc, er=er)
            ev = evaluate(cfg, window_d * 86400)
            if not ev["runnable"]:
                print(f"  {name:<16}{tc:>11.1e}{fmt_num(er):>8}{'no-fit':>6}"
                      f"{'weights exceed single node HBM (needs inference TP)':>63}")
                continue
            feas = "yes" if ev["feasible"] else "NO"
            gs = f"{ev['gpus']:,}" + ("" if ev["feasible"] else "*")
            cost_str = ("$" + fmt_num(ev["cost"])) if ev["feasible"] else "--"
            r = ev["r"]
            print(f"  {name:<16}{tc:>11.1e}{fmt_num(er):>8}{feas:>6}{gs:>11}{ev['pct_stock']:>7.1f}%"
                  f"{fmt_ratio(ev['frac']):>8}{fmt_num(ev['steps']):>9}{fmt_time(ev['wall_s']):>9}"
                  f"{cost_str:>10}{ev['bottleneck']:>15}"
                  f"{fmt_time(r.t_step):>9}{fmt_time(r.t_update):>10}"
                  f"{fmt_time(r.stages['rollout+verify']):>11}{fmt_time(r.t_bc):>9}")
    print("  (steps = training steps to spend C_RL at this E[R] (batch fixed) -- longer responses")
    print("   raise per-step FLOP so fewer, chunkier steps are needed; whether that beats the extra")
    print("   per-step rollout/verify time is the wall-clock story above; T_step/T_update/")
    print("   T_rollout(+verify)/T_bcast = the single-step breakdown at this operating point.)")
    print("  " + FLOOR_NOTE.replace("\n  ", "\n  "))


def write_er_csv(base, path="experiments/er_sensitivity.csv", targets=None,
                 ers=(1_000, 10_000, 50_000, 100_000, 500_000), window_d=180):
    """E[R] sensitivity as CSV: same per-row detail as write_csv's A1 sweep (feasible flag,
    full-stock-floor fallback, step-time breakdown) but swept over targets x E[R] at one fixed
    window instead of one target x windows."""
    import csv, os
    targets = targets or all_targets()
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        write_config_comment(fh, base)
        w = csv.writer(fh)
        w.writerow(["target", "C_RL", "er_tok", "feasible", "gpus", "pct_stock", "train_frac",
                    "steps", "wall_days", "cost_usd", "bottleneck", "t_step_s", "t_update_s",
                    "t_rollout_s", "t_bcast_s", "window_days"])
        for name, tc in targets.items():
            for er in ers:
                cfg = with_cfg(base, target_c_rl=tc, er=er)
                row = _ev_row(name, tc, er, evaluate(cfg, window_d * 86400))
                w.writerow(row + [window_d])
    print(f"\n  wrote {path}  ({len(targets)} targets x {len(ers)} E[R] points)")


# ---------------------------------------------------------------------------
# CSV writers  (every row carries wall_days + split + feasible, incl. infeasible via the floor)
# ---------------------------------------------------------------------------
def _ev_row(name, tc, d, ev):
    if not ev["runnable"]:
        return [name, f"{tc:.2e}", d, "no-fit", "", "", "", "", "", "", "no-fit", "", "", "", ""]
    r = ev["r"]
    return [name, f"{tc:.2e}", d, int(ev["feasible"]), ev["gpus"], round(ev["pct_stock"], 3),
            round(ev["frac"], 3), round(ev["steps"]), round(ev["wall_d"], 2), round(ev["cost"]),
            ev["bottleneck"], round(r.t_step, 3), round(r.t_update, 3),
            round(r.stages["rollout+verify"], 3), round(r.t_bc, 3)]


def write_csv(base, path="experiments/feasibility_sweep.csv", windows_d=(30, 90, 180, 720), targets=None):
    """A1 sweep as CSV. Infeasible rows carry the full-stock floor (wall_days, split, cost) so the
    'days to expend the FLOPs' + optimal split are ALWAYS present. `feasible` in {1,0,'no-fit'}.
    t_step/t_update/t_rollout/t_bcast are the single-step time breakdown at this operating point
    (raw seconds) -- distinct from wall_days, which is the full run (steps x t_step)."""
    import csv, os
    targets = targets or all_targets()
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        write_config_comment(fh, base)
        w = csv.writer(fh)
        w.writerow(["target", "C_RL", "window_days", "feasible", "gpus", "pct_stock",
                    "train_frac", "steps", "wall_days", "cost_usd", "bottleneck",
                    "t_step_s", "t_update_s", "t_rollout_s", "t_bcast_s",
                    "model", "gpu", "wan_mbps", "compression", "er_tok", "omega", "batch"])
        for name, tc in targets.items():
            cfg = with_cfg(base, target_c_rl=tc)
            batch = f"{cfg.prompts_per_batch}x{cfg.responses_per_prompt}"
            for d in windows_d:
                row = _ev_row(name, tc, d, evaluate(cfg, d * 86400))
                w.writerow(row + [cfg.model.name, cfg.gpu, cfg.wan_mbps, cfg.compression,
                                  cfg.er, cfg.omega, batch])
    print(f"\n  wrote {path}  ({len(targets)} targets x {len(windows_d)} windows)")


def write_batch_compare_csv(base, path="experiments/feasibility_batch_compare.csv",
                            targets=None, windows_d=(30, 90, 180, 720)):
    """Baseline vs high-end batch per target/window; both carry gpus + wall_days + split + feasible."""
    import csv, os
    targets = targets or all_targets()
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        write_config_comment(fh, base)
        w = csv.writer(fh)
        # steps is a property of the batch config + target (not the window), so it gets one
        # column per side rather than one per window -- that's the whole point of this table.
        cols = ["target", "C_RL", "base_steps", "high_steps"]
        for d in windows_d:
            cols += [f"{d}d_base_gpus", f"{d}d_base_wall_d", f"{d}d_base_feas",
                     f"{d}d_high_gpus", f"{d}d_high_wall_d", f"{d}d_high_feas", f"{d}d_flip"]
        w.writerow(cols)
        for name, tc in targets.items():
            cb = with_cfg(base, target_c_rl=tc, **BASELINE_BATCH)
            ch = with_cfg(base, target_c_rl=tc, **HIGH_BATCH)
            row = [name, f"{tc:.2e}", round(n_steps(cb)), round(n_steps(ch))]
            for d in windows_d:
                eb, eh = evaluate(cb, d * 86400), evaluate(ch, d * 86400)
                row += [eb["gpus"], round(eb["wall_d"], 2), int(eb["feasible"]),
                        eh["gpus"], round(eh["wall_d"], 2), int(eh["feasible"]),
                        int(eh["feasible"] and not eb["feasible"])]
            w.writerow(row)
    print(f"  wrote {path}")


def write_models_csv(base, path="experiments/models_feasibility.csv", target_c_rl=2.5e25,
                     windows_d=(90, 180)):
    import csv, os
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        write_config_comment(fh, base)
        w = csv.writer(fh)
        w.writerow(["model", "p_total_B", "active_B", "is_moe", "weights_gb", "fit_node",
                    "C_RL", "steps", "window_days", "feasible", "gpus", "wall_days",
                    "cost_usd", "train_frac", "bottleneck"])
        for name, m in MODELS.items():
            gpu = smallest_fitting_gpu(m, base.gpus_per_node)
            wt = 2.0 * m.p_total / GB
            base_row = [name, m.p_total/1e9, m.p_active_layers/1e9, int(m.is_moe), round(wt)]
            if gpu is None:
                # 10 fields after base_row's 5, to match the 15-column header. Was 9: the
                # bottleneck string landed in the train_frac column and every field after
                # fit_node was shifted one left, so the only no-fit row in this CSV (Kimi-K3)
                # was silently misaligned against its own header.
                w.writerow(base_row + ["none(multi-node)", f"{target_c_rl:.1e}", "", "", 0,
                                       "", "", "", "", "no-single-node-fit"])
                continue
            cfg = with_cfg(base, model=m, gpu=gpu, target_c_rl=target_c_rl)
            steps = round(target_c_rl / per_step_flop(cfg))
            for d in windows_d:
                ev = evaluate(cfg, d * 86400)
                w.writerow(base_row + [gpu, f"{target_c_rl:.1e}", steps, d, int(ev["feasible"]),
                                       ev["gpus"], round(ev["wall_d"], 2), round(ev["cost"]),
                                       round(ev["frac"], 3), ev["bottleneck"]])
    print(f"\n  wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_batch(s):
    """'128x512' -> dict(prompts_per_batch=128, responses_per_prompt=512)."""
    p, _, g = s.lower().partition("x")
    return dict(prompts_per_batch=int(p), responses_per_prompt=int(g))


def build_parser():
    ap = argparse.ArgumentParser(
        description="Distributed-RL feasibility sweeps. Any FeasConfig field can be overridden; "
                    "infeasible scenarios still report the wall-clock floor and optimal split.")
    ap.add_argument("--mode", choices=("default", "thorough", "models"), default="default",
                    help="default = headline tables; thorough = everything + CSVs; models = cross-model")
    ap.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL, help="target model")
    ap.add_argument("--gpu", choices=sorted(GPUS), default=None, help="GPU family")
    ap.add_argument("--er", type=float, default=None, help="mean response length E[R] in tokens")
    ap.add_argument("--er-cv", type=float, default=None, help="coefficient of variation of E[R]")
    ap.add_argument("--omega", type=float, default=None, help="oversampling ratio")
    ap.add_argument("--wan", type=float, default=None, help="WAN bandwidth, Mbps")
    ap.add_argument("--compression", type=float, default=None, help="weight-sync compression factor")
    ap.add_argument("--batch", type=str, default=None, metavar="PxG",
                    help="rollout batch as prompts_per_batch x responses_per_prompt, e.g. 128x512")
    ap.add_argument("--prompt-len", type=int, default=None)
    ap.add_argument("--mfu-train", type=float, default=None)
    ap.add_argument("--mfu-inf", type=float, default=None)
    ap.add_argument("--stock", type=int, default=None, help="GPU stock available")
    ap.add_argument("--gpu-hr-usd", type=float, default=None)
    ap.add_argument("--site-power-mw", type=float, default=None)
    ap.add_argument("--c-rl", type=float, default=2.5e25, help="headline target C_RL (FLOP)")
    ap.add_argument("--targets", type=str, default=None,
                    help="comma-separated subset of target labels (default: all model+threshold)")
    ap.add_argument("--windows", type=str, default="30,90,180,720", help="comma-separated windows in days")
    ap.add_argument("--csv", action=argparse.BooleanOptionalAction, default=None,
                    help="write CSVs (default: on for --mode thorough/models, off otherwise)")
    ap.add_argument("--fine-split", action="store_true",
                    help="search the trainer:inference split at every integer percent (1-99%%) "
                        "instead of the coarse 13-point default -- ~13x more simulate() calls, "
                        "worth it when the split fraction / GPU count is the number you're quoting")
    return ap


def config_from_args(a):
    """FeasConfig with every explicitly-passed override applied (None = keep the default)."""
    cfg = FeasConfig(target_c_rl=a.c_rl, model=MODELS[a.model])
    over = {}
    for arg, fld in (("gpu", "gpu"), ("er", "er"), ("er_cv", "er_cv"), ("omega", "omega"),
                     ("wan", "wan_mbps"), ("compression", "compression"), ("prompt_len", "prompt_len"),
                     ("mfu_train", "mfu_train"), ("mfu_inf", "mfu_inf"), ("stock", "stock_gpus"),
                     ("gpu_hr_usd", "gpu_hr_usd"), ("site_power_mw", "site_power_mw")):
        v = getattr(a, arg)
        if v is not None:
            over[fld] = v
    if a.batch:
        over.update(_parse_batch(a.batch))
    if a.fine_split:
        over["fine_split"] = True
    return with_cfg(cfg, **over) if over else cfg


def main(argv=None):
    a = build_parser().parse_args(argv)
    base = config_from_args(a)
    windows = tuple(int(x) for x in a.windows.split(","))
    every = all_targets()
    targets = ({k: every[k] for k in (t.strip() for t in a.targets.split(",")) if k in every}
               if a.targets else every)
    # Tables about an ENGINEERING lever (batch size, bandwidth, memory bandwidth) frame around
    # real published runs; the "C=1e24"-style rows in all_targets() are governance reporting
    # thresholds, and comparing a lab's batch-size decision against a regulation reads as a
    # category error. Respects --targets, falling back to the full model set if the filter left
    # none (an empty table looks broken, and "you filtered them out" is not worth a silent blank).
    m_targets = {k: v for k, v in targets.items() if k in MODEL_TARGETS} or model_targets()
    # Single-window diagnostics standardise on 180d, matching the dashboard's panels.
    w180 = windows[min(2, len(windows) - 1)]
    want_csv = a.csv if a.csv is not None else (a.mode in ("thorough", "models"))

    if a.mode == "models":
        print("CROSS-MODEL FEASIBILITY (fixed E[R]; smallest GPU node that holds each model's")
        print("weights). Infeasible cells show the full-stock wall-clock floor.")
        print_config_block(base)
        model_sweep(base, target_c_rl=a.c_rl, windows_d=windows[-2:])
        model_sweep(base, target_c_rl=2.5e26, windows_d=windows[-2:])
        max_compute_by_model(base, windows_d=windows[-3:])
        batch_group_table(base, m_targets, window_d=w180)
        batch_compare_table(base, m_targets, windows)
        bandwidth_requirement_table(base, m_targets, window_d=w180)
        if want_csv:
            write_models_csv(base, target_c_rl=a.c_rl, windows_d=windows[-2:])
            write_batch_compare_csv(base, targets=m_targets, windows_d=windows)
        return

    if a.mode == "thorough":
        print(f"THOROUGH ANALYSIS -- {base.model.name}. Infeasible entries report the full-stock")
        print("floor (days to expend the FLOPs) and the optimal trainer:rollout split.")
        print_config_block(base)
        frontier_diagnosis(base)
        # Three largest real model targets (the latency wall is a big-target phenomenon). Was a
        # hardcoded name tuple that included "DeepSeek-R1" -- commented out of MODEL_TARGETS, so
        # the `if k in every` guard silently dropped it and this table quietly showed 2 rows.
        batch_feasibility_table(base, model_targets(top=3), window_d=windows[-1])
        batch_group_table(base, m_targets, window_d=w180)
        batch_compare_table(base, m_targets, windows)
        model_sweep(base, target_c_rl=a.c_rl, windows_d=windows[-2:])
        policy_framing(base)
        max_compute_table(base, windows)
        max_compute_by_stock(base, windows_d=windows)
        feasibility_table(base, targets, windows, title="A1 (model + threshold anchored)")
        gpu_target_matrix(base, targets, window_d=w180)
        export_cap_table(base, a.c_rl, "headline", window_d=w180)
        power_node_table()
        power_envelope_table()
        bandwidth_requirement_table(base, m_targets, window_d=w180)
        hbm_scaling_table(base, m_targets, window_d=w180)
        power_sites_table(base, targets, window_d=windows[min(1, len(windows)-1)])
        centralised_vs_decentralised(base, a.c_rl, windows[min(1, len(windows)-1)])
        sensitivity(base, a.c_rl, windows[min(1, len(windows)-1)])
        er_table(base, targets)
        gpu_family_check(base, a.c_rl, windows[min(1, len(windows)-1)])
        if want_csv:
            write_csv(base, windows_d=windows, targets=targets)
            write_batch_compare_csv(base, targets=m_targets, windows_d=windows)
            write_models_csv(base, target_c_rl=a.c_rl, windows_d=windows[-2:])
            write_max_compute_csv(base, windows_d=windows)
            write_max_compute_by_stock_csv(base, windows_d=windows)
            write_er_csv(base, targets=targets)
        return

    print(f"Distributed-RL feasibility -- {base.model.name} (single-turn; see feasibility_core caveats)")
    print(f"per-step FLOP {per_step_flop(base):.2e} | steps for {a.c_rl:.1e} = {n_steps(base):,.0f}")
    print_config_block(base)
    policy_framing(base)
    max_compute_table(base, windows)
    max_compute_by_stock(base, windows_d=windows)
    feasibility_table(base, targets, windows, title="A1 (model + threshold anchored)")
    batch_compare_table(base, m_targets, windows)
    bandwidth_requirement_table(base, m_targets, window_d=w180)
    hbm_scaling_table(base, m_targets, window_d=w180)
    centralised_vs_decentralised(base, a.c_rl, windows[min(1, len(windows)-1)])
    sensitivity(base, a.c_rl, windows[min(1, len(windows)-1)])
    er_table(base, targets)
    gpu_family_check(base, a.c_rl, windows[min(1, len(windows)-1)])
    if want_csv:
        write_csv(base, windows_d=windows, targets=targets)
        write_max_compute_csv(base, windows_d=windows)
        write_max_compute_by_stock_csv(base, windows_d=windows)
        write_er_csv(base, targets=targets)
    print("\n  CAVEATS: MoE decode native (active-path bandwidth, dense KV). Residual optimism: no")
    print("  training FSDP penalty (A-ii, penalty_para=1); no Rahman eta + broadcast-every-step")
    print("  (A-iii, compression stands in for the DiLoCo sync interval -> WAN bound pessimistic).")


if __name__ == "__main__":
    main()

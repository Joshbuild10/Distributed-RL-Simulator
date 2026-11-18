"""
hardware.py -- GPU and server families, and the physical caps on how a node or site can be built:
the BIS TPP/HBM export cap, per-node and per-site facility power.
"""
# ---------------------------------------------------------------------------
# GPU / server-family presets  (per-GPU bf16 dense peak; HBM GB; HBM BW TB/s; board W)
# ---------------------------------------------------------------------------
# The Blackwell-Ultra parts keep roughly B200/GB200 dense BF16 throughput
# Their improvements are concentrated in FP4 and memory, which this BF16 roofline excludes
# They move results mainly via HBM capacity/bandwidth.
GPUS = {
    "A100":    dict(flops=312e12,  hbm_gb=80,  bw_tbs=2.03, watts=400),
    "H100":    dict(flops=989e12,  hbm_gb=80,  bw_tbs=3.35, watts=700),
    "H800":    dict(flops=989e12,  hbm_gb=80,  bw_tbs=3.35, watts=700),   # H100 compute + HBM; export-cut interconnect
    "H200":    dict(flops=989e12,  hbm_gb=141, bw_tbs=4.80, watts=700),
    "B200":    dict(flops=2250e12, hbm_gb=192, bw_tbs=8.00, watts=1000),
    "B300":    dict(flops=2250e12, hbm_gb=288, bw_tbs=8.00, watts=1100),  # Est: Blackwell Ultra HGX
    "GB200":   dict(flops=2500e12, hbm_gb=186, bw_tbs=8.00, watts=1200),  # Per-GPU in an NVL rack
    "GB300":   dict(flops=2500e12, hbm_gb=288, bw_tbs=8.00, watts=1400),  # Est, per-GPU
#    "MI300X":  dict(flops=1307e12, hbm_gb=192, bw_tbs=5.30, watts=750),   # EST: AMD, see note above
    "MI355X":  dict(flops=2500e12, hbm_gb=288, bw_tbs=8.2, watts=1400),
    "RTX4090": dict(flops=165e12,  hbm_gb=24,  bw_tbs=1.01, watts=450),   # non-flagship / consumer P2P
    "Cerebras-CS4": dict(flops=125e15, hbm_gb=44, bw_tbs=14333.0, watts=43333), # Cerebras CS-4, as reported by semianalysis.
}


# The high-end families, for the GPU-family x target sweeps. Ordered by HBM bandwidth then
# capacity, so a table reading left-to-right tracks the axis that actually binds rollout.
HIGH_END_GPUS = ("A100", "H100", "H200", "B200", "B300", "GB200", "GB300", "MI355X")


GB, TB = 1e9, 1e12


# Facility power multiplier on board TDP: hosts, networking, power conversion and cooling. Every
# power figure in the model is a FACILITY draw, so a "1 MW site" means 1 MW at the meter, not 1 MW
# of GPU board power. Set to 1.0 to quote raw board watts instead.
FACILITY_PUE = 1.6


# Granularity (and minimum) for right-sizing BOTH pools: trainer replica nodes and single inference
# nodes downsize in whole GPU_STEP chunks (16 = a good-topology unit), never to an arbitrary count.
# Bump this if a coarser/finer allocation grain is wanted.
GPU_STEP = 16


def node_hw(gpu, g=8):
    s = GPUS[gpu]
    return dict(node_flops=g * s["flops"], node_hbm=g * s["hbm_gb"] * GB, node_bw=g * s["bw_tbs"] * TB)


# ---------------------------------------------------------------------------
# Export-control node caps (US BIS "Total Processing Performance").
# ---------------------------------------------------------------------------
# TPP = 2 x (non-sparse MacTOPS) x (multiply-input bit length). With 1 MAC = 2 FLOP that collapses
# to flops x bits, so per-chip TPP at BF16 is just flops x 16: an H100 is 15,824 TFLOP-bit/s and the
# "16 H100" cap is 253,184. At a uniform BF16 rating this is numerically a pure FLOP-equivalent cap
# -- TPP only diverges once chips are rated at different precisions (an FP4-heavy part), which this
# BF16 roofline does not track. Read the caps as "N H100-equivalents of dense compute".
TPP_BITS = 16                          # reference precision for the TPP rating (BF16)


def tpp_per_chip(gpu):
    """Per-chip TPP in FLOP-bit/s (BF16 rating). Divide by 1e12 for TFLOP-bit/s."""
    return GPUS[gpu]["flops"] * TPP_BITS


def node_gpus_tpp_cap(gpu, h100_equiv=16, cap_by_memory=True):
    """Max chips per node such that the node does not exceed `h100_equiv` H100s of TPP (compute)
    -- and, when cap_by_memory, also not their HBM capacity (16 H100 = 1,280 GB). The node is the
    tighter of the two, since 'compute/memory of N H100s' means neither may be exceeded. >=1."""
    n_tpp = int(h100_equiv * tpp_per_chip("H100") // tpp_per_chip(gpu))
    if not cap_by_memory:
        return max(1, n_tpp)
    n_hbm = int(h100_equiv * GPUS["H100"]["hbm_gb"] // GPUS[gpu]["hbm_gb"])
    return max(1, min(n_tpp, n_hbm))


def node_gpus_power_cap(gpu, node_kw, pue=FACILITY_PUE):
    """Max chips per node under a per-node FACILITY power budget (kW). Board TDP x pue. >=1."""
    return max(1, int(node_kw * 1000 // (GPUS[gpu]["watts"] * pue)))


def effective_node_gpus(gpus_per_node, stock_gpus):
    """Node size usable at a given fleet size. Disaggregation needs >=2 nodes, so a node is at most
    half the stock -- otherwise a by-stock sweep allocates whole configured-size nodes and reports a
    fleet larger than `stock`. No-op once stock >= 2 x node, so it only reshapes a sweep's small-
    fleet end (where the honest answer is the small-node physics, or no-fit)."""
    return max(1, min(gpus_per_node, stock_gpus // 2))


def sites_power(cfg, gpus, pue=FACILITY_PUE):
    """(sites at the per-site cap, total facility MW). Both sides are facility draw: board TDP x pue."""
    w = GPUS[cfg.gpu]["watts"] * pue
    gpus_per_site = cfg.site_power_mw * 1e6 / w
    return gpus / gpus_per_site, gpus * w / 1e6

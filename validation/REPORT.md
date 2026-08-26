# Validation report — predicted vs published

> **DRAFT, auto-assembled from a run of `python -m validation.validate` on 2026-08-17.**
> Intended to be redrafted by hand. Every number below is reproducible with that command;
> scenario configs are in `validation/presets.py`.

## Headline

| run | model | bound by | model step | published | err |
|---|---|---|---|---|---|
| prime-rl reference | R1-Distill-Qwen-32B, 24×H200 | update | 1,377 s | 1,374 s | **+0%** |
| AReaL-7B | R1-Distill-Qwen-7B, 24 H800 nodes | — | 312 s | 366 s | −15% |
| INTELLECT-2 (short) | QwQ-32B, decentralised swarm | broadcast | 881 s | 1,320 s | −33% |
| INTELLECT-2 (long) | QwQ-32B, decentralised swarm | broadcast | 881 s | 1,260 s | −30% |
| AReaL-1.5B | R1-Distill-Qwen-1.5B, 16 nodes | — | 116 s | 213 s | −46% |
| AReaL-14B | R1-Distill-Qwen-14B, 32 nodes | update | 491 s | 986 s | −50% |
| INTELLECT-3 | GLM-4.5-Air MoE, 512×H200 | update | 445 s | 1,500 s | −70% |
| AReaL-32B | R1-Distill-Qwen-32B, 48 nodes | update | 531 s | 1,866 s | −72% |

Secondary metrics where published: prime-rl trainer **11,261 vs 11,300 tok/s (−0%)**, total
runtime **61.2 h vs 64 h (−4%)**; INTELLECT-2 broadcast volume **65.0 vs 62.0 GB (+5%)**
(difference = embeddings, excluded from theirs).

## Reading this table

**prime-rl is the only fully-pinned anchor and it matches.** Single-turn math (so the model's
single-turn assumption is exactly valid), and step time, trainer throughput, inference
throughput and MFU are all published. Its E[R] is triangulated from its own published
throughputs, and an independent measurement on its actual dataset (Skywork-OR1, Qwen3-32B,
T=1.0) gave 7,536 ± 11% against the 7,470 in use — so the match is not circular. This is the
evidence that the **core FLOP/time accounting is right when E[R] is known**.

The residuals are structured, not noise:

- **Response length was the dominant unknown, and measuring it changed the sign of the story
  per-run.** Measuring E[R] for AReaL-7B (8k → 10.1k) *closed* its gap (−35% → −15%); measuring
  it for AReaL-14B (13k → 10.4k) *widened* the gap (−37% → −50%). Same experiment, opposite
  conclusions — which is itself the finding: small/single-node runs were E[R]-limited, larger
  sharded runs have a real efficiency penalty.
- **The large-shard rows are a trainer-side efficiency gap (A-ii).** AReaL 14B/32B are
  update-bound, and the inverse-solved E[R] needed to match (19.0k / 25.7k) sits at or above the
  measured p99 for those models — i.e. not reachable. With `penalty_para = 1.0` and
  `mfu_train = 0.40`, the model is optimistic about FSDP at high shard degree; the residual grows
  monotonically with shard degree (prime-rl n_shard=8 → +0%; AReaL 64–96 → −50%/−72%).
- **INTELLECT-3 is MoE + multi-turn, not E[R].** Matching 1,500 s needs E[R] ≈ 48.6k, above the
  longest measured single-turn reasoning traces (~24k p99), and our own on-policy measurement of
  GLM-4.5-Air on its real RL data gave 22,961 (cv 0.47). The gap is inference-side MoE/TP
  inefficiency plus multi-turn context accumulation (its SWE and DeepDive environments are
  multi-turn; the model is single-turn — gap A-i). Measured agentic effective context is
  ~29k–123k median depending on environment, which comfortably covers the shortfall.
- **INTELLECT-2 is broadcast-bound and the broadcast is right (+5% on volume).** The step-time
  gap comes from the *swarm*, not the accounting: the implied capacity multiplier is 0.07–0.25,
  i.e. the volunteer pool ran 4–14× weaker than an assumed 8×H100 node. That is a plausible
  property of a heterogeneous donated-hardware swarm rather than a model error, and it recovers
  an inf:train GPU-time ratio of 3.3–8.4× against a published ~4.5×.

## Honest limitations

- **Under-prediction is the systematic direction.** Six of eight rows are negative. For feasibility
  work this biases *toward optimism* (runs look faster/cheaper than reality), so headline
  feasibility numbers should be read with the A-ii/A-iii caveats attached.
- **Only two runs test the inference side.** prime-rl is update-bound, so its step time is
  insensitive to inference peak throughput; only rollout-bound runs (INTELLECT-3, AReaL-14B)
  exercise it, and both have confounds.
- **E[R] is not published by AReaL.** Those four rows are validated by whether the E[R] that
  reproduces the published step is *plausible*, plus direct measurement where we could get it.
- **AReaL inference TP degree is an assumption** (1/1/2/4 for 1.5B/7B/14B/32B), inferred from
  memory fit. Results are stable for TP ≥ 2 but the 14B/32B verdicts are partly TP-assumption
  dependent.

## Reproducing

```bash
python -m validation.validate            # this table + inverse calibration
python -m validation.validate --sweep    # + response-length sensitivity
```

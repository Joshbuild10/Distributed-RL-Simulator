"""
models.py -- The model registry and which GPU family can hold each model on one node.
"""
from specs import ModelSpec
from .hardware import node_hw

# ---------------------------------------------------------------------------
# Model list for the cross-model feasibility sweep + the default target model.
# CAVEAT: Kimi-K3 internals (d_model/layers/heads/vocab) are ESTIMATES (post-cutoff, DeepSeek-V3/Kimi-K2-scaled)
# Its p_total/active (2.4T/104B) are given. Attention assumed MLA with Kimi-K2's latent dims (K2 is MLA).
# ---------------------------------------------------------------------------
MODELS = {
    "Llama3-8B":       ModelSpec(name="Llama-3.1-8B", is_moe=False, p_total=8.03e9, p_active=7.0e9,
                                 d_model=4096, n_layers=32, n_q_heads=32, n_kv_heads=8, head_dim=128, vocab=128256),
    "Llama3-70B":      ModelSpec(name="Llama-3.1-70B", is_moe=False, p_total=70.6e9, p_active=68.5e9,
                                 d_model=8192, n_layers=80, n_q_heads=64, n_kv_heads=8, head_dim=128, vocab=128256),
    "Llama3-405B":     ModelSpec(name="Llama-3.1-405B", is_moe=False, p_total=405e9, p_active=401e9,
                                 d_model=16384, n_layers=126, n_q_heads=128, n_kv_heads=8, head_dim=128, vocab=128256),
    "Qwen3-30B-A3B":   ModelSpec(name="Qwen3-30B-A3B (MoE)", is_moe=True, p_total=30.5e9, p_active=3.3e9,
                                 d_model=2048, n_layers=48, n_q_heads=32, n_kv_heads=4, head_dim=128, vocab=151936),
    "Qwen3-235B-A22B": ModelSpec(name="Qwen3-235B-A22B (MoE)", is_moe=True, p_total=235e9, p_active=22e9,
                                 d_model=4096, n_layers=94, n_q_heads=64, n_kv_heads=4, head_dim=128, vocab=151936),
    "Kimi-K3":         ModelSpec(name="Kimi-K3 (MoE, MLA, EST arch)", is_moe=True, p_total=2.4e12, p_active=104e9,
                                 d_model=7168, n_layers=90, n_q_heads=128, n_kv_heads=128, head_dim=128, vocab=163840,
                                 attention="mla", kv_lora_rank=512, qk_rope_dim=64, qk_nope_dim=128, v_head_dim=128),
    # DeepSeek-V3 / R1 (HF config.json). 37B activated less the embedding and head (2 x 7168 x 129,280).
    "DeepSeek-V3":     ModelSpec(name="DeepSeek-V3/R1 (MoE, MLA)", is_moe=True, p_total=671e9, p_active=35.1e9,
                                 d_model=7168, n_layers=61, n_q_heads=128, n_kv_heads=128, head_dim=128, vocab=129280,
                                 attention="mla", kv_lora_rank=512, qk_rope_dim=64, qk_nope_dim=128, v_head_dim=128),
    "Llama4-402B-A17B": ModelSpec(name="Llama4-402B-A17B", is_moe=True, p_total=400e9, p_active=17e9,
                                     d_model=5120, n_layers=48, n_q_heads=40, n_kv_heads=8, head_dim=128, vocab=202048),
}


# DEFAULT feasibility model: dense Llama-3.1-405B -- open weights, frontier-scale, and the arch
# every published figure in the writeup is quoted at. Switch to Qwen3-235B-A22B for the MoE path
# (an open MoE with a measured E[R]).
DEFAULT_MODEL = "Llama3-405B"


def smallest_fitting_gpu(m, gpus_per_node, b_weights=2.0, families=("H100", "H200", "B200")):
    """Smallest-HBM family whose node holds the inference weights (b_weights bytes/param: pass
    cfg.b_weights_inf); None -> multi-node inference TP, which this model does not represent."""
    fit = [(g, node_hw(g, gpus_per_node)["node_hbm"]) for g in families]
    fit = [(g, hbm) for g, hbm in fit if hbm >= b_weights * m.p_total]
    return min(fit, key=lambda x: x[1])[0] if fit else None

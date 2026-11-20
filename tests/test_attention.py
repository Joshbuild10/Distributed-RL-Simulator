"""Attention variants: MLA matches the R1-Zero compute derivation, and MHA/GQA is unchanged."""
from dataclasses import replace

from feasibility import MODELS


def test_mla_constants_match_the_derivation():
    m = MODELS["DeepSeek-V3"]                                  # L=61, H=128, r_kv=512, D_r=64, D_k=D_v=128
    assert 4 * m.attn_coef_decode == 16_990_208                # C_a = 2 L H (2 r_kv + D_r), absorbed decode
    assert 4 * m.attn_coef == 4_997_120                        # C_n = 2 L H (D_k + D_v + D_r), naive training
    assert m.kv_elems_per_tok == 61 * (512 + 64)               # latent + RoPE key per token per layer
    assert m.kv_head_shards == 1                               # shared latent: no split by head


def test_mha_unchanged():
    for m in MODELS.values():
        if m.attention == "mha":
            assert m.attn_coef == m.attn_coef_decode == m.n_q_heads * m.head_dim * m.n_layers
            assert m.kv_elems_per_tok == 2 * m.n_layers * m.n_kv_heads * m.head_dim


def test_mla_cache_is_smaller_than_gqa():
    m = MODELS["DeepSeek-V3"]
    assert m.kv_elems_per_tok < replace(m, attention="mha", n_kv_heads=8).kv_elems_per_tok

from dataclasses import dataclass
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
# Data about the Model being trained
class ModelSpec:
    is_moe: bool            # Uses mixture of experts (MoE) architecture or not
    p_total: float          # Total model parameters
    p_active: float  # Active params in decoder layers only (for compute)
    d_model: int            # Hidden dimension of the model
    n_layers: int           # Number of decoder layers in the transformer
    n_q_heads: int          # Number of heads for querying attention
    n_kv_heads: int         # Number of heads for KV values in attention
    head_dim: int           # Hidden dimension of the attention heads
    vocab: int              # Model vocabulary size
    name: str = ""                  # (optional) name of model being trained
    tied_embeddings: bool = False   # Tied embeddings mean a shared embedding/unembedding matrix. Reducing params

    @property
    def p_head(self) -> float:
        """Unembedding (LM head) params."""
        return self.d_model * self.vocab

    @property
    def p_embed(self) -> float: # Number of params in 
        """Embedding table params (+ head if untied)."""
        n = 1 if self.tied_embeddings else 2
        return n * self.d_model * self.vocab

    @property
    def attn_coef(self) -> float:
        """N*H*L -- the per-(query,key) attention FLOP coefficient, without the leading 2/4 coefficient"""
        return self.n_q_heads * self.head_dim * self.n_layers


    def attn_significance_T(self, fraction: float = 0.20) -> float:
        """
        Context length where dot-product attention's FLOPs make up a significant `fraction` of the layer's
        FLOPs.
        Derived on the SAME basis the actual c_attn term uses (attn_coef = N*H*L, not D*L):
        per token, attn ~ causal_coef*T*attn_coef and layers ~ 2*P_active_layers, so with the
        causal coefficient 2 the ratio is T*attn_coef/P_active_layers  =>  T = frac*P/attn_coef.
        Note this is much SMALLER for MoE (small P_active) at the same N, H, L.
        """
        return fraction * self.p_active / self.attn_coef


@dataclass
# Data about the Reinforcement Learning (RL) run being performed
class RLSpec:
    prompts_per_batch: int
    responses_per_prompt: int           # group size G
    prompt_len: int                     # mean prefill length
    response_len_mean: float            # E[R]. Expected response length
    response_len_cv: float = 0.0        # coefficient of variation, for E[R^2]
    max_response_len: Optional[int] = None
    n_prompts_total: Optional[int] = None
    n_epochs: int = 1
    n_steps_override: Optional[int] = None
    prefix_caching: bool = True           # Share the prefill cache across the G responses
    # Oversampling 
    success_rate: Optional[float] = None        # probability of task success. If given, it's used to derive the oversampling ratio
    oversample_override: Optional[float] = 1    # Manual input for the oversampling ratio

    def __post_init__(self):
        # A batch of 0 rollouts is not a valid RL config (there's no data to train on), and
        # left unchecked it causes n_waves = ceil(0/x) = 0 a few layers down in model.py, which
        # turns into a 0/0 ZeroDivisionError deep inside rollout_terms() -- fail here instead,
        # at construction, with a message that actually points at the problem.
        if self.prompts_per_batch <= 0:
            raise ValueError(f"prompts_per_batch must be >= 1, got {self.prompts_per_batch}")
        if self.responses_per_prompt <= 0:
            raise ValueError(f"responses_per_prompt must be >= 1, got {self.responses_per_prompt}")

    @property
    def size_batch(self) -> int:
        return self.prompts_per_batch * self.responses_per_prompt

    @property
    def context_len(self) -> float:
        return self.prompt_len + self.response_len_mean

    @property
    def er2(self) -> float:
        """E[R^2] = Var + E[R]^2, needed by the quadratic (attention / KV) terms."""
        var = (self.response_len_cv * self.response_len_mean) ** 2
        return var + self.response_len_mean ** 2

    def oversample_ratio(self) -> float:
        """Oversampling ratio.
        When a success rate is given, it's derived from zero-advantage filtering.
        A group of G binary-reward samples is discarded iff all G agree,
        so the fraction of accepted samples is f = 1 - [p^G + (1-p)^G] and oversample_ratio = 1/f.

        PRECEDENCE IS DELIBERATE: `oversample_override` wins over `success_rate`, and it defaults
        to 1 -- so the derived path only runs when the override is explicitly set to None. Real
        per-dataset success rates are hard to obtain and vary a lot, so a measured/calibrated omega
        is preferred to a derived one. Caveat if you do use the derived path: f is strongly
        non-linear and real RL datasets are BIMODAL in difficulty, so evaluating f at a single mean
        p understates omega badly -- the correct quantity is omega = 1/E_p[f(p)] over the prompt
        difficulty distribution (measured: 1.5-4.7x vs ~1.0 from the single-p form)."""
        if self.oversample_override is not None:
            return self.oversample_override
        if self.success_rate is None:
            return 1.0
        p, G = self.success_rate, self.responses_per_prompt
        f = 1.0 - (p ** G + (1.0 - p) ** G)
        return 1.0 / max(f, 1e-4) # Max caps oversampling at 10000


@dataclass
# Specifications for the training run's algorithm settings
class AlgoSpec:
    # Precision, in bytes per element
    b_weights_train: float = 2.0        # BF16 params on trainer
    b_grads: float = 2.0                
    b_optimiser: float = 8.0            # Adam momentum+variance fp32. Muon is lighter (see presets)
    b_weights_inf: float = 2.0          # Inference weights precision
    b_kv: float = 2.0                   # KV cache precision
    
    # Throughput multipliers relative to a bf16 dense peak (when other computational precisions are used)
    tput_mult_train: float = 1.0
    tput_mult_inf: float = 1.0
    
    # recomputation
    recomp_act: int = 0                 # 1 = Full activation checkpointing
    recomp_old: int = 0                 # 1 = Recomputation of old-policy logprobs on the trainer
    
    # loop structure
    opt_steps: int = 1                      # optimizer (minibatch) steps per rollout batch
    seqs_per_micro_per_gpu: float = 1.0     # gradient-accumulation micro-batch: this sets activation memory
    act_site_parallel: bool = False         # True: activations sequence/tensor-parallel sharded across the site
                                            #   (per-node micro-batch, independent of node size). False (default):
                                            #   data-parallel, one micro-batch per GPU (validation-calibrated path).
    
    compression_ratio: float = 1.0      # Weight compression factor
    sync_interval: float = 1.0          # Off-policy staleness degree k. Affects how stages and the per-step broadcast time:
                                        #   k=0  ON-POLICY: Rollout, update and broadcast run serialy, so t_step = sum(stages); broadcast full.
                                        #   k=1  One-step OFF-POLICY. Update and broadcast overlap generation. So t_step = max(stages); broadcast full        hidden behind the step.
                                        #   k>=2 k-step off-policy: weights are broadcast once every k steps, so the per-step broadcast cost is 1/k of a full sync.
    include_attention: bool = True      # Add the compute for dot-product attention terms
    causal_attention: bool = True      # Whether the LM uses causal self attention or not to reduce FLOPs
    #   "peak"     -- KV cache allocated for a max-context sequence (giving a conservative lower bound on concurrency)
    #   "expected" -- KV cache allocated for the mean in-flight footprint P + E[R]/2 (optimistic)
    kv_provisioning: str = "peak"

    @property
    def causal_coef(self) -> float:
        """Leading coefficient on self-attention FLOPs: 2 under causal masking
        (only ~half of query,key pairs are computed), 4 otherwise."""
        return 2.0 if self.causal_attention else 4.0


@dataclass
# Cluster hardware specifications, assumed homogeneity of hardware.
class HWSpec:
    label: str                          # Name of hardware cluster
    n_nodes: int                        # Number of nodes in network
    node_flops: float                   # BF16 FLOP/s per node
    node_hbm: float                     # bytes per node
    gpus_per_node: int = 8              # Number of GPUs per node
    node_bw: float = 0.0                # bytes/s HBM bandwidth per node
    mfu: float = 0.40                   # Baseline compute-bound utilisation
    bw_eff: float = 0.85                # Achievable fraction of peak reported bandwidth
    penalty_para: float = 1.0           # Throughput penalty under model sharding (<=1)
    n_shard: int = 1                    # ZeRO-3 sharding degree (for trainer nodes)
    capacity_mult: float = 1.0          # Scales down node FLOPs AND bandwidth to represent heterogeneous / weaker hardware

    @property
    def flops(self) -> float:
        return self.node_flops * self.capacity_mult

    @property
    def bw(self) -> float:
        return self.node_bw * self.capacity_mult


@dataclass
# Specifies the distributed network's upload/download bandwidth (and topology) 
class NetSpec:
    bandwidth: float                   # bytes/s for the weight broadcast
    latency: float = 0.0               # seconds of fixed overhead


@dataclass
# Specifies detail about prompt and reward verification times
class VerifySpec:
    mode: str = "rule"                  # Choice between "rule" vs "model" based verification
    seconds_per_rollout: float = 0.0    # rule-based, per rollout
    n_verifier_workers: int = 1         

    p_verifier: float = 0.0            # model-based verifier params
    verifier_flops: float = 0.0
    verifier_mfu: float = 0.4
    t_env_per_rollout: float = 0.0     # Environment latency per rollout


@dataclass
# Data stored for a scenario analysis
class Scenario:
    name: str
    model: ModelSpec
    rl: RLSpec
    algo: AlgoSpec
    train_hw: HWSpec
    inf_hw: HWSpec
    net: NetSpec
    verify: VerifySpec
    notes: str = ""
    published: Optional[Dict[str, Any]] = None
from dataclasses import dataclass

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Gemma3nTextArgs(BaseModelArgs):
    # Core transformer dimensions
    dim: int = 2304  # hidden_size
    n_layers: int = 26  # num_hidden_layers
    n_heads: int = 8  # num_attention_heads
    n_kv_heads: int = 4  # num_key_value_heads
    head_dim: int = 256
    vocab_size: int = 262208
    intermediate_size: int = 9216  # MLP dimension

    # Normalization/initialization
    norm_eps: float = 1e-6  # rms_norm_eps
    initializer_range: float = 0.02

    # Positional embeddings (RoPE)
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    rope_local_base_freq: float = 10_000.0
    max_seq_len: int = 32768  # Also used as max_position_embeddings

    # Attention settings
    attn_bias: bool = False  # attention_bias
    attention_dropout: float = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int = 4096
    sliding_window_pattern: int = 6
    layer_types: list[str] | None = None
    num_kv_shared_layers: int = 0

    # Activations / logits softcapping
    hidden_activation: str = "gelu_pytorch_tanh"
    activation_sparsity_pattern: list[float] | None = None  # per-layer sparsity value, 0.0 disables
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None

    # Caching and embeddings
    cache_implementation: str = "hybrid"  # aligns with Gemma2/3 caching style
    tie_word_embeddings: bool = True

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2

    # AltUp / Laurel / Per-layer input settings
    laurel_rank: int = 256
    altup_num_inputs: int = 2
    altup_active_idx: int = 0
    altup_coef_clip: float | None = None
    altup_correct_scale: bool = False
    hidden_size_per_layer_input: int = 1024
    vocab_size_per_layer_input: int = 64000

    # Trainer-facing/general args used across torchtitan models
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

        if (
            job_config.parallelism.pipeline_parallel_degree > 1
            and self.use_flex_attn
            and self.attn_mask_type == "block_causal"
        ):
            raise RuntimeError(
                "PP + block causal FlexAttention support will be fixed soon."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # Parameter counting (dense only; no MoE here)
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = 0
        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()

        # FLOPs approximation consistent with other transformer args in this repo
        num_layers, num_heads, head_hidden_dim, num_tokens = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        num_flops_per_token = (
            6 * (nparams - nparams_embedding) + 12 * num_layers * num_heads * head_hidden_dim * num_tokens
        )
        return nparams, num_flops_per_token
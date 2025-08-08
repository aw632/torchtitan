import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols import ModelProtocol

from .args import Gemma3nTextArgs


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    # Matches HF's gelu_pytorch_tanh implementation
    return F.gelu(x, approximate="tanh")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gemma3n uses fp32 compute for stability
        x_float = x.float()
        denom = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_float / denom
        out = out * self.weight.float()
        return out.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Simple RoPE with configurable base theta. Used for both global and local RoPE.

    This intentionally mirrors the math used by HF Gemma3n default RoPE path.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, num_heads, head_dim) used only for dtype/device
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # force float32
            t = position_ids.to(dtype=torch.float32)  # (B, S)
            freqs = torch.einsum("bs,d->bsd", t, self.inv_freq)  # (B, S, head_dim//2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, head_dim)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    # Broadcast cos/sin to match x
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seqlen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seqlen, head_dim)


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


def build_sliding_window_mask(seq_len: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # mask shape (1, 1, q_len, k_len) with -inf for disallowed positions
    idx = torch.arange(seq_len, device=device)
    q = idx.unsqueeze(1)
    k = idx.unsqueeze(0)
    # allow keys where 0 <= k <= q and (q - k) < window
    allowed = (k <= q) & ((q - k) < window)
    mask = torch.where(allowed, torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(float("-inf"), dtype=dtype, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class Gemma3nAttention(nn.Module):
    def __init__(self, args: Gemma3nTextArgs, layer_idx: int, is_sliding: bool):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding

        self.num_heads = args.n_heads
        self.num_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(args.dim, self.num_heads * self.head_dim, bias=args.attn_bias)
        self.k_proj = nn.Linear(args.dim, self.num_kv_heads * self.head_dim, bias=args.attn_bias)
        self.v_proj = nn.Linear(args.dim, self.num_kv_heads * self.head_dim, bias=args.attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.dim, bias=args.attn_bias)

        # Per-head RMSNorms (Gemma3n specific)
        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=args.norm_eps, with_scale=False)

        self.attn_dropout = args.attention_dropout
        self.attn_logit_softcap = args.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, S, D)
        pos_emb: tuple[torch.Tensor, torch.Tensor],  # (cos, sin) each (B, S, head_dim)
        attention_mask: Optional[torch.Tensor],  # (1, 1, S, S)
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        cos, sin = pos_emb
        # transpose to (B, H, S, D) for attention computation
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        v = v.transpose(1, 2)

        # repeat kv heads to match num_heads
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # scaled dot-product attention (Gemma3n uses q_norm/k_norm so we keep scaling=1.0)
        attn_scores = torch.matmul(q, k.transpose(2, 3))
        if self.attn_logit_softcap is not None:
            cap = torch.tensor(self.attn_logit_softcap, dtype=attn_scores.dtype, device=attn_scores.device)
            attn_scores = torch.tanh(attn_scores / cap) * cap

        if attention_mask is not None:
            # slice in case K is shorter when using cache (not used here but safe)
            attn_scores = attn_scores + attention_mask[..., : attn_scores.shape[-2], : attn_scores.shape[-1]]

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        out = torch.matmul(attn_weights, v)  # (B, H, S, D)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Gemma3nMLP(nn.Module):
    def __init__(self, args: Gemma3nTextArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.dim
        self.intermediate_size = args.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # activation
        if args.hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = gelu_pytorch_tanh
        else:
            # fallback to GELU approximate tanh since Gemma3n primarily uses it
            self.act_fn = gelu_pytorch_tanh

        # activation sparsity pattern per layer if provided
        self.activation_sparsity = 0.0
        if args.activation_sparsity_pattern is not None and 0 <= layer_idx < len(args.activation_sparsity_pattern):
            self.activation_sparsity = float(args.activation_sparsity_pattern[layer_idx])

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        # Match HF implementation using Normal icdf thresholding
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return F.relu(inputs - cutoff_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.activation_sparsity and self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)


class Gemma3nTextLaurelBlock(nn.Module):
    def __init__(self, args: Gemma3nTextArgs):
        super().__init__()
        self.linear_left = nn.Linear(args.dim, args.laurel_rank, bias=False)
        self.linear_right = nn.Linear(args.laurel_rank, args.dim, bias=False)
        self.post_laurel_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states


class Gemma3nTextAltUp(nn.Module):
    def __init__(self, args: Gemma3nTextArgs):
        super().__init__()
        self.args = args
        self.correct_output_scale = nn.Parameter(torch.zeros(args.dim))
        self.correction_coefs = nn.Linear(args.altup_num_inputs, args.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(args.altup_num_inputs, args.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(args.dim, args.altup_num_inputs, bias=False)
        self.router_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(args.dim**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [num_altup_inputs, B, T, D]
        modalities = self.compute_router_modalities(hidden_states[self.args.altup_active_idx])
        if self.training and self.args.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.args.altup_coef_clip, self.args.altup_coef_clip)
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.args.altup_num_inputs, self.args.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)
        predictions += hidden_states
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        # predictions: [num_altup_inputs, B, T, D]; activated: [B, T, D]
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.args.altup_active_idx]
        innovation = innovation.repeat(self.args.altup_num_inputs, 1, 1, 1)
        if self.args.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.args.altup_coef_clip, self.args.altup_coef_clip)
        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)
        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions
        return corrected.contiguous().type_as(activated)

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return self.forward(corrected)


class Gemma3nDecoderLayer(nn.Module):
    def __init__(self, args: Gemma3nTextArgs, layer_idx: int, is_sliding: bool):
        super().__init__()
        self.args = args
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = Gemma3nAttention(args, layer_idx=layer_idx, is_sliding=is_sliding)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = Gemma3nMLP(args, layer_idx=layer_idx)
        self.pre_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

        # AltUp / Laurel
        self.altup = Gemma3nTextAltUp(args)
        self.laurel = Gemma3nTextLaurelBlock(args)

        # Per-layer input gate / projection
        self.hidden_size_per_layer_input = args.hidden_size_per_layer_input
        self.act_fn = gelu_pytorch_tanh
        self.per_layer_input_gate = nn.Linear(args.dim, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, args.dim, bias=False)
        self.post_per_layer_input_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [P, B, T, D]
        pos_emb_global: tuple[torch.Tensor, torch.Tensor],
        pos_emb_local: tuple[torch.Tensor, torch.Tensor],
        attention_mask_full: torch.Tensor,
        attention_mask_sliding: torch.Tensor,
        is_sliding: bool,
        per_layer_input: torch.Tensor,  # [B, T, P_l]
        per_layer_input_scale: torch.Tensor,  # scalar tensor
        altup_correct_scale: bool,
        active_idx: int,
    ) -> torch.Tensor:
        # AltUp predict
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[active_idx]

        # Input layer norm + Laurel
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # Attention
        pos_emb = pos_emb_local if is_sliding else pos_emb_global
        attn_mask = attention_mask_sliding if is_sliding else attention_mask_full
        attn = self.self_attn(active_prediction_normed, pos_emb, attn_mask)
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2.0)

        # Feedforward (pre -> mlp -> post -> residual)
        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        # AltUp correct
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)
        first_prediction = corrected_predictions[active_idx].clone()
        if altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # Per-layer input pathway
        gated = self.per_layer_input_gate(first_prediction)
        gated = self.act_fn(gated)
        gated = gated * per_layer_input  # [B, T, P_l]
        projected = self.per_layer_projection(gated)
        projected = self.post_per_layer_input_norm(projected)
        corrected_predictions[1:] += projected

        return corrected_predictions


class Gemma3nTextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3nTextModel(nn.Module, ModelProtocol):
    """
    A faithful single-device implementation of Gemma 3n text-only decoder model, following
    the reference architecture. This covers token embedding, per-head RMSNorm q/k/v attention
    with optional sliding window, MLP with gelu_pytorch_tanh, and final RMSNorm.
    """

    def __init__(self, model_args: Gemma3nTextArgs):
        super().__init__()
        self.args = model_args

        if self.args.dim % self.args.n_heads != 0:
            raise ValueError("args.dim must be divisible by args.n_heads")

        if self.args.head_dim * self.args.n_heads != self.args.dim:
            raise ValueError("head_dim * n_heads must equal dim for Gemma3n")

        self.vocab_size = self.args.vocab_size
        self.n_layers = self.args.n_layers

        self.tok_embeddings = Gemma3nTextScaledWordEmbedding(
            self.args.vocab_size, self.args.dim, self.args.pad_token_id, embed_scale=self.args.dim**0.5
        )

        # Precompute RoPE embeddings: global and local
        self.register_buffer(
            "freqs_cis_dummy",
            torch.ones(1),
            persistent=False,
        )  # placeholder to track buffer device

        self.rotary_global = RotaryEmbedding(
            head_dim=self.args.head_dim,
            max_position_embeddings=self.args.max_seq_len,
            theta=float(self.args.rope_theta),
        )
        self.rotary_local = RotaryEmbedding(
            head_dim=self.args.head_dim,
            max_position_embeddings=self.args.max_seq_len,
            theta=float(self.args.rope_local_base_freq),
        )

        # Layer types: decide sliding vs full
        layer_types = self.args.layer_types
        if layer_types is None or len(layer_types) == 0:
            # default pattern: every k-th layer uses sliding attention
            pattern = max(1, int(self.args.sliding_window_pattern))
            layer_types = ["sliding_attention" if (i % pattern == 0) else "full_attention" for i in range(self.n_layers)]
        if len(layer_types) != self.n_layers:
            raise ValueError("len(layer_types) must equal n_layers")
        self.layer_is_sliding = [t == "sliding_attention" for t in layer_types]

        self.layers = nn.ModuleList(
            [
                Gemma3nDecoderLayer(self.args, layer_idx=i, is_sliding=self.layer_is_sliding[i])
                for i in range(self.n_layers)
            ]
        )

        # Per-layer input embeddings/projections
        self.hidden_size = self.args.dim
        self.hidden_size_per_layer_input = self.args.hidden_size_per_layer_input
        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            self.args.vocab_size_per_layer_input,
            self.args.n_layers * self.hidden_size_per_layer_input,
            self.args.pad_token_id,
            embed_scale=self.hidden_size_per_layer_input**0.5,
        )
        self.per_layer_model_projection = nn.Linear(
            self.hidden_size, self.args.n_layers * self.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection_norm = RMSNorm(self.hidden_size_per_layer_input, eps=self.args.norm_eps)
        self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)

        # Projections for AltUp inputs
        self.altup_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.args.altup_num_inputs)]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.args.altup_num_inputs)]
        )

        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)

        if self.args.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        self.init_weights()

    def init_weights(self, buffer_device: Optional[torch.device] = None):
        # Initialize parameters similar to other models in the repo
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=self.args.initializer_range)

        for layer in self.layers:
            # Attention projections
            for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                nn.init.trunc_normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            nn.init.trunc_normal_(layer.self_attn.o_proj.weight, mean=0.0, std=self._final_proj_std())
            if layer.self_attn.o_proj.bias is not None:
                nn.init.zeros_(layer.self_attn.o_proj.bias)

            # Norms
            for norm in (layer.input_layernorm, layer.post_attention_layernorm, layer.post_ffn_layernorm,
                         layer.self_attn.q_norm, layer.self_attn.k_norm, layer.self_attn.v_norm):
                # RMSNorm default init already sets weight ones
                pass

            # MLP
            nn.init.trunc_normal_(layer.mlp.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(layer.mlp.up_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(layer.mlp.down_proj.weight, mean=0.0, std=self._final_proj_std())

        if self.norm is not None:
            # RMSNorm default reset sufficient
            pass

        if self.output is not None and not self.args.tie_word_embeddings:
            final_out_std = self.args.dim ** -0.5
            cutoff_factor = 3
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        # Initialize per-layer and AltUp projections
        nn.init.trunc_normal_(self.per_layer_model_projection.weight, mean=0.0, std=0.02)
        for proj in self.altup_projections:
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
        for proj in self.altup_unembed_projections:
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)

    def _final_proj_std(self) -> float:
        # Match depth-scaled init like other transformers in repo if requested
        if self.args.depth_init:
            return 0.02 / math.sqrt(2 * (self.n_layers))
        return 0.02 / math.sqrt(2 * self.n_layers)

    def _prepare_attention_masks(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        full = build_causal_mask(seqlen, device=device, dtype=dtype)
        sliding = build_sliding_window_mask(seqlen, window=int(self.args.sliding_window), device=device, dtype=dtype)
        return full, sliding

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embs = self.embed_tokens_per_layer(input_ids)
        return embs.view(*input_ids.shape, self.args.n_layers, self.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor, per_layer_inputs: Optional[torch.Tensor]) -> torch.Tensor:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.to(dtype=inputs_embeds.dtype, device=per_layer_projection.device)
        per_layer_projection = per_layer_projection.view(
            *inputs_embeds.shape[:-1], self.args.n_layers, self.hidden_size_per_layer_input
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.args.n_layers, :]
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        eos_id: int | None = None,
        input_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must be (batch, seq_len)")

        inputs_embeds = self.tok_embeddings(tokens)

        bsz, seqlen, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        # position ids [B, S]
        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)

        # RoPE cos/sin
        pos_emb_global = self.rotary_global(inputs_embeds, pos_ids)
        pos_emb_local = self.rotary_local(inputs_embeds, pos_ids)

        # Attention masks
        attn_mask_full, attn_mask_sliding = self._prepare_attention_masks(seqlen, device, dtype)

        # Prepare per-layer inputs
        per_layer_inputs = self.get_per_layer_inputs(tokens)
        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        # Prepare AltUp stacked inputs
        hidden_states_0 = inputs_embeds
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.args.altup_num_inputs):
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=inputs_embeds.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)  # [P, B, T, D]

        # decoder layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                pos_emb_global,
                pos_emb_local,
                attn_mask_full,
                attn_mask_sliding,
                is_sliding=self.layer_is_sliding[i],
                per_layer_input=per_layer_inputs[:, :, i, :],
                per_layer_input_scale=self.per_layer_input_scale,
                altup_correct_scale=self.args.altup_correct_scale,
                active_idx=self.args.altup_active_idx,
            )

        # Combine AltUp inputs back to single stream
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.args.altup_num_inputs):
            altup_unemb_proj = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb_proj.to(dtype=inputs_embeds.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.args.final_logit_softcapping is not None:
            cap = float(self.args.final_logit_softcapping)
            logits = torch.tanh(logits / cap) * cap

        return logits




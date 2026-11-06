# coding=utf-8
# Copyright 2026 MatterWave AI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch GRT (Geometric Rotational Transformer) model for NXD inference.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

try:
    from neuronx_distributed.utils import cpu_mode
    from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
    from neuronx_distributed_inference.models.model_base import (
        NeuronBaseForCausalLM,
        NeuronBaseModel,
    )
    from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

    _HAS_NXD = True
except ImportError:
    _HAS_NXD = False

    def cpu_mode() -> bool:
        return True

    CustomRMSNorm = LlamaRMSNorm

    class NeuronConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs) -> None:
            self.batch_size = kwargs.pop("batch_size", 1)
            self.padding_side = kwargs.pop("padding_side", "right")
            self.seq_len = kwargs.pop("seq_len", 128)
            self.n_active_tokens = kwargs.pop("n_active_tokens", self.seq_len)
            self.n_positions = kwargs.pop("n_positions", self.seq_len)
            self.max_context_length = kwargs.pop("max_context_length", self.seq_len)
            self.max_length = kwargs.pop("max_length", self.seq_len)
            self.max_batch_size = kwargs.pop("max_batch_size", self.batch_size)
            self.buckets = kwargs.pop("buckets", [self.seq_len])
            self.tp_degree = kwargs.pop("tp_degree", 1)
            self.pp_degree = kwargs.pop("pp_degree", 1)
            self.ep_degree = kwargs.pop("ep_degree", 1)
            self.local_ranks_size = kwargs.pop("local_ranks_size", self.tp_degree)
            self.vocab_parallel = kwargs.pop("vocab_parallel", False)
            self.sequence_parallel_enabled = kwargs.pop("sequence_parallel_enabled", False)
            self.fused_qkv = kwargs.pop("fused_qkv", False)
            self.on_device_sampling_config = kwargs.pop("on_device_sampling_config", None)
            self.torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
            if isinstance(self.torch_dtype, str):
                self.torch_dtype = getattr(torch, self.torch_dtype)
            self.on_cpu = kwargs.pop("on_cpu", False)
            for k, v in kwargs.items():
                setattr(self, k, v)

    class InferenceConfig:  # type: ignore[no-redef]
        attribute_map: Dict[str, str] = {}

        def __init__(
            self,
            neuron_config: NeuronConfig,
            fused_spec_config=None,
            load_config=None,
            metadata: Optional[Dict] = None,
            **kwargs,
        ):
            self.neuron_config = neuron_config
            self.fused_spec_config = fused_spec_config
            if load_config is not None:
                load_config(self)
            else:
                self.load_config()
            self.metadata = metadata
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.add_derived_config()
            self.validate_config()

        def add_derived_config(self):
            self.num_cores_per_group = 1

        def load_config(self):
            pass

        def get_required_attributes(self) -> List[str]:
            return []

        def validate_config(self):
            missing_attributes = [
                x for x in self.get_required_attributes() if not hasattr(self, x)
            ]
            assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

        def to_json_string(self) -> str:
            def _ser(obj):
                if isinstance(obj, dict):
                    return {k: _ser(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_ser(v) for v in obj]
                if isinstance(obj, torch.dtype):
                    return str(obj).split(".")[1]
                if isinstance(obj, type):
                    return {"__module__": obj.__module__, "__name__": obj.__name__}
                if hasattr(obj, "__dict__"):
                    return {k: _ser(v) for k, v in obj.__dict__.items()}
                return obj

            return json.dumps(_ser(self), indent=2, sort_keys=True)

        def save(self, model_path: Union[str, os.PathLike]):
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, "neuron_config.json"), "w", encoding="utf-8") as f:
                f.write(self.to_json_string() + "\n")

        @classmethod
        def load(cls, model_path: Union[str, os.PathLike], **kwargs) -> "InferenceConfig":
            with open(os.path.join(model_path, "neuron_config.json"), "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            data.update(kwargs)
            nc_dict = data.pop("neuron_config", {})
            nc = (
                cls.get_neuron_config_cls()(**nc_dict)
                if isinstance(nc_dict, dict)
                else nc_dict
            )
            return cls(neuron_config=nc, **data)

        @classmethod
        def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
            return NeuronConfig

    class NeuronBaseModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: InferenceConfig, optimize_inference: bool = True):
            super().__init__()
            self.config = config
            self.neuron_config = config.neuron_config
            self.batch_size = config.neuron_config.batch_size
            self.n_positions = config.neuron_config.n_positions
            self.vocab_size = config.vocab_size
            self.setup_attr_for_model(config)
            self.init_model(config)

    class NeuronBaseForCausalLM(nn.Module):  # type: ignore[no-redef]
        _model_cls = None

        def __init__(self, *args, **kwargs):
            super().__init__()

        @classmethod
        def get_neuron_config_cls(cls):
            return cls.get_config_cls().get_neuron_config_cls()


def get_rmsnorm_cls():
    """Return LlamaRMSNorm on CPU and CustomRMSNorm on Neuron device."""
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class GRTPearlNorm(nn.Module):
    """Pair-aware RMSNorm preserving planar pair phase angles."""

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        radius_sq = (pairs * pairs).sum(dim=-1)
        rms = torch.rsqrt(radius_sq.mean(dim=-1, keepdim=True) + self.eps)
        normed = x * rms
        return normed.to(orig_dtype) * self.weight.to(orig_dtype)


class GRTNeuronConfig(NeuronConfig):
    """NeuronConfig specialized for Geometric Rotational Transformer (GRT)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGRTAttention


class GRTInferenceConfig(InferenceConfig):
    """InferenceConfig supporting GRT architectural hyperparameters and HF aliases."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = False
        self.o_bias = False

        if not hasattr(self, "hidden_size"):
            self.hidden_size = 256

        if hasattr(self, "num_attention_heads") and not hasattr(self, "num_heads"):
            self.num_heads = self.num_attention_heads
        elif hasattr(self, "num_heads") and not hasattr(self, "num_attention_heads"):
            self.num_attention_heads = self.num_heads
        elif not hasattr(self, "num_attention_heads"):
            self.num_attention_heads = 4
            self.num_heads = 4

        if hasattr(self, "num_key_value_heads") and not hasattr(self, "num_kv_heads"):
            self.num_kv_heads = self.num_key_value_heads
        elif hasattr(self, "num_kv_heads") and not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.num_kv_heads
        elif not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.num_attention_heads
            self.num_kv_heads = self.num_attention_heads

        if hasattr(self, "num_hidden_layers") and not hasattr(self, "num_layers"):
            self.num_layers = self.num_hidden_layers
        elif hasattr(self, "num_layers") and not hasattr(self, "num_hidden_layers"):
            self.num_hidden_layers = self.num_layers
        elif not hasattr(self, "num_hidden_layers"):
            self.num_hidden_layers = 12
            self.num_layers = 12

        if hasattr(self, "intermediate_size") and self.intermediate_size is not None:
            if not hasattr(self, "ffn_hidden_size"):
                self.ffn_hidden_size = self.intermediate_size
        elif hasattr(self, "ffn_hidden_size") and self.ffn_hidden_size is not None:
            self.intermediate_size = self.ffn_hidden_size
        else:
            self.intermediate_size = 4 * self.hidden_size
            self.ffn_hidden_size = self.intermediate_size

        if not hasattr(self, "vocab_size"):
            self.vocab_size = 151936

        if hasattr(self, "max_position_embeddings") and not hasattr(self, "max_seq_len"):
            self.max_seq_len = self.max_position_embeddings
        elif hasattr(self, "max_seq_len") and not hasattr(self, "max_position_embeddings"):
            self.max_position_embeddings = self.max_seq_len
        elif not hasattr(self, "max_position_embeddings"):
            self.max_position_embeddings = 2048
            self.max_seq_len = 2048

        if hasattr(self, "rope_theta") and not hasattr(self, "pos_freq_base"):
            self.pos_freq_base = float(self.rope_theta)
        elif hasattr(self, "pos_freq_base") and not hasattr(self, "rope_theta"):
            self.rope_theta = float(self.pos_freq_base)
        elif not hasattr(self, "rope_theta"):
            self.rope_theta = 10000.0
            self.pos_freq_base = 10000.0

        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = 1e-6

        if not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if hasattr(self, "head_scale") and not hasattr(self, "output_scale_init"):
            self.output_scale_init = float(self.head_scale)
        elif hasattr(self, "output_scale_init") and not hasattr(self, "head_scale"):
            self.head_scale = float(self.output_scale_init)
        elif not hasattr(self, "head_scale"):
            self.head_scale = 30.0
            self.output_scale_init = 30.0

        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = 0

        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"

        if not hasattr(self, "tie_word_embeddings"):
            self.tie_word_embeddings = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "head_dim",
            "head_scale",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[GRTNeuronConfig]:
        return GRTNeuronConfig


class NeuronGRTEmbedding(nn.Module):
    """XLA/Neuron-compatible rotational phase-angle embedding.

    Stores `phase_angles` of shape `(vocab_size, hidden_size // 2)` and
    `log_magnitude` of shape `(vocab_size,)`, reconstructing planar pairs
    `(r * cos(theta), r * sin(theta))` in pure PyTorch/XLA tensor ops.
    """

    def __init__(
        self,
        vocab_size: Union[int, GRTInferenceConfig],
        hidden_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        clamp_min: float = -5.0,
        clamp_max: float = 5.0,
    ):
        super().__init__()
        if not isinstance(vocab_size, int):
            cfg = vocab_size
            hidden_size = cfg.hidden_size
            padding_idx = getattr(cfg, "pad_token_id", padding_idx)
            if hasattr(cfg, "neuron_config") and getattr(cfg.neuron_config, "torch_dtype", None) is not None:
                dtype = cfg.neuron_config.torch_dtype
            vocab_size = cfg.vocab_size

        if hidden_size is None:
            raise ValueError("`hidden_size` must be provided when `vocab_size` is an int.")
        if hidden_size % 2 != 0:
            raise ValueError(f"`hidden_size` must be even for rotational pairs, got {hidden_size}")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.n_pairs = self.hidden_size // 2
        self.padding_idx = padding_idx
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.phase_angles = nn.Parameter(torch.zeros(self.vocab_size, self.n_pairs, dtype=dtype))
        self.log_magnitude = nn.Parameter(torch.zeros(self.vocab_size, dtype=dtype))

    def table(self) -> torch.Tensor:
        """Materialize the full `(vocab_size, hidden_size)` rotational embedding table."""
        r = torch.exp(torch.clamp(self.log_magnitude, min=self.clamp_min, max=self.clamp_max)).unsqueeze(-1)
        cos_t = torch.cos(self.phase_angles)
        sin_t = torch.sin(self.phase_angles)
        pairs = torch.stack([r * cos_t, r * sin_t], dim=-1)
        return pairs.reshape(self.vocab_size, self.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        theta = F.embedding(input_ids, self.phase_angles)
        log_mag = F.embedding(input_ids, self.log_magnitude.unsqueeze(-1))
        r = torch.exp(torch.clamp(log_mag, min=self.clamp_min, max=self.clamp_max))
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        pairs = torch.stack([r * cos_t, r * sin_t], dim=-1)
        out = pairs.reshape(*input_ids.shape, self.hidden_size)
        if self.padding_idx is not None:
            mask = (input_ids != self.padding_idx).unsqueeze(-1).to(out.dtype)
            out = out * mask
        return out


class NeuronGRTRotationalLinear(nn.Module):
    """XLA/Neuron-compatible Givens rotational linear projection.

    Applies 2x2 Givens rotations `(x_even * cos - x_odd * sin, x_even * sin + x_odd * cos)`
    across dimension pairs, followed by an optional linear projection when
    `in_features != out_features`, and optional per-feature `scale` and `bias`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
        use_scale: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        if in_features % 2 != 0:
            raise ValueError(f"`in_features` must be even for 2x2 Givens rotations, got {in_features}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_pairs = self.in_features // 2

        self.angles = nn.Parameter(torch.zeros(self.n_pairs, dtype=dtype))
        if use_scale:
            self.scale = nn.Parameter(torch.ones(self.out_features, dtype=dtype))
        else:
            self.register_parameter("scale", None)

        if self.in_features != self.out_features:
            self.proj = nn.Linear(self.in_features, self.out_features, bias=False, dtype=dtype)
        else:
            self.proj = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pairs = x.reshape(*x.shape[:-1], self.n_pairs, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        cos_a = torch.cos(self.angles).to(x.dtype)
        sin_a = torch.sin(self.angles).to(x.dtype)

        rot_even = x_even * cos_a - x_odd * sin_a
        rot_odd = x_even * sin_a + x_odd * cos_a
        out = torch.stack([rot_even, rot_odd], dim=-1).reshape(*x.shape[:-1], self.in_features)

        if self.proj is not None:
            out = self.proj(out)

        if self.scale is not None:
            out = out * self.scale.to(out.dtype)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out


class NeuronGRTCosineHead(nn.Module):
    """XLA/Neuron-compatible cosine similarity language modeling head.

    Computes scaled cosine similarity between L2-normalized hidden states and
    unit target vectors reconstructed from `phase_angles` `(vocab_size, n_pairs)`.
    """

    def __init__(
        self,
        hidden_size: Union[int, GRTInferenceConfig],
        vocab_size: Optional[int] = None,
        init_scale: float = 30.0,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not isinstance(hidden_size, int):
            cfg = hidden_size
            vocab_size = cfg.vocab_size
            init_scale = getattr(cfg, "head_scale", getattr(cfg, "output_scale_init", 30.0))
            eps = getattr(cfg, "rms_norm_eps", eps)
            if hasattr(cfg, "neuron_config") and getattr(cfg.neuron_config, "torch_dtype", None) is not None:
                dtype = cfg.neuron_config.torch_dtype
            hidden_size = cfg.hidden_size

        if vocab_size is None:
            raise ValueError("`vocab_size` must be provided when `hidden_size` is an int.")
        if hidden_size % 2 != 0:
            raise ValueError(f"`hidden_size` must be even for rotational pairs, got {hidden_size}")

        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.n_pairs = self.hidden_size // 2
        self.eps = eps

        self.phase_angles = nn.Parameter(torch.zeros(self.vocab_size, self.n_pairs, dtype=dtype))
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=dtype))

    def unit_table(self) -> torch.Tensor:
        """Return unit-normalized target vectors of shape `(vocab_size, hidden_size)`."""
        cos_t = torch.cos(self.phase_angles)
        sin_t = torch.sin(self.phase_angles)
        w = torch.stack([cos_t, sin_t], dim=-1).reshape(self.vocab_size, self.hidden_size)
        return w * (1.0 / math.sqrt(float(self.n_pairs)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_norm = hidden_states / (torch.norm(hidden_states, p=2, dim=-1, keepdim=True) + self.eps)
        w_norm = self.unit_table().to(x_norm.dtype)
        return torch.matmul(x_norm, w_norm.t()) * self.scale.to(x_norm.dtype)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    rope_theta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply 2D planar rotary position encoding to `(batch, num_heads, seq_len, head_dim)`."""
    head_dim = x.shape[-1]
    n_pairs = head_dim // 2
    seq_len = x.shape[2]
    device = x.device
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    if position_ids is None:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        pos = position_ids.to(torch.float32)
    freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (B, S, n_pairs)
    cos = torch.cos(freqs).unsqueeze(1).to(x.dtype)  # (B, 1, S, n_pairs)
    sin = torch.sin(freqs).unsqueeze(1).to(x.dtype)

    pairs = x.reshape(*x.shape[:-1], n_pairs, 2)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    rot = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape_as(x)
    return rot, cos, sin


class NeuronGRTAttention(nn.Module):
    """XLA/Neuron-compatible GRT attention with Givens rotational Q/K/V/O projections."""

    def __init__(self, config: GRTInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = max(1, self.num_heads // self.num_kv_heads)
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.scaling = self.head_dim**-0.5

        dtype = getattr(config.neuron_config, "torch_dtype", torch.float32)
        self.q_proj = NeuronGRTRotationalLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.k_proj = NeuronGRTRotationalLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.v_proj = NeuronGRTRotationalLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.o_proj = NeuronGRTRotationalLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, cos_cache, sin_cache = _apply_rotary_pos_emb(q, position_ids, self.rope_theta)
        k, _, _ = _apply_rotary_pos_emb(k, position_ids, self.rope_theta)

        if past_key_value is not None and len(past_key_value) == 2 and past_key_value[0] is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v)

        if self.num_key_value_groups > 1:
            k_expanded = k[:, :, None, :, :].expand(
                bsz, self.num_kv_heads, self.num_key_value_groups, k.shape[2], self.head_dim
            ).reshape(bsz, self.num_heads, k.shape[2], self.head_dim)
            v_expanded = v[:, :, None, :, :].expand(
                bsz, self.num_kv_heads, self.num_key_value_groups, v.shape[2], self.head_dim
            ).reshape(bsz, self.num_heads, v.shape[2], self.head_dim)
        else:
            k_expanded = k
            v_expanded = v

        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attention_mask
        elif q_len > 1:
            kv_len = k_expanded.shape[2]
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=hidden_states.device, dtype=torch.float32),
                diagonal=kv_len - q_len + 1,
            )
            attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value, cos_cache, sin_cache


class NeuronGRTMLP(nn.Module):
    """XLA/Neuron-compatible rotational SwiGLU feed-forward network for GRT."""

    def __init__(self, config: GRTInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "intermediate_size", 4 * self.hidden_size)
        dtype = getattr(config.neuron_config, "torch_dtype", torch.float32)

        self.gate_proj = NeuronGRTRotationalLinear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=dtype
        )
        self.up_proj = NeuronGRTRotationalLinear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = NeuronGRTRotationalLinear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        output = self.down_proj(gated)
        return output, None


class NeuronGRTDecoderLayer(nn.Module):
    """Single GRT decoder block with rotational attention and rotational MLP."""

    def __init__(self, config: GRTInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronGRTAttention(config)
        self.mlp = NeuronGRTMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states, adapter_ids=adapter_ids)[0]
        hidden_states = residual + mlp_out

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronGRTModel(NeuronBaseModel):
    """NeuronBaseModel implementation for Geometric Rotational Transformer (GRT)."""

    def setup_attr_for_model(self, config: GRTInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: GRTInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        dtype = getattr(config.neuron_config, "torch_dtype", torch.float32)
        self.embed_tokens = NeuronGRTEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [NeuronGRTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = NeuronGRTCosineHead(
            config.hidden_size,
            config.vocab_size,
            init_scale=config.head_scale,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )


class NeuronGRTForCausalLM(NeuronBaseForCausalLM):
    """NeuronBaseForCausalLM wrapper for GRT causal language modeling."""

    _model_cls = NeuronGRTModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Convert HuggingFace GRT state_dict keys into NeuronGRTModel layout."""
        neuron_config = config.neuron_config
        converted: dict = {}

        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]

            if new_key.startswith("embedding."):
                new_key = "embed_tokens." + new_key[len("embedding."):]
            elif new_key.startswith("final_norm."):
                suffix = new_key[len("final_norm."):]
                if suffix == "scale":
                    suffix = "weight"
                new_key = "norm." + suffix
            elif new_key == "lm_head.output_scale":
                new_key = "lm_head.scale"
            elif new_key.startswith("blocks."):
                new_key = "layers." + new_key[len("blocks."):]
                new_key = new_key.replace(".attention.", ".self_attn.")
                new_key = new_key.replace(".ffn.", ".mlp.")
                new_key = new_key.replace(".norm1.scale", ".input_layernorm.weight")
                new_key = new_key.replace(".norm1.weight", ".input_layernorm.weight")
                new_key = new_key.replace(".norm_ffn.scale", ".post_attention_layernorm.weight")
                new_key = new_key.replace(".norm_ffn.weight", ".post_attention_layernorm.weight")

            converted[new_key] = value

        if getattr(neuron_config, "vocab_parallel", False):
            converted["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            converted[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        converted["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        if getattr(config, "tie_word_embeddings", True):
            if (
                "embed_tokens.phase_angles" in converted
                and "lm_head.phase_angles" not in converted
            ):
                NeuronGRTForCausalLM.update_state_dict_for_tied_weights(converted)

        return converted

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: dict):
        if "embed_tokens.phase_angles" in state_dict:
            state_dict["lm_head.phase_angles"] = state_dict["embed_tokens.phase_angles"].clone()
        if "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return GRTInferenceConfig

    def get_compiler_args(self):
        compiler_args = (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--auto-cast=none --model-type transformer -O1"
        )
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        )
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args

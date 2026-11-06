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

import math
import tempfile

import pytest
import torch

from neuronx_distributed_inference.models.grt.modeling_grt import (
    GRTInferenceConfig,
    GRTNeuronConfig,
    NeuronGRTAttention,
    NeuronGRTCosineHead,
    NeuronGRTDecoderLayer,
    NeuronGRTEmbedding,
    NeuronGRTForCausalLM,
    NeuronGRTMLP,
    NeuronGRTModel,
    NeuronGRTRotationalLinear,
)


def test_grt_config_initialization_and_aliases():
    neuron_config = GRTNeuronConfig(tp_degree=2, batch_size=2, seq_len=64, torch_dtype=torch.float32)
    assert neuron_config.attn_cls == NeuronGRTAttention

    config = GRTInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=128,
        num_heads=8,
        num_kv_heads=4,
        num_layers=6,
        ffn_hidden_size=512,
        vocab_size=1024,
        max_seq_len=2048,
        pos_freq_base=10000.0,
        output_scale_init=25.0,
    )

    assert config.hidden_size == 128
    assert config.num_attention_heads == 8
    assert config.num_heads == 8
    assert config.num_key_value_heads == 4
    assert config.num_kv_heads == 4
    assert config.num_hidden_layers == 6
    assert config.num_layers == 6
    assert config.intermediate_size == 512
    assert config.ffn_hidden_size == 512
    assert config.vocab_size == 1024
    assert config.max_position_embeddings == 2048
    assert config.max_seq_len == 2048
    assert config.rope_theta == 10000.0
    assert config.head_dim == 16
    assert config.head_scale == 25.0
    assert config.get_neuron_config_cls() == GRTNeuronConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config.save(tmpdir)
        loaded = GRTInferenceConfig.load(tmpdir)
        assert loaded.hidden_size == 128
        assert loaded.num_attention_heads == 8
        assert loaded.num_key_value_heads == 4
        assert loaded.intermediate_size == 512
        assert loaded.head_scale == 25.0


def test_neuron_grt_embedding():
    vocab_size = 32
    hidden_size = 16
    emb = NeuronGRTEmbedding(vocab_size, hidden_size, padding_idx=0, dtype=torch.float32)
    assert emb.phase_angles.shape == (vocab_size, hidden_size // 2)
    assert emb.log_magnitude.shape == (vocab_size,)

    with torch.no_grad():
        emb.phase_angles.fill_(math.pi / 4.0)
        emb.log_magnitude.fill_(math.log(2.0))

    input_ids = torch.tensor([[0, 1, 5], [2, 3, 0]], dtype=torch.long)
    out = emb(input_ids)
    assert out.shape == (2, 3, hidden_size)

    # Padding token index 0 should be zeroed out
    assert torch.allclose(out[0, 0], torch.zeros(hidden_size))
    assert torch.allclose(out[1, 2], torch.zeros(hidden_size))

    # Non-padding tokens have r = 2.0, theta = pi/4 -> each coordinate = 2 * sqrt(2)/2 = sqrt(2)
    expected_val = 2.0 * math.cos(math.pi / 4.0)
    assert torch.allclose(out[0, 1], torch.full((hidden_size,), expected_val), atol=1e-5)

    table = emb.table()
    assert table.shape == (vocab_size, hidden_size)


def test_neuron_grt_rotational_linear():
    dim = 16
    rot_lin = NeuronGRTRotationalLinear(dim, dim, bias=True, use_scale=True, dtype=torch.float32)
    x = torch.randn(2, 4, dim)

    # At zero angles, unit scale, zero bias: output equals input
    out_identity = rot_lin(x)
    assert torch.allclose(out_identity, x, atol=1e-6)

    # Non-zero angles with unit scale and zero bias preserve Euclidean norm per token
    with torch.no_grad():
        rot_lin.angles.copy_(torch.linspace(0.1, 1.5, dim // 2))
    out_rot = rot_lin(x)
    assert torch.allclose(out_rot.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    # Verify exact 2x2 Givens rotation formula on a single pair
    rot2 = NeuronGRTRotationalLinear(2, 2, bias=False, use_scale=False, dtype=torch.float32)
    with torch.no_grad():
        rot2.angles.fill_(math.pi / 2.0)
    vec = torch.tensor([[1.0, 0.0]])
    rotated = rot2(vec)
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0]]), atol=1e-6)

    # Dimension-changing projection (in_features != out_features)
    rot_proj = NeuronGRTRotationalLinear(16, 32, bias=True, use_scale=True, dtype=torch.float32)
    out_proj = rot_proj(x)
    assert out_proj.shape == (2, 4, 32)


def test_neuron_grt_cosine_head():
    hidden_size = 16
    vocab_size = 24
    init_scale = 20.0
    head = NeuronGRTCosineHead(hidden_size, vocab_size, init_scale=init_scale, dtype=torch.float32)

    with torch.no_grad():
        head.phase_angles.uniform_(-math.pi, math.pi)

    unit_table = head.unit_table()
    assert unit_table.shape == (vocab_size, hidden_size)
    norms = unit_table.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Hidden state aligned with token 3's unit vector should achieve maximum logit = scale at index 3
    target_vec = unit_table[3].unsqueeze(0).unsqueeze(0) * 5.0  # (1, 1, hidden_size)
    logits = head(target_vec)
    assert logits.shape == (1, 1, vocab_size)
    assert torch.allclose(logits[0, 0, 3], torch.tensor(init_scale), atol=1e-4)
    assert int(logits.argmax(dim=-1).item()) == 3


def test_neuron_grt_model_and_state_dict_conversion():
    neuron_config = GRTNeuronConfig(tp_degree=2, batch_size=1, seq_len=8, torch_dtype=torch.float32, vocab_parallel=True)
    config = GRTInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=128,
        head_scale=30.0,
        tie_word_embeddings=True,
    )

    model = NeuronGRTModel(config)
    assert isinstance(model.embed_tokens, NeuronGRTEmbedding)
    assert isinstance(model.lm_head, NeuronGRTCosineHead)
    assert len(model.layers) == 2

    x = torch.randn(1, 4, 32)
    layer_out = model.layers[0](x)
    assert layer_out[0].shape == (1, 4, 32)

    hf_state_dict = {
        "model.embedding.phase_angles": torch.randn(64, 16),
        "model.embedding.log_magnitude": torch.zeros(64),
        "model.blocks.0.norm1.scale": torch.ones(32),
        "model.blocks.0.norm_ffn.scale": torch.ones(32),
        "model.final_norm.scale": torch.ones(32),
        "lm_head.output_scale": torch.tensor(30.0),
    }

    converted = NeuronGRTForCausalLM.convert_hf_to_neuron_state_dict(hf_state_dict, config)
    assert "embed_tokens.phase_angles" in converted
    assert "embed_tokens.log_magnitude" in converted
    assert "layers.0.input_layernorm.weight" in converted
    assert "layers.0.post_attention_layernorm.weight" in converted
    assert "norm.weight" in converted
    assert "lm_head.scale" in converted
    assert "lm_head.phase_angles" in converted
    assert torch.equal(converted["lm_head.phase_angles"], converted["embed_tokens.phase_angles"])
    assert "rank_util.rank" in converted
    assert "embed_tokens.rank_util.rank" in converted
    assert "layers.0.self_attn.rank_util.rank" in converted
    assert "layers.1.self_attn.rank_util.rank" in converted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

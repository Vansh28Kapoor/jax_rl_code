"""
Tests for the In-Context MMDiT model and components.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from mmdit_jax.in_context_models import InContextMMDiT
from mmdit_jax.in_context_blocks import InContextMMDiTBlock
from mmdit_jax.in_context_final_layer import SimpleFinalLayer, RMSNorm
from mmdit_jax.layers import TimestepEmbedding


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_rmsnorm_basic(self):
        """Test basic RMSNorm forward pass."""
        key = jax.random.PRNGKey(42)
        dim = 64

        norm = RMSNorm(dim)
        x = jax.random.normal(key, (dim,))

        out = norm(x)

        assert out.shape == (dim,)
        # RMS of output (before gamma) should be approximately 1
        rms_out = jnp.sqrt(jnp.mean(out**2))
        assert jnp.isclose(rms_out, 1.0, atol=0.1)

    def test_rmsnorm_gamma_learnable(self):
        """Test that gamma is learnable."""
        dim = 32
        norm = RMSNorm(dim)

        assert norm.gamma.shape == (dim,)
        assert jnp.allclose(norm.gamma, jnp.ones(dim))

    def test_rmsnorm_batched(self):
        """Test RMSNorm with vmap."""
        key = jax.random.PRNGKey(42)
        dim = 64
        seq_len = 10

        norm = RMSNorm(dim)
        x = jax.random.normal(key, (seq_len, dim))

        out = jax.vmap(norm)(x)

        assert out.shape == (seq_len, dim)


class TestSimpleFinalLayer:
    """Tests for SimpleFinalLayer."""

    def test_simple_final_layer_basic(self):
        """Test basic SimpleFinalLayer forward pass."""
        key = jax.random.PRNGKey(42)
        dim_in, dim_out = 64, 32
        seq_len = 10

        layer = SimpleFinalLayer(dim_in, dim_out, key=key)
        x = jax.random.normal(key, (seq_len, dim_in))

        out = layer(x)

        assert out.shape == (seq_len, dim_out)

    def test_simple_final_layer_zero_init(self):
        """Test that linear layer is zero-initialized."""
        key = jax.random.PRNGKey(42)
        dim_in, dim_out = 64, 32

        layer = SimpleFinalLayer(dim_in, dim_out, key=key)

        # Linear weights and biases should be zero
        assert jnp.allclose(layer.linear_out.weight, 0.0)
        assert jnp.allclose(layer.linear_out.bias, 0.0)

    def test_simple_final_layer_initial_output(self):
        """Test that initial output is zero (due to zero-initialized linear)."""
        key = jax.random.PRNGKey(42)
        dim_in, dim_out = 64, 32
        seq_len = 10

        layer = SimpleFinalLayer(dim_in, dim_out, key=key)
        x = jax.random.normal(key, (seq_len, dim_in))

        out = layer(x)

        # Output should be all zeros initially
        assert jnp.allclose(out, 0.0)


class TestInContextMMDiTBlock:
    """Tests for InContextMMDiTBlock."""

    def test_block_basic_forward(self):
        """Test basic forward pass of InContextMMDiTBlock."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        seq_lens = (10, 8)

        block = InContextMMDiTBlock(
            dim_modalities=dim_modalities,
            dim_head=32,
            heads=4,
            key=keys[0]
        )

        # Create input tokens
        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        # Forward pass
        outputs = block(tokens)

        assert len(outputs) == 2
        assert outputs[0].shape == (10, 64)
        assert outputs[1].shape == (8, 32)

    def test_block_with_attention_mask(self):
        """Test InContextMMDiTBlock with attention mask."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        seq_lens = (10, 8)
        total_seq = sum(seq_lens)

        block = InContextMMDiTBlock(
            dim_modalities=dim_modalities,
            dim_head=32,
            heads=4,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        # Create causal mask for second modality
        mask = jnp.ones((total_seq, total_seq), dtype=bool)
        causal = jnp.tril(jnp.ones((seq_lens[1], seq_lens[1]), dtype=bool))
        mask = mask.at[seq_lens[0]:, seq_lens[0]:].set(causal)

        outputs = block(tokens, attention_mask=mask)

        assert len(outputs) == 2
        assert outputs[0].shape == (10, 64)
        assert outputs[1].shape == (8, 32)

    def test_block_no_conditioning(self):
        """Test that block doesn't require any conditioning."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        dim_modalities = (64, 32)
        seq_lens = (5, 5)

        block = InContextMMDiTBlock(
            dim_modalities=dim_modalities,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        # Should work without any conditioning argument
        outputs = block(tokens)

        assert len(outputs) == 2

    def test_block_uses_rmsnorm(self):
        """Test that block uses RMSNorm instead of AdaptiveLayerNorm."""
        key = jax.random.PRNGKey(42)
        dim_modalities = (64, 32)

        block = InContextMMDiTBlock(
            dim_modalities=dim_modalities,
            key=key
        )

        # Check that norms are RMSNorm
        assert all(isinstance(norm, RMSNorm) for norm in block.attn_norms)
        assert all(isinstance(norm, RMSNorm) for norm in block.ff_norms)


class TestInContextMMDiT:
    """Tests for InContextMMDiT model."""

    def test_model_basic_forward(self):
        """Test basic forward pass of InContextMMDiT."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        dim_outs = (64, 32)
        seq_lens = (10, 8)

        model = InContextMMDiT(
            depth=2,
            dim_modalities=dim_modalities,
            dim_outs=dim_outs,
            dim_head=32,
            heads=4,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        outputs = model(tokens)

        assert len(outputs) == 2
        assert outputs[0].shape == (10, 64)
        assert outputs[1].shape == (8, 32)

    def test_model_batched(self):
        """Test batched processing of InContextMMDiT."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        dim_outs = (64, 32)
        batch_size = 3
        seq_lens = (10, 8)

        model = InContextMMDiT(
            depth=2,
            dim_modalities=dim_modalities,
            dim_outs=dim_outs,
            dim_head=32,
            heads=4,
            key=keys[0]
        )
        model = eqx.filter_vmap(model, in_axes=0)

        tokens = tuple(
            jax.random.normal(keys[i + 1], (batch_size, seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        outputs = model(tokens)

        assert len(outputs) == 2
        assert outputs[0].shape == (batch_size, 10, 64)
        assert outputs[1].shape == (batch_size, 8, 32)

    def test_model_with_attention_mask(self):
        """Test InContextMMDiT with attention mask."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        dim_outs = (64, 32)
        seq_lens = (10, 8)
        total_seq = sum(seq_lens)

        model = InContextMMDiT(
            depth=2,
            dim_modalities=dim_modalities,
            dim_outs=dim_outs,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        # Create attention mask
        mask = jnp.ones((total_seq, total_seq), dtype=bool)

        outputs = model(tokens, attention_mask=mask)

        assert len(outputs) == 2

    def test_model_gradient_flow(self):
        """Test gradient flow through InContextMMDiT."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        dim_outs = (64, 32)
        seq_lens = (8, 6)

        model = InContextMMDiT(
            depth=2,
            dim_modalities=dim_modalities,
            dim_outs=dim_outs,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        def loss_fn(model):
            outputs = model(tokens)
            return jnp.mean(outputs[0]**2) + jnp.mean(outputs[1]**2)

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)

        assert isinstance(loss_val, jax.Array)
        assert loss_val.ndim == 0
        # Gradients should exist
        assert grads.blocks is not None

    def test_model_no_timestep_parameter(self):
        """Test that model doesn't take timestep parameter."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        dim_modalities = (64, 32)
        dim_outs = (64, 32)
        seq_lens = (5, 5)

        model = InContextMMDiT(
            depth=1,
            dim_modalities=dim_modalities,
            dim_outs=dim_outs,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        # Should work without timestep
        outputs = model(tokens)
        assert len(outputs) == 2

    def test_model_uses_simple_final_layer(self):
        """Test that model uses SimpleFinalLayer."""
        key = jax.random.PRNGKey(42)

        model = InContextMMDiT(
            depth=2,
            dim_modalities=(64, 32),
            dim_outs=(64, 32),
            key=key
        )

        assert all(isinstance(fl, SimpleFinalLayer) for fl in model.final_layers)

    def test_model_different_output_dims(self):
        """Test model with different input and output dimensions."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        dim_modalities = (64, 32)
        dim_outs = (128, 16)  # Different from inputs
        seq_lens = (10, 8)

        model = InContextMMDiT(
            depth=2,
            dim_modalities=dim_modalities,
            dim_outs=dim_outs,
            key=keys[0]
        )

        tokens = tuple(
            jax.random.normal(keys[i + 1], (seq_len, dim))
            for i, (seq_len, dim) in enumerate(zip(seq_lens, dim_modalities))
        )

        outputs = model(tokens)

        assert outputs[0].shape == (10, 128)  # Changed output dim
        assert outputs[1].shape == (8, 16)   # Changed output dim


class TestInContextMMDiTWithTimeTokens:
    """
    Tests for using InContextMMDiT with externally managed time tokens.
    This demonstrates the intended usage pattern for imitation learning.
    """

    def test_prepend_time_tokens(self):
        """Test prepending time tokens to action modality."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)

        obs_dim = 64
        action_dim = 32
        obs_seq_len = 10
        action_seq_len = 8
        num_time_tokens = 2

        # Create model
        model = InContextMMDiT(
            depth=2,
            dim_modalities=(obs_dim, action_dim),
            dim_outs=(obs_dim, action_dim),
            key=keys[0]
        )

        # Create time embedding (user handles this)
        time_embed = TimestepEmbedding(dim_embed=256, dim_out=256, key=keys[1])
        time_to_token = eqx.nn.Linear(256, action_dim, key=keys[2])

        # Embed timestep and create time tokens
        t = jnp.array(0.5)
        time_cond = time_embed(t)
        time_token = time_to_token(time_cond)
        time_tokens = jnp.tile(time_token[None, :], (num_time_tokens, 1))

        # Create observation and action tokens
        obs_tokens = jax.random.normal(keys[3], (obs_seq_len, obs_dim))
        action_tokens = jax.random.normal(keys[4], (action_seq_len, action_dim))

        # Prepend time tokens to actions
        action_with_time = jnp.concatenate([time_tokens, action_tokens], axis=0)

        # Forward pass
        outputs = model((obs_tokens, action_with_time))

        # Check output shapes
        assert outputs[0].shape == (obs_seq_len, obs_dim)
        assert outputs[1].shape == (num_time_tokens + action_seq_len, action_dim)

        # Strip time tokens from output
        action_out = outputs[1][num_time_tokens:]
        assert action_out.shape == (action_seq_len, action_dim)

    def test_imitation_learning_mask(self):
        """Test attention mask for imitation learning (obs full, action causal)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        obs_dim = 64
        action_dim = 32
        obs_seq_len = 10
        action_seq_len = 6
        num_time_tokens = 2
        action_with_time_len = num_time_tokens + action_seq_len
        total_seq = obs_seq_len + action_with_time_len

        model = InContextMMDiT(
            depth=2,
            dim_modalities=(obs_dim, action_dim),
            dim_outs=(obs_dim, action_dim),
            key=keys[0]
        )

        # Create tokens
        obs_tokens = jax.random.normal(keys[1], (obs_seq_len, obs_dim))
        action_with_time = jax.random.normal(keys[2], (action_with_time_len, action_dim))

        # Create mask: obs can see all obs, action sees obs + causal action
        mask = jnp.ones((total_seq, total_seq), dtype=bool)

        # Action tokens (including time) are causal among themselves
        action_causal = jnp.tril(jnp.ones((action_with_time_len, action_with_time_len), dtype=bool))
        mask = mask.at[obs_seq_len:, obs_seq_len:].set(action_causal)

        # Obs tokens don't see action tokens (optional, depends on use case)
        # mask = mask.at[:obs_seq_len, obs_seq_len:].set(False)

        outputs = model((obs_tokens, action_with_time), attention_mask=mask)

        assert outputs[0].shape == (obs_seq_len, obs_dim)
        assert outputs[1].shape == (action_with_time_len, action_dim)

    def test_multiple_modalities_with_selective_time(self):
        """Test with 3 modalities, only one gets time tokens."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 7)

        dims = (64, 48, 32)  # obs, state, action
        seq_lens = (10, 5, 8)
        num_time_tokens = 2

        model = InContextMMDiT(
            depth=2,
            dim_modalities=dims,
            dim_outs=dims,
            key=keys[0]
        )

        # Create tokens (only action gets time tokens)
        obs_tokens = jax.random.normal(keys[1], (seq_lens[0], dims[0]))
        state_tokens = jax.random.normal(keys[2], (seq_lens[1], dims[1]))

        # Create time tokens for action modality
        time_embed = TimestepEmbedding(dim_embed=256, dim_out=256, key=keys[3])
        time_to_token = eqx.nn.Linear(256, dims[2], key=keys[4])
        time_cond = time_embed(jnp.array(0.5))
        time_tokens = jnp.tile(time_to_token(time_cond)[None, :], (num_time_tokens, 1))

        action_tokens = jax.random.normal(keys[5], (seq_lens[2], dims[2]))
        action_with_time = jnp.concatenate([time_tokens, action_tokens], axis=0)

        # Forward pass
        outputs = model((obs_tokens, state_tokens, action_with_time))

        assert outputs[0].shape == (seq_lens[0], dims[0])
        assert outputs[1].shape == (seq_lens[1], dims[1])
        assert outputs[2].shape == (num_time_tokens + seq_lens[2], dims[2])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

"""
Tests for attention mechanisms in mmdit_jax.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.attention import JointAttention


def test_joint_attention_basic():
    """Test basic JointAttention functionality."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Create layer for two modalities
    dim_inputs = (128, 64)
    layer = JointAttention(dim_inputs=dim_inputs, dim_head=64, heads=8, key=keys[0])
    layer = eqx.filter_vmap(layer)

    # Test inputs
    batch_size = 2
    seq_len1, seq_len2 = 10, 8
    x1 = jax.random.normal(keys[1], (batch_size, seq_len1, dim_inputs[0]))
    x2 = jax.random.normal(keys[2], (batch_size, seq_len2, dim_inputs[1]))
    inputs = (x1, x2)

    # Forward pass
    outputs = layer(inputs)

    # Check output shapes
    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, seq_len1, dim_inputs[0])
    assert outputs[1].shape == (batch_size, seq_len2, dim_inputs[1])

    # Check that outputs are different from inputs (attention applied)
    assert not jnp.allclose(outputs[0], x1)
    assert not jnp.allclose(outputs[1], x2)


def test_joint_attention_with_masks():
    """Test JointAttention with 2D attention masks."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Create layer for two modalities
    dim_inputs = (64, 32)
    seq_len1, seq_len2 = 8, 6
    total_seq = seq_len1 + seq_len2
    layer = JointAttention(dim_inputs=dim_inputs, key=keys[0])

    # Test inputs
    x1 = jax.random.normal(keys[1], (seq_len1, dim_inputs[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_inputs[1]))
    inputs = (x1, x2)

    # Create 2D attention mask (causal for second modality)
    # First modality can attend to everything
    # Second modality can attend to first modality + causal within itself
    attention_mask = jnp.ones((total_seq, total_seq), dtype=bool)
    # Make second modality causal
    action_causal = jnp.tril(jnp.ones((seq_len2, seq_len2), dtype=bool))
    attention_mask = attention_mask.at[seq_len1:, seq_len1:].set(action_causal)

    # Forward pass with mask
    outputs_masked = layer(inputs, attention_mask=attention_mask)

    # Forward pass without mask
    outputs_unmasked = layer(inputs)

    # Outputs should be different due to masking
    # Note: First modality can attend to everything, but second modality is causal,
    # so the joint attention output changes for both modalities because
    # the attention patterns are different (second modality attends differently)
    # At minimum, verify shapes are correct and values are finite
    assert outputs_masked[0].shape == outputs_unmasked[0].shape
    assert outputs_masked[1].shape == outputs_unmasked[1].shape
    assert jnp.all(jnp.isfinite(outputs_masked[0]))
    assert jnp.all(jnp.isfinite(outputs_masked[1]))

    # The second modality should definitely be different due to causal masking
    # (it can't attend to future tokens within its own sequence)
    assert not jnp.allclose(outputs_masked[1], outputs_unmasked[1])


def test_joint_attention_three_modalities():
    """Test JointAttention with three modalities."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # Create layer for three modalities
    dim_inputs = (64, 32, 48)
    layer = JointAttention(dim_inputs=dim_inputs, key=keys[0])

    # Test inputs
    seq_lens = [6, 8, 4]
    inputs = tuple(
        jax.random.normal(keys[i+1], (seq_lens[i], dim_inputs[i]))
        for i in range(3)
    )

    # Forward pass
    outputs = layer(inputs)

    # Check outputs
    assert len(outputs) == 3
    for i in range(3):
        assert outputs[i].shape == (seq_lens[i], dim_inputs[i])
        assert not jnp.allclose(outputs[i], inputs[i])


def test_joint_attention_qk_rmsnorm():
    """Test JointAttention with QK RMSNorm (now mandatory)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Create layer - QK RMSNorm is now always enabled
    dim_inputs = (64, 32)
    layer = JointAttention(dim_inputs=dim_inputs, key=keys[0])

    # Test inputs
    seq_len1, seq_len2 = 6, 4
    x1 = jax.random.normal(keys[1], (seq_len1, dim_inputs[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_inputs[1]))
    inputs = (x1, x2)

    # Forward pass
    outputs = layer(inputs)

    # Check that layer has RMS norm components (always enabled now)
    assert layer.q_rmsnorms is not None
    assert layer.k_rmsnorms is not None
    assert len(layer.q_rmsnorms) == 2
    assert len(layer.k_rmsnorms) == 2

    # Check outputs
    assert len(outputs) == 2
    assert outputs[0].shape == (seq_len1, dim_inputs[0])
    assert outputs[1].shape == (seq_len2, dim_inputs[1])


def test_attention_input_validation():
    """Test input validation in attention layers."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Test invalid inputs for JointAttention
    with pytest.raises(AssertionError, match="At least one input modality is required"):
        JointAttention(dim_inputs=(), key=keys[0])

    with pytest.raises(AssertionError, match="dim_head must be positive"):
        JointAttention(dim_inputs=(64,), dim_head=0, key=keys[0])


def test_attention_wrong_input_shapes():
    """Test attention layers with wrong input shapes."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    # Test JointAttention with wrong number of inputs
    joint_layer = JointAttention(dim_inputs=(64, 32), key=keys[0])
    x = jax.random.normal(keys[1], (8, 64))
    
    with pytest.raises(AssertionError, match="Expected 2 inputs, got 1"):
        joint_layer((x,))


def test_attention_gradient_flow():
    """Test that gradients flow through attention layers."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Test JointAttention
    dim_inputs = (32, 32)
    seq_len = 4
    layer = JointAttention(dim_inputs=dim_inputs, heads=2, key=keys[0])
    x1 = jax.random.normal(keys[1], (seq_len, dim_inputs[0]))
    x2 = jax.random.normal(keys[2], (seq_len, dim_inputs[1]))

    def loss_fn(layer, inputs):
        outputs = layer(inputs)
        return jnp.mean(outputs[0]**2) + jnp.mean(outputs[1]**2)

    # Compute gradients
    grad_fn = eqx.filter_grad(loss_fn)
    grads = grad_fn(layer, (x1, x2))

    # Check that gradients exist and are non-zero
    assert grads.to_qkv[0].weight is not None
    assert grads.to_out[0].weight is not None
    assert not jnp.allclose(grads.to_qkv[0].weight, 0.0)
    assert not jnp.allclose(grads.to_out[0].weight, 0.0)


if __name__ == "__main__":
    test_joint_attention_basic()
    test_joint_attention_with_masks()
    test_joint_attention_three_modalities()
    test_joint_attention_qk_rmsnorm()
    test_attention_input_validation()
    test_attention_wrong_input_shapes()
    test_attention_gradient_flow()
    print("All attention tests passed!")
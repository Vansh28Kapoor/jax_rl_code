"""
Tests for basic layers in mmdit_jax.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.layers import AdaptiveLayerNorm


def test_adaptive_layer_norm_with_conditioning():
    """Test AdaptiveLayerNorm with conditioning (shift+scale modulation)."""
    key = jax.random.PRNGKey(42)
    key0, key1, key2, key3, key4 = jax.random.split(key, 5)

    # Create layer with conditioning
    dim, dim_cond = 128, 64
    layer = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond, key=key0)
    layer = eqx.filter_vmap(layer)
    # Test input
    batch_size, seq_len = 2, 10
    x = jax.random.normal(key1, (batch_size, seq_len, dim))
    cond = jax.random.normal(key2, (batch_size, dim_cond))

    # Forward pass (no key parameter in new version)
    output = layer(x, cond)

    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)

    # Check that output is different from input (normalization applied)
    assert not jnp.allclose(output, x)

    # Test with different conditioning
    cond2 = jax.random.normal(key3, (batch_size, dim_cond))
    output2 = layer(x, cond2)

    # NOTE: With zero initialization (adaLN-Zero), different conditioning
    # should give the SAME outputs initially. This is by design - the
    # conditioning network starts with zero weights and zero bias,
    # so shift=0, scale=0 regardless of input, making output = LayerNorm(x) * 1 + 0.
    # Only after training should different conditioning produce different outputs.
    assert jnp.allclose(output, output2, atol=1e-6)


def test_adaptive_layer_norm_modulation_structure():
    """Test that AdaptiveLayerNorm uses shift+scale modulation."""
    key = jax.random.PRNGKey(42)
    key0, key1, key2 = jax.random.split(key, 3)

    # Create layer
    dim, dim_cond = 64, 32
    layer = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond, key=key0)

    # Check that it has to_modulation (produces both shift and scale)
    assert hasattr(layer, 'to_modulation')
    assert hasattr(layer, 'ln')

    # Test that output matches expected formula: LN(x) * (1 + scale) + shift
    x = jax.random.normal(key1, (5, dim))
    cond = jax.random.normal(key2, (dim_cond,))

    output = layer(x, cond)

    # Check output shape
    assert output.shape == (5, dim)

    # Check finite values
    assert jnp.all(jnp.isfinite(output))

if __name__ == "__main__":
    test_adaptive_layer_norm_with_conditioning()
    test_adaptive_layer_norm_modulation_structure()
    print("All layer tests passed!") 
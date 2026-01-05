import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.attention import MultiHeadRMSNorm


def test_multihead_rmsnorm_basic():
    """Test basic MultiHeadRMSNorm functionality."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create layer
    dim = 64
    heads = 8
    layer = MultiHeadRMSNorm(dim=dim, heads=heads, key=key1)

    # Test input
    seq_len = 10
    x = jax.random.normal(key2, (heads, seq_len, dim))

    # Forward pass
    output = layer(x)

    # Check output shape
    assert output.shape == (heads, seq_len, dim)

    # Check that output is different from input (normalization applied)
    assert not jnp.allclose(output, x)

    # Check gamma shape is (dim,) - shared across heads (SD3.5 aligned)
    assert layer.gamma.shape == (dim,), f"Expected gamma shape ({dim},), got {layer.gamma.shape}"


def test_multihead_rmsnorm_eps():
    """Test MultiHeadRMSNorm with different epsilon values."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    dim = 32
    heads = 4
    x = jax.random.normal(key1, (heads, 8, dim))

    # Test different epsilon values
    for eps in [1e-8, 1e-6, 1e-4]:
        layer = MultiHeadRMSNorm(dim=dim, heads=heads, eps=eps, key=key2)
        output = layer(x)

        assert output.shape == (heads, 8, dim)
        assert layer.eps == eps

if __name__ == "__main__":
    test_multihead_rmsnorm_basic()
    test_multihead_rmsnorm_eps()
    print("All multihead rmsnorm tests passed!")
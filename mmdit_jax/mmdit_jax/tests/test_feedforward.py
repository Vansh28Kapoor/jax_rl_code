"""
Tests for feedforward layers in mmdit_jax.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.feedforward import FeedForward


def test_feedforward_basic():
    """Test basic FeedForward functionality."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create layer
    dim = 128
    mult = 4
    layer = FeedForward(dim=dim, mult=mult, key=key1)
    layer = eqx.filter_vmap(layer)

    # Test input
    batch_size, seq_len = 2, 10
    x = jax.random.normal(key2, (batch_size, seq_len, dim))

    # Forward pass
    output = layer(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)

    # Check that output is different from input
    assert not jnp.allclose(output, x)

    # Check that the layer maintains finite values
    assert jnp.all(jnp.isfinite(output))


def test_feedforward_properties():
    """Test FeedForward layer properties."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create layer
    dim = 64
    mult = 2
    layer = FeedForward(dim=dim, mult=mult, key=key1)

    # Check attributes
    assert layer.dim == dim
    assert layer.mult == mult
    
    # Check SwiGLU structure (3 linear layers: in, gate, out)
    assert hasattr(layer, 'linear_in')
    assert hasattr(layer, 'linear_gate')
    assert hasattr(layer, 'linear_out')

    # Test with sequence input
    seq_len = 5
    x = jax.random.normal(key2, (seq_len, dim))
    output = layer(x)

    # Check output shape
    assert output.shape == (seq_len, dim)

    # Check finite values
    assert jnp.all(jnp.isfinite(output))


def test_feedforward_different_inputs():
    """Test FeedForward with different inputs."""
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    # Create layer
    dim = 96
    mult = 3
    layer = FeedForward(dim=dim, mult=mult, key=key1)
    layer = eqx.filter_vmap(layer)

    # Test inputs
    batch_size, seq_len = 2, 8
    x1 = jax.random.normal(key2, (batch_size, seq_len, dim))
    x2 = jax.random.normal(key3, (batch_size, seq_len, dim))

    # Forward passes
    output1 = layer(x1)
    output2 = layer(x2)

    # Check outputs are different
    assert not jnp.allclose(output1, output2)

    # Check shapes
    assert output1.shape == output2.shape == (batch_size, seq_len, dim)


def test_feedforward_gradient_flow():
    """Test that gradients flow through FeedForward layer."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create layer (no vmap needed - layer handles sequences internally)
    dim = 64
    layer = FeedForward(dim=dim, mult=2, key=key1)

    # Test input (sequence of length 5)
    x = jax.random.normal(key2, (2, 5, dim))

    # Define loss function
    def loss_fn(model, x):
        model = eqx.filter_vmap(model)
        return jnp.mean(model(x)**2)

    # Compute gradients
    grad_fn = eqx.filter_grad(loss_fn)
    grads = grad_fn(layer, x)

    # Check that gradients exist for all parameters
    def check_grads(grad_tree):
        for leaf in jax.tree_util.tree_leaves(grad_tree):
            assert jnp.all(jnp.isfinite(leaf))
            assert not jnp.allclose(leaf, 0)  # Should have non-zero gradients

    check_grads(grads)


def test_feedforward_compilation():
    """Test that FeedForward layer can be JIT compiled."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create layer (no vmap needed - layer handles sequences internally)
    dim = 48
    layer = FeedForward(dim=dim, mult=4, key=key1)
    layer = eqx.filter_vmap(layer)

    # Test input (sequence of length 3)
    x = jax.random.normal(key2, (2, 3, dim))

    # Compile forward pass
    compiled_layer = eqx.filter_jit(layer)
    output = compiled_layer(x)

    # Check output shape
    assert output.shape == (2, 3, dim)

    # Check that output is finite
    assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    test_feedforward_basic()
    test_feedforward_properties()
    test_feedforward_different_inputs()
    test_feedforward_gradient_flow()
    test_feedforward_compilation()
    print("All feedforward tests passed!") 
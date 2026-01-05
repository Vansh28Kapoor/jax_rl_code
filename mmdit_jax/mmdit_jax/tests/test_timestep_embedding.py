"""
Tests for timestep embedding in mmdit_jax.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.layers import TimestepEmbedding, timestep_embedding


def test_timestep_embedding_function():
    """Test basic timestep embedding function."""
    # Test with different timesteps
    t = jnp.array(0.5)
    dim = 256
    
    emb = timestep_embedding(t, dim)
    
    # Check shape
    assert emb.shape == (dim,)
    
    # Check finite values
    assert jnp.all(jnp.isfinite(emb))
    
    # Test different timesteps give different embeddings
    t1 = jnp.array(0.0)
    t2 = jnp.array(1.0)
    emb1 = timestep_embedding(t1, dim)
    emb2 = timestep_embedding(t2, dim)
    
    assert not jnp.allclose(emb1, emb2)


def test_timestep_embedding_module():
    """Test TimestepEmbedding module."""
    key = jax.random.PRNGKey(42)
    
    # Create module
    embed = TimestepEmbedding(dim_embed=256, dim_out=1024, key=key)
    
    # Test forward pass
    t = jnp.array(0.5)
    output = embed(t)
    
    # Check shape
    assert output.shape == (1024,)
    
    # Check finite values
    assert jnp.all(jnp.isfinite(output))


def test_timestep_embedding_different_timesteps():
    """Test TimestepEmbedding with different timesteps."""
    key = jax.random.PRNGKey(42)
    
    # Create module
    embed = TimestepEmbedding(dim_embed=128, dim_out=512, key=key)
    
    # Test with different timesteps
    timesteps = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    for t in timesteps:
        output = embed(t)
        assert output.shape == (512,)
        assert jnp.all(jnp.isfinite(output))


def test_timestep_embedding_batched():
    """Test TimestepEmbedding with batched timesteps using vmap."""
    key = jax.random.PRNGKey(42)
    
    # Create module
    embed = TimestepEmbedding(dim_embed=256, dim_out=1024, key=key)
    
    # Batched forward pass
    batch_size = 4
    timesteps = jnp.linspace(0.0, 1.0, batch_size)
    
    batched_embed = jax.vmap(embed)
    outputs = batched_embed(timesteps)
    
    # Check shape
    assert outputs.shape == (batch_size, 1024)
    
    # Check finite values
    assert jnp.all(jnp.isfinite(outputs))
    
    # Check that different timesteps give different outputs
    assert not jnp.allclose(outputs[0], outputs[1])
    assert not jnp.allclose(outputs[0], outputs[3])


def test_timestep_embedding_gradient_flow():
    """Test that gradients flow through TimestepEmbedding."""
    key = jax.random.PRNGKey(42)
    
    # Create module
    embed = TimestepEmbedding(dim_embed=128, dim_out=256, key=key)
    
    # Test timestep
    t = jnp.array(0.5)
    
    def loss_fn(model, t):
        output = model(t)
        return jnp.mean(output**2)
    
    # Compute gradients
    grad_fn = eqx.filter_grad(loss_fn)
    grads = grad_fn(embed, t)
    
    # Check that gradients exist and are non-zero
    assert grads.linear1.weight is not None
    assert grads.linear2.weight is not None
    assert not jnp.allclose(grads.linear1.weight, 0.0)
    assert not jnp.allclose(grads.linear2.weight, 0.0)


def test_timestep_embedding_sinusoidal_properties():
    """Test sinusoidal embedding properties."""
    dim = 256
    
    # Test that sinusoidal embeddings are smooth
    t_values = jnp.linspace(0.0, 1.0, 11)
    embeddings = jax.vmap(lambda t: timestep_embedding(t, dim))(t_values)
    
    # Check shape
    assert embeddings.shape == (11, dim)
    
    # Check that neighboring embeddings are similar
    for i in range(len(t_values) - 1):
        diff = jnp.abs(embeddings[i+1] - embeddings[i]).mean()
        # Neighboring embeddings should be relatively close
        assert diff < 1.0


if __name__ == "__main__":
    test_timestep_embedding_function()
    test_timestep_embedding_module()
    test_timestep_embedding_different_timesteps()
    test_timestep_embedding_batched()
    test_timestep_embedding_gradient_flow()
    test_timestep_embedding_sinusoidal_properties()
    print("All timestep embedding tests passed!")



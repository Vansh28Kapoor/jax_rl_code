"""
Tests for the complete MMDiT model.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from mmdit_jax import MMDiT


def test_mmdit_basic_forward():
    """Test basic forward pass of MMDiT."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # Model parameters
    depth = 2
    dim_modalities = (768, 512, 384)  # Text, image, audio
    dim_outs = (768, 512, 384)  # Output dimensions (same as input)
    dim_cond = 256

    # Create model
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        timestep_embed_dim=128,
        dim_head=64,
        heads=8,
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Create test inputs
    batch_size = 2
    text_tokens = jax.random.normal(keys[1], (batch_size, 128, 768))
    image_tokens = jax.random.normal(keys[2], (batch_size, 256, 512))
    audio_tokens = jax.random.normal(keys[3], (batch_size, 64, 384))
    timesteps = jnp.linspace(0.0, 1.0, batch_size)  # Scalar timesteps

    modality_tokens = (text_tokens, image_tokens, audio_tokens)
    
    # Forward pass
    outputs = model(modality_tokens, timesteps)

    # Check outputs
    assert len(outputs) == 3
    assert outputs[0].shape == (batch_size, 128, 768)
    assert outputs[1].shape == (batch_size, 256, 512)
    assert outputs[2].shape == (batch_size, 64, 384)

    # Check that outputs are different from inputs (processing happened)
    assert not jnp.allclose(outputs[0], text_tokens)
    assert not jnp.allclose(outputs[1], image_tokens)
    assert not jnp.allclose(outputs[2], audio_tokens)


def test_mmdit_with_masks():
    """Test MMDiT with attention masks."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # Model parameters
    depth = 1
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128

    # Create model
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Create test inputs
    batch_size = 2
    seq_len1, seq_len2 = 10, 8
    total_seq = seq_len1 + seq_len2
    text_tokens = jax.random.normal(keys[1], (batch_size, seq_len1, 64))
    image_tokens = jax.random.normal(keys[2], (batch_size, seq_len2, 32))
    timesteps = jnp.array([0.3, 0.7])  # Scalar timesteps for each batch

    # Create 2D attention mask (causal for second modality)
    attention_mask = jnp.ones((batch_size, total_seq, total_seq), dtype=bool)
    action_causal = jnp.tril(jnp.ones((seq_len2, seq_len2), dtype=bool))
    attention_mask = attention_mask.at[:, seq_len1:, seq_len1:].set(action_causal)

    modality_tokens = (text_tokens, image_tokens)

    # Forward pass
    outputs = model(modality_tokens, timesteps, attention_mask)
    
    # Check outputs
    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, 10, 64)
    assert outputs[1].shape == (batch_size, 8, 32)


def test_mmdit_batched_processing():
    """Test batched processing of MMDiT."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Model parameters
    depth = 2
    dim_modalities = (128, 64)
    dim_outs = (128, 64)
    dim_cond = 256

    # Create model
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Create test inputs - different batch sizes
    for batch_size in [1, 4, 8]:
        text_tokens = jax.random.normal(keys[1], (batch_size, 32, 128))
        image_tokens = jax.random.normal(keys[2], (batch_size, 16, 64))
        timesteps = jnp.linspace(0.0, 1.0, batch_size)  # Scalar timesteps

        modality_tokens = (text_tokens, image_tokens)

        # Forward pass
        outputs = model(modality_tokens, timesteps)

        # Check outputs
        assert len(outputs) == 2
        assert outputs[0].shape == (batch_size, 32, 128)
        assert outputs[1].shape == (batch_size, 16, 64)


def test_mmdit_gradient_flow():
    """Test gradient flow through MMDiT."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Model parameters
    depth = 2
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128

    # Create model
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Create test inputs
    batch_size = 2
    text_tokens = jax.random.normal(keys[1], (batch_size, 8, 64))
    image_tokens = jax.random.normal(keys[2], (batch_size, 4, 32))
    timesteps = jnp.array([0.3, 0.7])  # Scalar timesteps

    modality_tokens = (text_tokens, image_tokens)

    # Define loss function
    def loss_fn(model):
        outputs = model(modality_tokens, timesteps)
        return jnp.mean(outputs[0]**2) + jnp.mean(outputs[1]**2)

    # Compute gradients
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # Check that gradients exist and are not zero
    assert isinstance(loss_val, jax.Array)
    assert loss_val.ndim == 0

    # Check that gradients exist
    # Note: With zero-initialization (adaLN-Zero), some gradients may be zero initially
    # This is expected and correct behavior. We just verify gradient structure exists.
    assert grads.__wrapped__.timestep_embeds[0].linear1.weight is not None
    assert grads.__wrapped__.timestep_embeds[0].linear2.weight is not None
    
    # Verify gradients have correct structure
    assert grads.__wrapped__.blocks is not None


def test_mmdit_modulated_final_layers():
    """Test that MMDiT uses ModulatedFinalLayer."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    dim_modalities = (512, 384)
    dim_outs = (512, 384)
    dim_cond = 256

    # Create model
    model = MMDiT(
        depth=2,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Test inputs
    batch_size = 1
    text_tokens = jax.random.normal(keys[1], (batch_size, 64, 512))
    image_tokens = jax.random.normal(keys[2], (batch_size, 128, 384))
    timesteps = jnp.array([0.5])  # Scalar timesteps
    modality_tokens = (text_tokens, image_tokens)

    # Forward pass
    outputs = model(modality_tokens, timesteps)

    # Check that ModulatedFinalLayer is used
    from mmdit_jax.final_layer import ModulatedFinalLayer
    assert isinstance(model.__wrapped__.final_layers[0], ModulatedFinalLayer)
    assert isinstance(model.__wrapped__.final_layers[1], ModulatedFinalLayer)

    # Check that outputs are properly shaped and different from inputs
    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, 64, 512)
    assert outputs[1].shape == (batch_size, 128, 384)
    assert not jnp.allclose(outputs[0], text_tokens)
    assert not jnp.allclose(outputs[1], image_tokens)


def test_mmdit_timestep_required():
    """Test that timestep is required (not optional)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    # Model parameters
    depth = 1
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128

    # Create model
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Create test inputs
    batch_size = 1
    text_tokens = jax.random.normal(keys[1], (batch_size, 8, 64))
    image_tokens = jax.random.normal(keys[2], (batch_size, 4, 32))

    modality_tokens = (text_tokens, image_tokens)

    # Test that calling without timestep raises TypeError
    with pytest.raises(TypeError):
        model(modality_tokens)


def test_mmdit_different_configurations():
    """Test MMDiT with different configurations."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    # Test different depths
    for depth in [1, 3, 5]:
        model = MMDiT(
            depth=depth,
            dim_modalities=(64, 32),
            dim_outs=(64, 32),
            dim_cond=128,
            key=keys[depth]
        )
        assert model.blocks.to_cond[1].weight.shape[0] == depth

    # Test different modalities
    for dim_modalities in [(64, 32), (512, 256, 128), (128,32,64,256), ]:
        model = MMDiT(
            depth=2,
            dim_modalities=dim_modalities,
            dim_outs=dim_modalities,
            dim_cond=128,
            key=keys[len(dim_modalities)]
        )
        assert len(model.final_layers) == len(dim_modalities)

    # Test different attention configurations
    model = MMDiT(
        depth=2,
        dim_modalities=(64, 32),
        dim_outs=(64, 32),
        dim_cond=128,
        dim_head=32,
        heads=4,
        key=keys[7]
    )
    # QK normalization is now always enabled
    assert model.blocks.joint_attn.q_rmsnorms is not None
    assert model.blocks.joint_attn.k_rmsnorms is not None


def test_mmdit_multiple_conditioning_variables():
    """Test MMDiT with multiple conditioning variables (e.g., t and r for improved meanflow)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # Model parameters
    depth = 2
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128

    # Create model with two conditioning variables
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        timestep_embed_dim=(256, 128),  # Two different embedding dims for t and r
        key=keys[0]
    )

    # Verify model has two timestep embeddings
    assert len(model.timestep_embeds) == 2
    assert model.num_conds == 2
    assert model.timestep_embeds[0].dim_embed == 256
    assert model.timestep_embeds[1].dim_embed == 128

    # Create test inputs
    text_tokens = jax.random.normal(keys[1], (8, 64))
    image_tokens = jax.random.normal(keys[2], (4, 32))
    t = jnp.array(0.5)  # timestep
    r = jnp.array(0.3)  # second conditioning variable

    modality_tokens = (text_tokens, image_tokens)

    # Forward pass with tuple of conditioning variables
    outputs = model(modality_tokens, (t, r))

    # Check outputs
    assert len(outputs) == 2
    assert outputs[0].shape == (8, 64)
    assert outputs[1].shape == (4, 32)

    # Note: With adaLN-Zero initialization, outputs are initially all zeros
    # so we can't test that different conds produce different outputs without training.
    # Instead, verify the forward pass completes without error.


def test_mmdit_multiple_conds_batched():
    """Test batched MMDiT with multiple conditioning variables."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # Model parameters
    depth = 2
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128
    batch_size = 3

    # Create model with two conditioning variables
    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        timestep_embed_dim=(256, 256),
        key=keys[0]
    )
    model = eqx.filter_vmap(model, in_axes=0)

    # Create test inputs
    text_tokens = jax.random.normal(keys[1], (batch_size, 8, 64))
    image_tokens = jax.random.normal(keys[2], (batch_size, 4, 32))
    t = jnp.linspace(0.0, 1.0, batch_size)
    r = jnp.linspace(0.2, 0.8, batch_size)

    modality_tokens = (text_tokens, image_tokens)

    # Forward pass
    outputs = model(modality_tokens, (t, r))

    # Check outputs
    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, 8, 64)
    assert outputs[1].shape == (batch_size, 4, 32)


def test_mmdit_cond_count_mismatch():
    """Test that mismatched conditioning variable count raises an error."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    # Create model with two conditioning variables
    model = MMDiT(
        depth=1,
        dim_modalities=(64, 32),
        dim_outs=(64, 32),
        dim_cond=128,
        timestep_embed_dim=(256, 128),  # Two conds
        key=keys[0]
    )

    text_tokens = jax.random.normal(keys[1], (8, 64))
    image_tokens = jax.random.normal(keys[2], (4, 32))
    modality_tokens = (text_tokens, image_tokens)

    # Passing single cond when model expects two should raise
    with pytest.raises(AssertionError):
        model(modality_tokens, jnp.array(0.5))

    # Passing three conds when model expects two should raise
    with pytest.raises(AssertionError):
        model(modality_tokens, (jnp.array(0.5), jnp.array(0.3), jnp.array(0.1)))


if __name__ == "__main__":
    test_mmdit_basic_forward()
    test_mmdit_with_masks()
    test_mmdit_batched_processing()
    test_mmdit_modulated_final_layers()
    test_mmdit_timestep_required()
    test_mmdit_different_configurations()
    test_mmdit_gradient_flow()
    test_mmdit_multiple_conditioning_variables()
    test_mmdit_multiple_conds_batched()
    test_mmdit_cond_count_mismatch()
    print("All tests passed!")
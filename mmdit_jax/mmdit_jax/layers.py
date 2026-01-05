"""
Basic layers for the mmdit library.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange
from jaxtyping import PRNGKeyArray, Array
import math

from mmdit_jax.helpers import exists


def timestep_embedding(t: Array, dim: int, max_period: int = 10000) -> Array:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        t: Timestep(s) - shape () or (,) for single timestep
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding
        
    Returns:
        Sinusoidal embedding of shape (dim,)
    """
    # Ensure t is a scalar or 0-d array
    t = jnp.atleast_1d(t).squeeze()
    
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = t * freqs
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    
    # If dim is odd, pad with a zero
    if dim % 2 == 1:
        embedding = jnp.concatenate([embedding, jnp.zeros(1)], axis=-1)
    
    return embedding


class TimestepEmbedding(eqx.Module):
    """
    Timestep embedding module that converts scalar timesteps to high-dimensional embeddings.
    
    Uses sinusoidal encoding followed by an MLP: t -> SinEmbed -> Linear -> SiLU -> Linear
    This is the standard approach in modern diffusion transformers.
    """
    
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dim_embed: int
    dim_out: int
    
    def __init__(
        self,
        dim_embed: int = 256,
        dim_out: int = 1024,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize TimestepEmbedding.
        
        Args:
            dim_embed: Dimension of sinusoidal embedding (default: 256)
            dim_out: Output dimension after MLP (default: 1024)
            key: Random key for initialization
        """
        key1, key2 = jax.random.split(key, 2)
        
        self.dim_embed = dim_embed
        self.dim_out = dim_out
        
        self.linear1 = eqx.nn.Linear(dim_embed, dim_out, key=key1)
        self.linear2 = eqx.nn.Linear(dim_out, dim_out, key=key2)
    
    def __call__(self, t: Array) -> Array:
        """
        Embed a timestep.
        
        Args:
            t: Timestep value (scalar or 0-d array)
            
        Returns:
            Embedded timestep of shape (dim_out,)
        """
        # Get sinusoidal embedding
        t_emb = timestep_embedding(t, self.dim_embed)
        
        # Apply MLP: Linear -> SiLU -> Linear
        t_emb = self.linear1(t_emb)
        t_emb = jax.nn.silu(t_emb)
        t_emb = self.linear2(t_emb)
        
        return t_emb


class AdaptiveLayerNorm(eqx.Module):
    """
    Adaptive Layer Normalization with modulation-based conditioning.

    Following SD3.5, this uses shift+scale modulation:
    output = LayerNorm(x) * (1 + scale) + shift

    where shift and scale are predicted from the conditioning signal.
    """

    ln: eqx.nn.LayerNorm
    to_modulation: eqx.nn.Sequential
    dim: int

    def __init__(
        self,
        dim: int,
        dim_cond: int,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize AdaptiveLayerNorm.

        Args:
            dim: Feature dimension
            dim_cond: Conditioning dimension
            key: Random key for initialization
        """
        self.dim = dim
        # Layer norm without learnable parameters
        self.ln = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)

        # Linear layer to produce shift and scale from conditioning
        linear_layer = eqx.nn.Linear(dim_cond, 2 * dim, key=key)

        # CRITICAL FIX: Do NOT zero-initialize the weights!
        # Zero weights block ALL gradients through the normalization path.
        # Instead: keep normal weight init, but bias to small negative value
        # so that silu(W@x + b) starts near zero while still allowing gradients.
        linear_layer = eqx.tree_at(
            lambda m: m.bias,
            linear_layer,
            jnp.full_like(linear_layer.bias, -1.0) if linear_layer.bias is not None else None
        )

        self.to_modulation = eqx.nn.Sequential([
            eqx.nn.Lambda(jax.nn.silu),
            linear_layer,
        ])

    def __call__(
        self,
        x: Array,
        cond: Array,
    ) -> Array:
        """
        Forward pass of adaptive layer norm with modulation.

        Args:
            x: Input tensor of shape (seq, dim)
            cond: Conditioning tensor of shape (dim_cond,)

        Returns:
            Modulated normalized tensor of shape (seq, dim)
        """
        # Apply layer norm
        x = jax.vmap(self.ln)(x)

        # Get shift and scale from conditioning
        modulation = self.to_modulation(cond)  # (2 * dim,)
        shift, scale = jnp.split(modulation, 2, axis=-1)  # (dim,), (dim,)
        shift = shift.reshape(1, -1)  # (1, dim)
        scale = scale.reshape(1, -1)  # (1, dim)

        # Apply modulation: x * (1 + scale) + shift
        x = x * (1.0 + scale) + shift

        return x

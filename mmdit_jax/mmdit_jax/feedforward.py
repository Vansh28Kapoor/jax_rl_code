"""
Feedforward layers for the mmdit library.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array


class FeedForward(eqx.Module):
    """
    Feedforward layer with SwiGLU activation.

    This module processes individual examples (no batch dimension) and should be
    used with eqx.filter_vmap for batched processing.
    
    Uses SwiGLU (Swish-Gated Linear Unit) activation, which is the standard
    in modern diffusion transformers (SD3, SD3.5, etc).
    SwiGLU splits the hidden layer: out = Linear2(Linear1(x) * swish(Gate(x)))
    """

    linear_in: eqx.nn.Linear
    linear_gate: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    dim: int
    mult: int

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize FeedForward layer with SwiGLU.

        Args:
            dim: Input/output dimension
            mult: Hidden dimension multiplier (default: 4)
            key: Random key for initialization
        """
        key0, key1, key2 = jax.random.split(key, 3)

        self.dim = dim
        self.mult = mult
        dim_hidden = dim * mult
        
        # SwiGLU uses two parallel projections for gating
        self.linear_in = eqx.nn.Linear(dim, dim_hidden, use_bias=False, key=key0)
        self.linear_gate = eqx.nn.Linear(dim, dim_hidden, use_bias=False, key=key1)
        self.linear_out = eqx.nn.Linear(dim_hidden, dim, use_bias=False, key=key2)

    def __call__(self, x: Array) -> Array:
        """
        Forward pass of feedforward layer with SwiGLU.

        Args:
            x: Input tensor of shape (seq_len, dim)

        Returns:
            Output tensor of shape (seq_len, dim)
        """
        # SwiGLU: out = Linear_out(Linear_in(x) * swish(Linear_gate(x)))
        # Process sequence dimension with vmap
        def swiglu_fn(x_single):
            x_in = self.linear_in(x_single)
            x_gate = self.linear_gate(x_single)
            x_gated = x_in * jax.nn.swish(x_gate)  # Element-wise gating with swish
            return self.linear_out(x_gated)
        
        return jax.vmap(swiglu_fn)(x)





 
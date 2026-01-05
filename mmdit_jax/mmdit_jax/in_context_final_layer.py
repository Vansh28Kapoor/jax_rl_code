"""
Simple final layer for In-Context MMDiT (no modulation).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array


class RMSNorm(eqx.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes by the RMS of the input, without mean centering.
    This is more efficient than LayerNorm and is used in modern LLMs
    (LLaMA, Mistral, etc.).

    Formula: x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x²) + eps)
    """

    gamma: Array
    dim: int
    eps: float

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        key: PRNGKeyArray = None,
    ):
        """
        Initialize RMSNorm.

        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
            key: Random key (unused, for API consistency)
        """
        self.dim = dim
        self.eps = eps
        self.gamma = jnp.ones((dim,))

    def __call__(self, x: Array) -> Array:
        """
        Forward pass of RMSNorm.

        Args:
            x: Input tensor of shape (dim,)

        Returns:
            Normalized tensor of shape (dim,)
        """
        rms = jnp.sqrt(jnp.mean(x**2) + self.eps)
        return (x / rms) * self.gamma


class SimpleFinalLayer(eqx.Module):
    """
    Simple final layer for In-Context MMDiT.

    Uses RMSNorm followed by a linear projection.
    Unlike ModulatedFinalLayer, this has no conditioning/modulation.
    The linear layer is zero-initialized for stable training.
    """

    rms_norm: RMSNorm
    linear_out: eqx.nn.Linear
    dim_in: int
    dim_out: int

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize SimpleFinalLayer.

        Args:
            dim_in: Input dimension
            dim_out: Output dimension
            key: Random key for initialization
        """
        self.dim_in = dim_in
        self.dim_out = dim_out

        # RMSNorm (learnable gamma)
        self.rms_norm = RMSNorm(dim_in)

        # Output projection - zero initialized for stable training
        self.linear_out = eqx.nn.Linear(dim_in, dim_out, key=key)
        self.linear_out = eqx.tree_at(
            lambda m: m.weight,
            self.linear_out,
            jnp.zeros_like(self.linear_out.weight)
        )
        self.linear_out = eqx.tree_at(
            lambda m: m.bias,
            self.linear_out,
            jnp.zeros_like(self.linear_out.bias)
        )

    def __call__(self, x: Array) -> Array:
        """
        Forward pass of simple final layer.

        Args:
            x: Input tensor of shape (seq, dim_in)

        Returns:
            Output tensor of shape (seq, dim_out)
        """
        # Apply RMSNorm to each token
        x = jax.vmap(self.rms_norm)(x)

        # Apply output projection
        x = jax.vmap(self.linear_out)(x)

        return x

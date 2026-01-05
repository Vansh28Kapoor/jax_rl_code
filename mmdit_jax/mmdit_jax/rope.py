"""
Rotary Position Embedding (RoPE) for attention.

RoPE encodes position information by rotating query and key vectors,
allowing the attention dot product to naturally encode relative positions.
"""

import jax.numpy as jnp
from jaxtyping import Array


def rotate_half(x: Array) -> Array:
    """
    Rotate half the dimensions for RoPE.

    Rearranges [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    This is the standard rotation pattern for RoPE.

    Args:
        x: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor of same shape
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """
    Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor of shape (..., seq_len, dim)
        cos: Cosine frequencies of shape (seq_len, dim)
        sin: Sine frequencies of shape (seq_len, dim)

    Returns:
        Rotated tensor of same shape as x
    """
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_frequencies(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
) -> tuple[Array, Array]:
    """
    Build cosine and sine frequency matrices for RoPE.

    The frequencies follow the standard RoPE formula:
        θ_i = 1 / (base^(2i/dim)) for i in [0, dim/2)

    Args:
        seq_len: Sequence length
        dim: Head dimension (must be even)
        base: Base for frequency computation (default: 10000.0)

    Returns:
        Tuple of (cos, sin) each of shape (seq_len, dim)
    """
    # Compute inverse frequencies: (dim/2,)
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    # Compute position indices: (seq_len,)
    positions = jnp.arange(seq_len, dtype=jnp.float32)

    # Outer product to get angles: (seq_len, dim/2)
    freqs = jnp.outer(positions, inv_freq)

    # Duplicate frequencies for pairing: (seq_len, dim)
    freqs = jnp.concatenate([freqs, freqs], axis=-1)

    return jnp.cos(freqs), jnp.sin(freqs)

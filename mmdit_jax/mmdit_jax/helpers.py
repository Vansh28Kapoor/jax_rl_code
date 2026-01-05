"""
Utility functions for the mmdit library.
"""

from typing import Optional, Any
import jax
import jax.numpy as jnp


def exists(v: Any) -> bool:
    """Check if a value exists (is not None)."""
    return v is not None


def default(v: Any, d: Any) -> Any:
    """Return v if it exists, otherwise return d."""
    return v if exists(v) else d


def dot_product_attention_compat(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
) -> jnp.ndarray:
    """
    Compatibility wrapper for scaled dot-product attention.
    
    For JAX >= 0.4.25, uses jax.nn.dot_product_attention.
    For older versions (like 0.4.13), implements the official scaled dot-product 
    attention algorithm from "Attention is All You Need" (Vaswani et al., 2017).
    
    This implementation is based on the official JAX implementation:
    https://github.com/google/jax/blob/main/jax/_src/nn/functions.py
    
    Args:
        query: Query array of shape (..., seq_len, num_heads, head_dim)
        key: Key array of shape (..., seq_len, num_heads, head_dim)
        value: Value array of shape (..., seq_len, num_heads, head_dim)
        bias: Optional bias array broadcastable to (..., num_heads, seq_len, seq_len)
        mask: Optional boolean mask of shape broadcastable to (..., num_heads, seq_len, seq_len).
              True means the query can attend to the key.
        scale: Optional scale factor. If None, uses 1/sqrt(head_dim).
    
    Returns:
        Output of shape (..., seq_len, num_heads, head_dim)
    """
    # Try to use the built-in function if available
    if hasattr(jax.nn, 'dot_product_attention'):
        return jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            bias=bias,
            mask=mask,
            scale=scale,
        )
    
    # Fallback implementation based on official JAX source
    # Reference: https://github.com/google/jax/blob/main/jax/_src/nn/functions.py
    
    # query, key, value: (..., seq_len, num_heads, head_dim)
    head_dim = query.shape[-1]
    
    # Determine scale factor (default: 1/sqrt(head_dim))
    if scale is None:
        scale = jnp.asarray(head_dim, dtype=query.dtype)
        scale = jnp.reciprocal(jnp.sqrt(scale))
    else:
        scale = jnp.asarray(scale, dtype=query.dtype)
    
    # Compute attention logits: Q @ K^T
    # einsum notation: batch..., query_seq, heads, dim x batch..., key_seq, heads, dim
    #                  -> batch..., heads, query_seq, key_seq
    attn_logits = jnp.einsum('...qhd,...khd->...hqk', query, key)
    
    # Scale the logits
    attn_logits = attn_logits * scale
    
    # Apply bias if provided
    if bias is not None:
        attn_logits = attn_logits + jnp.asarray(bias, dtype=attn_logits.dtype)
    
    # Apply mask if provided
    # In JAX convention: True means attend, False means mask out
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
        neg = jnp.finfo(attn_logits.dtype).min
        attn_logits = jnp.where(mask, attn_logits, neg)

        all_masked = ~jnp.any(mask, axis=-1, keepdims=True)
        attn_logits = jnp.where(all_masked, jnp.zeros_like(attn_logits), attn_logits)

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    if mask is not None:
        attn_weights = jnp.where(all_masked, jnp.zeros_like(attn_weights), attn_weights)
    

    # Compute attention output: weights @ V
    # einsum notation: batch..., heads, query_seq, key_seq x batch..., key_seq, heads, dim
    #                  -> batch..., query_seq, heads, dim
    attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
    
    return attn_output 
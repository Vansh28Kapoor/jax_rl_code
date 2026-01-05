"""
Joint attention mechanism for multi-modal transformer.
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array
from einops import rearrange, pack, unpack

from mmdit_jax.helpers import exists, default, dot_product_attention_compat
from mmdit_jax.rope import apply_rope, build_rope_frequencies


class MultiHeadRMSNorm(eqx.Module):
    """
    RMS normalization for multi-head attention (SD3.5 aligned).

    Applies RMS normalization to each attention head with learnable scale
    gamma SHARED across all heads. This matches the SD3.5 implementation
    where a single (dim,) gamma vector is broadcast across heads.

    The normalization is: (x / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x²) + eps)
    """

    gamma: Array
    dim: int
    eps: float

    def __init__(
        self,
        dim: int,
        heads: int = 1,  # kept for API compatibility but not used for gamma shape
        eps: float = 1e-6,
        *,
        key: PRNGKeyArray
    ):
        """
        Initialize MultiHeadRMSNorm.

        Args:
            dim: Feature dimension per head (head_dim)
            heads: Number of attention heads (kept for API compatibility)
            eps: Small constant for numerical stability (default: 1e-6, matching SD3.5)
            key: Random key for initialization (not used but kept for consistency)
        """
        self.dim = dim
        self.eps = eps
        # Shared gamma across all heads, shape (dim,) - matches SD3.5
        self.gamma = jnp.ones((dim,))

    def __call__(self, x: Array) -> Array:
        """
        Forward pass of MultiHeadRMSNorm.

        Args:
            x: Input tensor of shape (heads, seq_len, dim)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS along the last dimension (per head, per position)
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

        # Normalize and apply shared gamma (broadcasts across heads and seq)
        return (x / rms) * self.gamma


class JointAttention(eqx.Module):
    """
    Joint attention mechanism that handles multiple modalities.

    This attention mechanism processes multiple input modalities jointly,
    allowing cross-modal attention between different types of inputs.
    The module processes individual examples (no batch dimension) and should be
    used with eqx.filter_vmap for batched processing.
    
    Implements attention computation manually to handle multi-modal inputs.
    """

    to_qkv: Tuple[eqx.nn.Linear, ...]
    to_out: Tuple[eqx.nn.Linear, ...]
    q_rmsnorms: Tuple[MultiHeadRMSNorm, ...]
    k_rmsnorms: Tuple[MultiHeadRMSNorm, ...]

    dim_inputs: Tuple[int, ...]
    dim_head: int
    heads: int

    def __init__(
        self,
        *,
        dim_inputs: Tuple[int, ...],
        dim_head: int = 64,
        heads: int = 8,
        key: PRNGKeyArray,
    ):
        """
        Initialize Joint Attention.

        Args:
            dim_inputs: Tuple of input dimensions for each modality
            dim_head: Dimension of each attention head (default: 64)
            heads: Number of attention heads (default: 8)
            key: Random key for initialization
            
        Note:
            QK RMS normalization is now mandatory (following SD3.5 architecture)
        """
        # Validate inputs
        assert len(dim_inputs) > 0, "At least one input modality is required"
        assert dim_head > 0, "dim_head must be positive"
        assert heads > 0, "heads must be positive"

        num_inputs = len(dim_inputs)
        # Always allocate keys for QK RMS norm
        keys = jax.random.split(key, num_inputs * 4)
        key_idx = 0

        self.dim_inputs = dim_inputs
        self.dim_head = dim_head
        self.heads = heads

        dim_inner = dim_head * heads

        # Linear projections for QKV for each modality
        self.to_qkv = tuple(
            eqx.nn.Linear(dim_input, dim_inner * 3, use_bias=False, key=keys[key_idx + i])
            for i, dim_input in enumerate(dim_inputs)
        )
        key_idx += num_inputs

        # Output projections for each modality
        self.to_out = tuple(
            eqx.nn.Linear(dim_inner, dim_input, use_bias=False, key=keys[key_idx + i])
            for i, dim_input in enumerate(dim_inputs)
        )
        key_idx += num_inputs

        # QK RMS normalization (now mandatory following SD3.5)
        self.q_rmsnorms = tuple(
            MultiHeadRMSNorm(dim_head, heads=heads, key=keys[key_idx + i])
            for i in range(num_inputs)
        )
        key_idx += num_inputs

        self.k_rmsnorms = tuple(
            MultiHeadRMSNorm(dim_head, heads=heads, key=keys[key_idx + i])
            for i in range(num_inputs)
        )

    def __call__(
        self,
        inputs: Tuple[Array, ...],
        attention_mask: Optional[Array] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[Array, ...]:
        """
        Forward pass of joint attention.

        Args:
            inputs: Tuple of input tensors for each modality,
                   each of shape (seq_len, dim_input)
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                  where total_seq is sum of all modality sequence lengths.
                  True indicates the query position can attend to the key position.
                  If None, full bidirectional attention is used (all True).

        Returns:
            Tuple of output tensors for each modality,
            each of shape (seq_len, dim_input)
        """
        assert len(inputs) == len(self.dim_inputs), \
            f"Expected {len(self.dim_inputs)} inputs, got {len(inputs)}"

        # Project each modality separately for qkv
        all_qkvs = []

        for i, (x, to_qkv) in enumerate(zip(inputs, self.to_qkv)):
            # Validate input shape
            seq_len, dim_input = x.shape
            assert dim_input == self.dim_inputs[i], \
                f"Input {i} has dimension {dim_input}, expected {self.dim_inputs[i]}"

            # Project to QKV
            qkv = jax.vmap(to_qkv)(x)  # (seq_len, dim_inner * 3)

            # Split heads: (seq_len, dim_inner * 3) -> (3, heads, seq_len, dim_head)
            qkv = rearrange(qkv, 's (qkv h d) -> qkv h s d', qkv = 3, h = self.heads)

            # Apply QK RMS normalization per modality (mandatory)
            q, k, v = qkv
            q = self.q_rmsnorms[i](q)
            k = self.k_rmsnorms[i](k)

            # Apply RoPE to queries and keys (per-modality positions: 0, 1, 2, ...)
            cos, sin = build_rope_frequencies(seq_len, self.dim_head)
            q = apply_rope(q, cos, sin)  # (heads, seq_len, dim_head)
            k = apply_rope(k, cos, sin)

            qkv = jnp.stack([q, k, v])

            all_qkvs.append(qkv)

        # Combine all qkv
        all_qkvs, packed_shape = pack(all_qkvs, 'qkv h * d')  # (3, heads, total_seq, dim_head)
        total_seq = all_qkvs.shape[2]

        # Validate attention mask if provided
        if attention_mask is not None:
            assert attention_mask.shape == (total_seq, total_seq), \
                f"attention_mask has shape {attention_mask.shape}, expected ({total_seq}, {total_seq})"

        # Attention
        q, k, v = all_qkvs  # Each is (heads, total_seq, dim_head)

        # Convert to format expected by jax.nn.dot_product_attention: (seq_len, num_heads, head_dim)
        q = rearrange(q, 'h s d -> s h d')  # (total_seq, heads, dim_head)
        k = rearrange(k, 'h s d -> s h d')  # (total_seq, heads, dim_head)
        v = rearrange(v, 'h s d -> s h d')  # (total_seq, heads, dim_head)

        # Apply optimized attention (with compatibility for JAX 0.4.13+)
        # Mask shape for dot_product_attention without batch dim: (1, T, S) or (T, S) broadcastable
        out = dot_product_attention_compat(
            query=q,
            key=k,
            value=v,
            mask=attention_mask,  # (total_seq, total_seq) or None
        )  # (total_seq, heads, dim_head)

        # Merge heads back to (total_seq, heads * dim_head)
        outs = rearrange(out, 's h d -> s (h d)')
        outs = unpack(outs, packed_shape, '* d')  # list of (modality 0 seq, modality 1 seq, etc...)
        all_outs = [eqx.filter_vmap(to_out)(out) for out, to_out in zip(outs, self.to_out)]
        return tuple(all_outs)


 
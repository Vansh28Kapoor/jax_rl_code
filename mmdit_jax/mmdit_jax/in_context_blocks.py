"""
In-Context MMDiT Block implementation - transformer block without AdaLN modulation.

This block uses standard RMSNorm instead of Adaptive LayerNorm, and standard
residual connections without gating. Time conditioning is expected to be
provided as tokens in the input sequence (handled by the user externally).
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.attention import JointAttention
from mmdit_jax.feedforward import FeedForward
from mmdit_jax.in_context_final_layer import RMSNorm


class InContextMMDiTBlock(eqx.Module):
    """
    In-Context Multi-Modal Diffusion Transformer Block.

    Unlike the standard MMDiTBlock, this block:
    - Uses standard RMSNorm instead of AdaptiveLayerNorm (no modulation)
    - Uses standard residual connections (no gating)
    - Does not take time conditioning - conditioning is expected to be
      provided as tokens in the input sequence

    This follows the modern LLM transformer pattern (pre-norm + residual).
    """

    # Attention components
    attn_norms: Sequence[RMSNorm]
    joint_attn: JointAttention

    # Feedforward components
    ff_norms: Sequence[RMSNorm]
    feedforwards: Sequence[FeedForward]

    # Metadata
    num_modalities: int
    dim_modalities: Tuple[int, ...]

    def __init__(
        self,
        *,
        dim_modalities: Tuple[int, ...],
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        key: jax.random.PRNGKey,
    ):
        """
        Initialize InContextMMDiTBlock.

        Args:
            dim_modalities: Tuple of dimensions for each modality
            dim_head: Dimension of each attention head
            heads: Number of attention heads
            ff_mult: Multiplier for feedforward hidden dimension
            key: Random key for initialization
        """
        assert len(dim_modalities) > 0, 'must have at least one modality'
        keys = jax.random.split(key, len(dim_modalities) * 2 + 1)
        key_idx = 0

        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # Pre-attention RMSNorm for each modality
        self.attn_norms = [
            RMSNorm(dim)
            for dim in dim_modalities
        ]

        # Joint attention
        self.joint_attn = JointAttention(
            dim_inputs=dim_modalities,
            dim_head=dim_head,
            heads=heads,
            key=keys[key_idx],
        )
        key_idx += 1

        # Pre-feedforward RMSNorm for each modality
        self.ff_norms = [
            RMSNorm(dim)
            for dim in dim_modalities
        ]

        # Feedforward layers
        self.feedforwards = [
            FeedForward(dim, mult=ff_mult, key=keys[key_idx + i])
            for i, dim in enumerate(dim_modalities)
        ]

    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass of InContextMMDiTBlock.

        Args:
            modality_tokens: Tuple of token tensors for each modality,
                each of shape (seq_len, dim)
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of processed token tensors for each modality
        """
        assert len(modality_tokens) == self.num_modalities

        # ATTENTION BLOCK (Pre-norm + Residual)
        # Store residuals
        attn_residuals = modality_tokens

        # Apply RMSNorm (pre-norm)
        modality_tokens = tuple(
            jax.vmap(norm)(tokens)
            for tokens, norm in zip(modality_tokens, self.attn_norms)
        )

        # Apply joint attention
        modality_tokens = self.joint_attn(modality_tokens, attention_mask=attention_mask)

        # Add residual (no gating)
        modality_tokens = tuple(
            residual + tokens
            for tokens, residual in zip(modality_tokens, attn_residuals)
        )

        # FEEDFORWARD BLOCK (Pre-norm + Residual)
        # Store residuals
        ff_residuals = modality_tokens

        # Apply RMSNorm (pre-norm)
        modality_tokens = tuple(
            jax.vmap(norm)(tokens)
            for tokens, norm in zip(modality_tokens, self.ff_norms)
        )

        # Apply feedforward layers
        modality_tokens = tuple(
            ff(tokens)
            for tokens, ff in zip(modality_tokens, self.feedforwards)
        )

        # Add residual (no gating)
        modality_tokens = tuple(
            residual + tokens
            for tokens, residual in zip(modality_tokens, ff_residuals)
        )

        return modality_tokens

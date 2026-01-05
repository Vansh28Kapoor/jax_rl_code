"""
MMDiT Block implementation - the core transformer block for multi-modal diffusion.
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.helpers import exists, default
from mmdit_jax.layers import AdaptiveLayerNorm
from mmdit_jax.attention import JointAttention
from mmdit_jax.feedforward import FeedForward


class MMDiTBlock(eqx.Module):
    """
    Multi-Modal Diffusion Transformer Block.
    
    This block processes multiple modalities jointly using:
    1. Adaptive layer normalization (conditioned on time)
    2. Joint attention across modalities
    3. Feedforward layers for each modality
    4. Residual connections (ResNet-style)
    """
    
    # Attention components
    attn_layernorms: Sequence[AdaptiveLayerNorm]
    joint_attn: JointAttention
    
    # Feedforward components
    ff_layernorms: Sequence[AdaptiveLayerNorm]
    feedforwards: Sequence[FeedForward]
    
    # Conditioning
    to_cond: eqx.nn.Sequential
    
    # Metadata
    num_modalities: int
    dim_modalities: Tuple[int, ...]
    dim_cumsum: jax.Array

    def __init__(
        self,
        *,
        dim_modalities: Tuple[int, ...],
        dim_cond: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        key: jax.random.PRNGKey,
    ):
        """
        Initialize MMDiTBlock.
        
        Args:
            dim_modalities: Tuple of dimensions for each modality
            dim_cond: Conditioning dimension for flow matching timesteps
            dim_head: Dimension of each attention head
            heads: Number of attention heads
            ff_mult: Multiplier for feedforward hidden dimension
            key: Random key for initialization
            
        Note:
            QK RMS normalization is now mandatory (following SD3.5 architecture)
        """
        assert len(dim_modalities) > 0, 'must have at least one modality'
        keys = jax.random.split(key, len(dim_modalities) * 3 + 2)
        key_idx = 0

        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities
        # self.dim_cumsum = jnp.cumsum(jnp.array(dim_modalities))
        self.dim_cumsum = [dim_modalities[0]]
        for i in range(1, len(dim_modalities)):
            self.dim_cumsum.append(self.dim_cumsum[i - 1] + dim_modalities[i])

        # Attention layer norms (adaptive with conditioning)
        self.attn_layernorms = [
            AdaptiveLayerNorm(dim, dim_cond=dim_cond, key=keys[key_idx + i])
            for i, dim in enumerate(dim_modalities)
        ]
        key_idx += len(dim_modalities)

        # Joint attention (QK normalization is now mandatory)
        self.joint_attn = JointAttention(
            dim_inputs=dim_modalities,
            dim_head=dim_head,
            heads=heads,
            key=keys[key_idx],
        )
        key_idx += 1

        # Feedforward layer norms (adaptive with conditioning)
        self.ff_layernorms = [
            AdaptiveLayerNorm(dim, dim_cond=dim_cond, key=keys[key_idx + i])
            for i, dim in enumerate(dim_modalities)
        ]
        key_idx += len(dim_modalities)

        # Feedforward layers
        self.feedforwards = [
            FeedForward(dim, mult=ff_mult, key=keys[key_idx + i])
            for i, dim in enumerate(dim_modalities)
        ]
        key_idx += len(dim_modalities)

        # Conditioning linear layer - following SD3.5 modulation approach
        # Each modality gets 2 parameters: gate_msa (post-attention), gate_mlp (post-feedforward)
        # Note: scale parameters are produced by AdaptiveLayerNorm
        total_cond_dims = sum(dim_modalities) * 2  # 2 gates per modality

        linear_layer = eqx.nn.Linear(dim_cond, total_cond_dims, key=keys[key_idx])
        # Zero initialization for gates (adaLN-Zero)
        linear_layer = eqx.tree_at(
            lambda m: m.weight,
            linear_layer,
            jnp.zeros_like(linear_layer.weight)
        )
        # Initialize bias to 0 (so initial gate value is 0, making residual identity)
        linear_layer = eqx.tree_at(
            lambda m: m.bias,
            linear_layer,
            jnp.zeros_like(linear_layer.bias)
        )
        self.to_cond = eqx.nn.Sequential([
            eqx.nn.Lambda(jax.nn.silu),
            linear_layer,
        ])

    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        time_cond: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass of MMDiTBlock.

        Args:
            modality_tokens: Tuple of token tensors for each modality
            time_cond: Time conditioning tensor for flow matching timesteps
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of processed token tensors for each modality
        """
        assert len(modality_tokens) == self.num_modalities
        
        # Get gate parameters from conditioning (following SD3.5 modulation)
        gates = self.to_cond(time_cond)  # (2 * sum(dim_modalities),)
        # Split into attention and feedforward gates
        attn_gates, ff_gates = jnp.split(gates, 2, axis=-1)  # each (sum(dim_modalities),)

        # Split by modality dimensions
        attn_gate_splits = jnp.split(attn_gates, self.dim_cumsum, axis=-1)
        ff_gate_splits = jnp.split(ff_gates, self.dim_cumsum, axis=-1)

        # ATTENTION BLOCK WITH RESIDUAL CONNECTION
        # Store residuals for attention
        attn_residuals = modality_tokens

        # Apply adaptive layer norm with modulation (produces: LN(x) * (1 + scale))
        modality_tokens = tuple(
            ln(tokens, cond=time_cond) 
            for tokens, ln in zip(modality_tokens, self.attn_layernorms)
        )

        # Apply joint attention
        modality_tokens = self.joint_attn(modality_tokens, attention_mask=attention_mask)

        # Apply gate and add residual: residual + gate * attn_output
        # When gate=0 initially, this is identity (adaLN-Zero principle)
        modality_tokens = tuple(
            residual + gate.reshape(1, -1) * tokens
            for tokens, residual, gate in zip(modality_tokens, attn_residuals, attn_gate_splits)
        )

        # FEEDFORWARD BLOCK WITH RESIDUAL CONNECTION
        # Store residuals for feedforward
        ff_residuals = modality_tokens

        # Apply adaptive layer norm with modulation
        modality_tokens = tuple(
            ln(tokens, cond=time_cond)
            for tokens, ln in zip(modality_tokens, self.ff_layernorms)
        )

        # Apply feedforward layers
        modality_tokens = tuple(
            ff(tokens)
            for tokens, ff in zip(modality_tokens, self.feedforwards)
        )

        # Apply gate and add residual: residual + gate * ff_output
        modality_tokens = tuple(
            residual + gate.reshape(1, -1) * tokens
            for tokens, residual, gate in zip(modality_tokens, ff_residuals, ff_gate_splits)
        )

        return modality_tokens 
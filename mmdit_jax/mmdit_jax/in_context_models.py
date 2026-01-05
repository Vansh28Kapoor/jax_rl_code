"""
In-Context MMDiT model implementation.

This is a pure token-processing transformer without any conditioning logic.
Time conditioning should be handled externally by the user (e.g., prepending
time tokens to specific modalities). The model simply processes tokens through
joint attention and feedforward layers.
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx

from mmdit_jax.in_context_blocks import InContextMMDiTBlock
from mmdit_jax.in_context_final_layer import SimpleFinalLayer


class InContextMMDiT(eqx.Module):
    """
    In-Context Multi-Modal Diffusion Transformer.

    Unlike the standard MMDiT which uses AdaLN modulation for conditioning,
    this model is a pure token processor. Time conditioning should be
    handled externally by the user (e.g., by prepending time tokens to
    specific modalities before passing to this model).

    Key differences from MMDiT:
    - No timestep embedding or conditioning logic
    - Uses RMSNorm instead of AdaptiveLayerNorm
    - Uses standard residuals instead of gated residuals
    - SimpleFinalLayer instead of ModulatedFinalLayer

    Example usage for imitation learning:
        # User handles time embedding externally
        time_embed = TimestepEmbedding(dim_embed=256, dim_out=256, key=key1)
        time_to_token = eqx.nn.Linear(256, action_dim, key=key2)

        # Create time tokens
        time_cond = time_embed(t=0.5)
        time_token = time_to_token(time_cond)
        time_tokens = jnp.tile(time_token[None, :], (2, 1))  # 2 tokens

        # Prepend to actions (user's choice)
        action_with_time = jnp.concatenate([time_tokens, action_tokens], axis=0)

        # Model just processes tokens
        model = InContextMMDiT(dim_modalities=(obs_dim, action_dim), ...)
        obs_out, action_out = model((obs_tokens, action_with_time), attention_mask)

        # User strips time tokens from output
        action_out = action_out[2:]
    """

    blocks: Sequence[InContextMMDiTBlock]
    final_layers: Sequence[SimpleFinalLayer]
    dim_modalities: Tuple[int, ...]
    dim_outs: Tuple[int, ...]

    def __init__(
        self,
        *,
        depth: int,
        dim_modalities: Tuple[int, ...],
        dim_outs: Tuple[int, ...],
        key: jax.random.PRNGKey,
        **block_kwargs,
    ):
        """
        Initialize InContextMMDiT model.

        Args:
            depth: Number of transformer blocks
            dim_modalities: Tuple of dimensions for each modality
            dim_outs: Tuple of output dimensions for each modality
            key: Random key for initialization
            **block_kwargs: Additional arguments for InContextMMDiTBlock
                (e.g., dim_head, heads, ff_mult)
        """
        keys = jax.random.split(key, depth + len(dim_modalities))

        self.dim_modalities = dim_modalities
        self.dim_outs = dim_outs

        # Create transformer blocks using filter_vmap for efficient scanning
        make_blocks = lambda k: InContextMMDiTBlock(
            dim_modalities=dim_modalities,
            key=k,
            **block_kwargs,
        )
        self.blocks = eqx.filter_vmap(make_blocks)(keys[:depth])

        # Simple final layers (one per modality)
        self.final_layers = [
            SimpleFinalLayer(
                dim_in=dim,
                dim_out=dim_out,
                key=keys[depth + i]
            )
            for i, (dim, dim_out) in enumerate(zip(dim_modalities, dim_outs))
        ]

    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass through InContextMMDiT.

        Args:
            modality_tokens: Tuple of token tensors, each of shape (seq_len, dim).
                Time conditioning tokens (if any) should already be included
                in the appropriate modality sequences by the user.
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of output tensors for each modality
        """
        # Process through transformer blocks using scan
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        def scan_blocks(_tokens, _dynamic_block):
            block = eqx.combine(_dynamic_block, static_blocks)
            tokens = block(
                modality_tokens=_tokens,
                attention_mask=attention_mask,
            )
            return tokens, None

        modality_tokens, _ = jax.lax.scan(scan_blocks, modality_tokens, dynamic_blocks)

        # Apply final layers
        modality_tokens = tuple(
            final_layer(tokens)
            for tokens, final_layer in zip(modality_tokens, self.final_layers)
        )

        return modality_tokens

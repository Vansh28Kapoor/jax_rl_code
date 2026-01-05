"""
Main MMDiT model implementation.
"""

from typing import Tuple, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

from mmdit_jax.blocks import MMDiTBlock
from mmdit_jax.layers import TimestepEmbedding
from mmdit_jax.final_layer import ModulatedFinalLayer


class MMDiT(eqx.Module):
    """
    Multi-Modal Diffusion Transformer (SD3.5 architecture).
    
    This is the main model that stacks multiple MMDiTBlocks to create
    a deep transformer for processing multiple modalities with timestep conditioning.
    
    Key features:
    - Sinusoidal timestep embedding with MLP
    - Modulation-based conditioning throughout
    - Mandatory QK normalization in attention
    - SwiGLU activation in feedforward
    - Modulated final output layers
    """

    timestep_embeds: Sequence[TimestepEmbedding]
    blocks: Sequence[MMDiTBlock]
    final_layers: Sequence[ModulatedFinalLayer]
    dim_modalities: Tuple[int, ...]
    dim_outs: Tuple[int, ...]
    num_conds: int

    def __init__(
        self,
        *,
        depth: int,
        dim_modalities: Tuple[int, ...],
        dim_cond: int = 1024,
        dim_outs: Tuple[int, ...],
        timestep_embed_dim: Union[int, Tuple[int, ...]] = 256,
        key: jax.random.PRNGKey,
        **block_kwargs,
    ):
        """
        Initialize MMDiT model.
        
        Args:
            depth: Number of transformer blocks
            dim_modalities: Tuple of dimensions for each modality
            dim_cond: Conditioning dimension (output of timestep embedding, default: 1024)
            dim_out: Tuple of dimensions for each modality (output of final layers)
            timestep_embed_dim: Dimension(s) for sinusoidal timestep embedding (default: 256).
                Can be a single int or a tuple of ints for multiple conditioning variables.
                When multiple dims are provided, each conditioning variable gets its own
                TimestepEmbedding and their outputs are summed.
            key: Random key for initialization
            **block_kwargs: Additional arguments for MMDiTBlock (e.g., dim_head, heads, ff_mult)
        """
        # Normalize timestep_embed_dim to tuple
        if isinstance(timestep_embed_dim, int):
            timestep_embed_dim = (timestep_embed_dim,)

        self.num_conds = len(timestep_embed_dim)

        keys = jax.random.split(key, self.num_conds + depth + len(dim_modalities))

        self.dim_modalities = dim_modalities
        self.dim_outs = dim_outs

        # Timestep embedding modules (one per conditioning variable)
        self.timestep_embeds = [
            TimestepEmbedding(
                dim_embed=dim,
                dim_out=dim_cond,
                key=keys[i]
            )
            for i, dim in enumerate(timestep_embed_dim)
        ]
        
        # Create transformer blocks
        block_key_start = self.num_conds
        make_blocks = lambda k: MMDiTBlock(
            dim_modalities=dim_modalities,
            dim_cond=dim_cond,
            key=k,
            **block_kwargs,
        )
        self.blocks = eqx.filter_vmap(make_blocks)(keys[block_key_start:block_key_start + depth])

        # Modulated final layers (one per modality)
        final_key_start = block_key_start + depth
        self.final_layers = [
            ModulatedFinalLayer(
                dim_in=dim,
                dim_out=dim_out,
                dim_cond=dim_cond,
                key=keys[final_key_start + i]
            )
            for i, (dim, dim_out) in enumerate(zip(dim_modalities, dim_outs))
        ]


    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        timestep: Union[jax.Array, Tuple[jax.Array, ...]],
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass through MMDiT.

        Args:
            modality_tokens: Tuple of token tensors, each of shape (seq_len, dim)
            timestep: Scalar timestep value(s) for diffusion process. Can be a single
                scalar or a tuple of scalars matching the number of conditioning variables.
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of output tensors for each modality
        """
        # Normalize timestep to tuple
        if not isinstance(timestep, (tuple, list)):
            timestep = (timestep,)

        assert len(timestep) == self.num_conds, \
            f"Expected {self.num_conds} conditioning variables, got {len(timestep)}"

        # Embed each timestep and sum
        time_cond = sum(
            embed(t) for embed, t in zip(self.timestep_embeds, timestep)
        )  # (dim_cond,)

        # Process through transformer blocks
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        def scan_blocks(_tokens, _dynamic_block):
            block = eqx.combine(_dynamic_block, static_blocks)
            tokens = block(
                modality_tokens=_tokens,
                attention_mask=attention_mask,
                time_cond=time_cond,
            )
            return tokens, None

        modality_tokens, _ = jax.lax.scan(scan_blocks, modality_tokens, dynamic_blocks)

        # Apply modulated final layers
        modality_tokens = tuple(
            final_layer(tokens, time_cond)
            for tokens, final_layer in zip(modality_tokens, self.final_layers)
        )

        return modality_tokens

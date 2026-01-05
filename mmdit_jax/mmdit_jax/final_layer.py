"""
Final output layer for MMDiT with modulation.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array


class ModulatedFinalLayer(eqx.Module):
    """
    Modulated final layer for MMDiT output.

    Applies adaptive layer normalization followed by a linear projection,
    with both conditioned on the timestep embedding. This follows the SD3.5
    approach of using modulation-based conditioning throughout.

    The layer produces: LayerNorm(x) * (1 + scale) + shift -> Linear
    where shift and scale are predicted from the timestep conditioning.
    """

    ln: eqx.nn.LayerNorm
    adaLN_modulation: eqx.nn.Sequential
    linear_out: eqx.nn.Linear
    dim_in: int
    dim_out: int
    
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_cond: int,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize ModulatedFinalLayer.
        
        Args:
            dim_in: Input dimension
            dim_out: Output dimension
            dim_cond: Conditioning dimension
            key: Random key for initialization
        """
        key1, key2 = jax.random.split(key, 2)
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # Layer norm without learnable parameters
        self.ln = eqx.nn.LayerNorm(dim_in, use_weight=False, use_bias=False)
        
        # Conditioning MLP to produce shift and scale parameters
        linear_layer = eqx.nn.Linear(dim_cond, 2 * dim_in, key=key1)

        # Zero initialization for modulation (adaLN-Zero)
        linear_layer = eqx.tree_at(
            lambda m: m.weight,
            linear_layer,
            jnp.zeros_like(linear_layer.weight)
        )
        # Initialize bias to 0 (so initial shift=0, scale=0 -> modulation is identity)
        linear_layer = eqx.tree_at(
            lambda m: m.bias,
            linear_layer,
            jnp.zeros_like(linear_layer.bias)
        )

        self.adaLN_modulation = eqx.nn.Sequential([
            eqx.nn.Lambda(jax.nn.silu),
            linear_layer,
        ])
        
        # Final output projection - NORMAL initialization (not zero!)
        # Zero-init blocks gradients completely. The adaLN-Zero is handled
        # by the modulation (shift/scale) being zero-initialized above.
        self.linear_out = eqx.nn.Linear(dim_in, dim_out, key=key2)
        # DO NOT zero-initialize the output projection!
    
    def __call__(self, x: Array, cond: Array) -> Array:
        """
        Forward pass of modulated final layer.

        Args:
            x: Input tensor of shape (seq, dim_in)
            cond: Conditioning tensor of shape (dim_cond,)

        Returns:
            Output tensor of shape (seq, dim_out)
        """
        # Apply layer norm
        x = jax.vmap(self.ln)(x)

        # Get shift and scale from conditioning
        modulation = self.adaLN_modulation(cond)  # (2 * dim_in,)
        shift, scale = jnp.split(modulation, 2, axis=-1)  # (dim_in,), (dim_in,)
        shift = shift.reshape(1, -1)  # (1, dim_in)
        scale = scale.reshape(1, -1)  # (1, dim_in)

        # Modulate: x * (1 + scale) + shift
        x = x * (1.0 + scale) + shift

        # Apply output projection
        x = jax.vmap(self.linear_out)(x)

        return x

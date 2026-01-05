"""
Multi-modal DiT (Diffusion Transformer) implementation in JAX/Equinox.

This library provides a JAX/Equinox implementation of multi-modal diffusion transformers.
"""

from .layers import AdaptiveLayerNorm, TimestepEmbedding, timestep_embedding
from .blocks import MMDiTBlock
from .models import MMDiT
from .final_layer import ModulatedFinalLayer
from .feedforward import FeedForward
from .attention import JointAttention, MultiHeadRMSNorm
from .rope import apply_rope, build_rope_frequencies, rotate_half

# In-Context MMDiT (no AdaLN modulation, conditioning via tokens)
from .in_context_models import InContextMMDiT
from .in_context_blocks import InContextMMDiTBlock
from .in_context_final_layer import SimpleFinalLayer, RMSNorm

__version__ = "0.2.0"
__all__ = [
    # Original MMDiT (AdaLN modulation)
    "AdaptiveLayerNorm",
    "TimestepEmbedding",
    "timestep_embedding",
    "MMDiTBlock",
    "MMDiT",
    "ModulatedFinalLayer",
    "FeedForward",
    "JointAttention",
    "MultiHeadRMSNorm",
    "apply_rope",
    "build_rope_frequencies",
    "rotate_half",
    # In-Context MMDiT (conditioning via tokens)
    "InContextMMDiT",
    "InContextMMDiTBlock",
    "SimpleFinalLayer",
    "RMSNorm",
] 
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
import equinox as eqx


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool

    def __call__(self, observations: Dict[str, jnp.ndarray], train: bool = False, **kwargs) -> jnp.ndarray:
        encoding = self.encoder(observations["image"])
        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)
        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)
        return encoding


class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    goal_encoder: Optional[nn.Module]
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat goals so that there's a goal for each frame
            goal_image = repeat(
                goals["image"], "B H W C -> (B repeat) H W C", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            goal_image = goals["image"]

        if self.goal_encoder is None:
            # early goal concat
            encoder_inputs = jnp.concatenate([obs_image, goal_image], axis=-1)
            encoding = self.encoder(encoder_inputs)
        else:
            # late fusion
            encoding = self.encoder(obs_image)
            goal_encoding = self.goal_encoder(goals["image"])
            encoding = jnp.concatenate([encoding, goal_encoding], axis=-1)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding


class LCEncodingWrapper(nn.Module):
    """
    Encodes observations and language instructions into a single flat encoding.

    Takes a tuple (observations, goals) as input, where goals contains the language instruction.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat language so that there's an instruction for each frame
            language = repeat(
                goals["language"], "B E -> (B repeat) E", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            language = goals["language"]

        encoding = self.encoder(obs_image, cond_var=language)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding
    
class Imaginations_LCEncodingWrapper(nn.Module):
    """
    Encodes observations, imaginations, and language instructions into a single flat encoding.
    Uses a validity mask to filter out padded similar instructions.

    Takes a tuple (observations, goals) as input, where goals contains the language instruction.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool
    num_similar_instructions_used: int = 4


    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 6:
            # obs history case: (B, T, I_total, H, W, C)
            batch_size, obs_horizon, total_views = observations["image"].shape[:3]
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :, :num_views, :, :, :]  # (B, T, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]    # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]          # (B, num_views)
            
            # fold batch_size and obs_horizon into a single dimension
            obs_image = rearrange(obs_image_restricted, "B T I H W C -> (B T I) H W C")
            
            # Handle language with proper shape
            # repeat for obs_horizon: (B, T, I, E)
            language = repeat(
                language_restricted, "B I E -> B T I E", T=obs_horizon
            )
            # Flatten to match obs_image: (B*T*I, E)
            language = rearrange(language, "B T I E -> (B T I) E")
            
            # Get validity mask: (B, I) -> (B, T, I) -> (B*T*I,)
            valid_mask = repeat(
                mask_restricted, "B I -> B T I", T=obs_horizon
            )
            valid_mask = rearrange(valid_mask, "B T I -> (B T I)")
        else:
            # No history case: (B, I_total, H, W, C)
            batch_size = observations["image"].shape[0]
            total_views = observations["image"].shape[1]
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :num_views, :, :, :]  # (B, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]  # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]        # (B, num_views)
            
            obs_image = rearrange(obs_image_restricted, "B I H W C -> (B I) H W C")
            language = rearrange(language_restricted, "B I E -> (B I) E")
            
            # Get validity mask: (B, I) -> (B*I,)
            valid_mask = rearrange(mask_restricted, "B I -> (B I)")

        # Encode all views (including padded ones)
        encoding = self.encoder(obs_image, cond_var=language)
        # encoding shape: (B*T*I, F) or (B*I, F)
        
        # Apply validity mask by zeroing out encodings for padded views
        # Expand mask to match encoding dimensions: (B*T*I,) -> (B*T*I, 1)
        valid_mask_expanded = valid_mask[:, jnp.newaxis]
        encoding = encoding * valid_mask_expanded

        if len(observations["image"].shape) == 6:
            # unfold dimensions: (B*T*I, F) -> (B, T, I, F) -> (B, T*I*F)
            encoding = rearrange(
                encoding, "(B T I) F -> B (T I F)", 
                B=batch_size, T=obs_horizon, I=num_views
            )
        else:
            # unfold dimensions: (B*I, F) -> (B, I, F) -> (B, I*F)
            encoding = rearrange(
                encoding, "(B I) F -> B (I F)", 
                B=batch_size, I=num_views
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding
    
class Imaginations_LCEncodingWrapper_Attention(nn.Module):
    """
    Encodes observations, imaginations, and language instructions into a single flat encoding.
    Uses attention mechanism to compute a weighted combination of similar instruction encodings.
    
    The attention mechanism uses:
    - Query: Original language instruction embedding (index 0)
    - Keys: Similar language instruction embeddings (indices 1+)
    - Values: Similar instruction image encodings (indices 1+)
    
    Final output concatenates the original observation encoding with the attention-weighted
    similar instruction encodings.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
        attention_dim: Dimension for attention projection (default: 128).
        dropout_rate: Dropout rate for attention weights regularization (default: 0.1).
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool
    attention_dim: int = 128
    dropout_rate: float = 0.1
    num_similar_instructions_used: int = 4
    use_null_similar_images: bool = False

    def setup(self):
        # Attention projection layers
        self.query_proj = nn.Dense(self.attention_dim, name="query_proj")
        self.key_proj = nn.Dense(self.attention_dim, name="key_proj")
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals
        return_attention_weights = kwargs.get('return_attention_weights', False)

        if len(observations["image"].shape) == 6:
            # obs history case: (B, T, I_total, H, W, C)
            batch_size, obs_horizon, total_views = observations["image"].shape[:3]
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :, :num_views, :, :, :]  # (B, T, num_views, H, W, C)
            if self.use_null_similar_images:
                # Use only the original observation image, zero out similar images
                obs_image_restricted = jnp.concatenate([
                    obs_image_restricted[:, :, :1, :, :, :],  # Original observation
                    jnp.zeros_like(obs_image_restricted[:, :, 1:, :, :, :])  # Zero padding for similar images
                ], axis=2)
            language_restricted = goals["language_with_similar"][:, :num_views, :]    # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]          # (B, num_views)
            
            # fold batch_size and obs_horizon into a single dimension
            obs_image = rearrange(obs_image_restricted, "B T I H W C -> (B T I) H W C")
            
            # Handle language with proper shape
            # repeat for obs_horizon: (B, T, I, E)
            language = repeat(
                language_restricted, "B I E -> B T I E", T=obs_horizon
            )
            # Flatten to match obs_image: (B*T*I, E)
            language = rearrange(language, "B T I E -> (B T I) E")
            
            # Get validity mask: (B, I) -> (B, T, I) -> (B*T*I,)
            valid_mask = repeat(
                mask_restricted, "B I -> B T I", T=obs_horizon
            )
            valid_mask = rearrange(valid_mask, "B T I -> (B T I)")
        else:
            # No history case: (B, I_total, H, W, C)
            batch_size = observations["image"].shape[0]
            total_views = observations["image"].shape[1]
            obs_horizon = None
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :num_views, :, :, :]  # (B, num_views, H, W, C)
            if self.use_null_similar_images:
                # Use only the original observation image, zero out similar images
                obs_image_restricted = jnp.concatenate([
                    obs_image_restricted[:, :1, :, :, :],  # Original observation
                    jnp.zeros_like(obs_image_restricted[:, 1:, :, :, :])  # Zero padding for similar images
                ], axis=1)
            language_restricted = goals["language_with_similar"][:, :num_views, :]  # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]        # (B, num_views)
            
            obs_image = rearrange(obs_image_restricted, "B I H W C -> (B I) H W C")
            language = rearrange(language_restricted, "B I E -> (B I) E")
            
            # Get validity mask: (B, I) -> (B*I,)
            valid_mask = rearrange(mask_restricted, "B I -> (B I)")

        # Encode all views (including padded ones)
        encoding = self.encoder(obs_image, cond_var=language)
        # encoding shape: (B*T*I, F) or (B*I, F)
        
        # Apply validity mask by zeroing out encodings for padded views
        valid_mask_expanded = valid_mask[:, jnp.newaxis]
        encoding = encoding * valid_mask_expanded

        # Reshape to separate batch, time, and view dimensions
        if obs_horizon is not None:
            # (B*T*I, F) -> (B, T, I, F)
            encoding = rearrange(
                encoding, "(B T I) F -> B T I F", 
                B=batch_size, T=obs_horizon, I=num_views
            )
            language = rearrange(
                language, "(B T I) E -> B T I E",
                B=batch_size, T=obs_horizon, I=num_views
            )
            mask_restricted = repeat(mask_restricted, "B I -> B T I", T=obs_horizon)
            
            # Separate original and similar encodings
            original_encoding = encoding[:, :, 0, :]  # (B, T, F)
            similar_encodings = encoding[:, :, 1:, :]  # (B, T, num_similar, F)
            
            original_lang = language[:, :, 0, :]  # (B, T, E)
            similar_lang = language[:, :, 1:, :]  # (B, T, num_similar, E)
            
            similar_mask = mask_restricted[:, :, 1:]  # (B, T, num_similar)
        else:
            # (B*I, F) -> (B, I, F)
            encoding = rearrange(
                encoding, "(B I) F -> B I F", 
                B=batch_size, I=num_views
            )
            language = rearrange(
                language, "(B I) E -> B I E",
                B=batch_size, I=num_views
            )
            
            # Separate original and similar encodings
            original_encoding = encoding[:, 0, :]  # (B, F)
            similar_encodings = encoding[:, 1:, :]  # (B, num_similar, F)
            
            original_lang = language[:, 0, :]  # (B, E)
            similar_lang = language[:, 1:, :]  # (B, num_similar, E)
            
            similar_mask = mask_restricted[:, 1:]  # (B, num_similar)
        
        # Compute attention weights
        # Query from original language instruction
        query = self.query_proj(original_lang)  # (B, T, attention_dim) or (B, attention_dim)
        
        # Keys from similar language instructions
        keys = self.key_proj(similar_lang)  # (B, T, num_similar, attention_dim) or (B, num_similar, attention_dim)
        
        # Compute attention scores
        if obs_horizon is not None:
            # (B, T, attention_dim) @ (B, T, attention_dim, num_similar) -> (B, T, num_similar)
            query_expanded = query[:, :, jnp.newaxis, :]  # (B, T, 1, attention_dim)
            keys_transposed = jnp.transpose(keys, (0, 1, 3, 2))  # (B, T, attention_dim, num_similar)
            attention_scores = jnp.matmul(query_expanded, keys_transposed).squeeze(2)  # (B, T, num_similar)
        else:
            # (B, attention_dim) @ (B, attention_dim, num_similar) -> (B, num_similar)
            query_expanded = query[:, jnp.newaxis, :]  # (B, 1, attention_dim)
            keys_transposed = jnp.transpose(keys, (0, 2, 1))  # (B, attention_dim, num_similar)
            attention_scores = jnp.matmul(query_expanded, keys_transposed).squeeze(1)  # (B, num_similar)
        
        # Scale by sqrt of attention dimension
        attention_scores = attention_scores / jnp.sqrt(self.attention_dim)
        
        # Mask out invalid similar instructions (set to large negative value before softmax)
        attention_scores = jnp.where(similar_mask < 1e-9, -1e9, attention_scores)

        all_masked = jnp.all(similar_mask < 1e-9, axis=-1, keepdims=True)
        attention_scores = jnp.where(all_masked, jnp.zeros_like(attention_scores), attention_scores)
        
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)  # (B, T, num_similar) or (B, num_similar)
        
        # Apply dropout to attention weights for regularization (only active during training)
        attention_weights = self.dropout(attention_weights, deterministic=not train)
        
        attention_weights = jnp.where(all_masked, jnp.zeros_like(attention_weights), attention_weights)

        # Apply attention weights to similar encodings (values)
        if obs_horizon is not None:
            # (B, T, num_similar) @ (B, T, num_similar, F) -> (B, T, F)
            attention_weights_expanded = attention_weights[:, :, :, jnp.newaxis]  # (B, T, num_similar, 1)
            weighted_similar = (similar_encodings * attention_weights_expanded).sum(axis=2)  # (B, T, F)
            
            # Concatenate original encoding with attention-weighted similar encodings
            combined_encoding = jnp.concatenate([original_encoding, weighted_similar], axis=-1)  # (B, T, 2*F)
            
            # Flatten time dimension
            combined_encoding = rearrange(combined_encoding, "B T F -> B (T F)")
        else:
            # (B, num_similar) @ (B, num_similar, F) -> (B, F)
            attention_weights_expanded = attention_weights[:, :, jnp.newaxis]  # (B, num_similar, 1)
            weighted_similar = (similar_encodings * attention_weights_expanded).sum(axis=1)  # (B, F)
            
            # Concatenate original encoding with attention-weighted similar encodings
            combined_encoding = jnp.concatenate([original_encoding, weighted_similar], axis=-1)  # (B, 2*F)

        if self.use_proprio:
            combined_encoding = jnp.concatenate([combined_encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            combined_encoding = jax.lax.stop_gradient(combined_encoding)

        if return_attention_weights:
            return combined_encoding, attention_weights
        else:
            return combined_encoding

class Imaginations_LCEncodingWrapper_MultiLayerMultiHeadAttention(nn.Module):
    """
    Encodes observations, imaginations, and language instructions into a single flat encoding.
    Uses multi-layer multi-head attention mechanism to compute a weighted combination of similar instruction encodings.
    
    The attention mechanism uses multi-head self-attention across all views (original + similar instructions).
    
    Final output aggregates attended encodings across all views.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
        attention_dim: Dimension for attention projection (default: 512).
        dropout_rate: Dropout rate for attention weights regularization (default: 0.1).
        num_heads: Number of attention heads (default: 4).
        num_layers: Number of attention layers (default: 2).
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool
    attention_dim: int = 512
    dropout_rate: float = 0.1
    num_similar_instructions_used: int = 4
    num_heads: int = 4
    num_layers: int = 2
    aggregate: str = "task_instruct"  # or "mean"

    def setup(self):
        # Multi-head multi-layer attention
        self.attention_layers = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.attention_dim,
                dropout_rate=self.dropout_rate,
                name=f"attention_layer_{i}"
            ) for i in range(self.num_layers)
        ]
        
        # Layer normalization for each attention layer
        self.layer_norms = [
            nn.LayerNorm(name=f"layer_norm_{i}") for i in range(self.num_layers)
        ]
        
        # Projection layers
        self.input_proj = nn.Dense(self.attention_dim, name="input_proj")
        self.output_proj = nn.Dense(self.attention_dim, name="output_proj")
        
    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 6:
            # obs history case: (B, T, I_total, H, W, C)
            batch_size, obs_horizon, total_views = observations["image"].shape[:3]
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :, :num_views, :, :, :]  # (B, T, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]    # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]          # (B, num_views)
            
            # fold batch_size and obs_horizon into a single dimension
            obs_image = rearrange(obs_image_restricted, "B T I H W C -> (B T I) H W C")
            
            # Handle language with proper shape
            # repeat for obs_horizon: (B, T, I, E)
            language = repeat(
                language_restricted, "B I E -> B T I E", T=obs_horizon
            )
            # Flatten to match obs_image: (B*T*I, E)
            language = rearrange(language, "B T I E -> (B T I) E")
            
            # Get validity mask: (B, I) -> (B, T, I) -> (B*T*I,)
            valid_mask = repeat(
                mask_restricted, "B I -> B T I", T=obs_horizon
            )
            valid_mask = rearrange(valid_mask, "B T I -> (B T I)")
        else:
            # No history case: (B, I_total, H, W, C)
            batch_size = observations["image"].shape[0]
            total_views = observations["image"].shape[1]
            obs_horizon = None
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :num_views, :, :, :]  # (B, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]  # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]        # (B, num_views)
            
            obs_image = rearrange(obs_image_restricted, "B I H W C -> (B I) H W C")
            language = rearrange(language_restricted, "B I E -> (B I) E")
            
            # Get validity mask: (B, I) -> (B*I,)
            valid_mask = rearrange(mask_restricted, "B I -> (B I)")

        # Encode all views (including padded ones)
        encoding = self.encoder(obs_image, cond_var=language)
        # encoding shape: (B*T*I, F) or (B*I, F)
        
        # Apply validity mask by zeroing out encodings for padded views
        valid_mask_expanded = valid_mask[:, jnp.newaxis]
        encoding = encoding * valid_mask_expanded

        # Reshape to separate batch, time, and view dimensions
        if obs_horizon is not None:
            # (B*T*I, F) -> (B, T, I, F)
            encoding = rearrange(
                encoding, "(B T I) F -> B T I F", 
                B=batch_size, T=obs_horizon, I=num_views
            )
            language = rearrange(
                language, "(B T I) E -> B T I E",
                B=batch_size, T=obs_horizon, I=num_views
            )
            mask_restricted = repeat(mask_restricted, "B I -> B T I", T=obs_horizon)  # (B, T, num_similar+1)
            
            concat_encodings = jnp.concatenate([encoding, language], axis=-1)  # (B, T, I/num_similar+1, F+E)
            
            # Reshape for attention: (B*T, I, F+E)
            concat_encodings = rearrange(concat_encodings, "B T I D -> (B T) I D")
            attention_mask = rearrange(mask_restricted, "B T I -> (B T) I")  # (B*T, I)
        else:
            # (B*I, F) -> (B, I, F)
            encoding = rearrange(
                encoding, "(B I) F -> B I F", 
                B=batch_size, I=num_views
            )
            language = rearrange(
                language, "(B I) E -> B I E",
                B=batch_size, I=num_views
            )
            
            concat_encodings = jnp.concatenate([encoding, language], axis=-1)  # (B, I, F+E)
            attention_mask = mask_restricted  # (B, I)
        
        # Project concat_encodings to attention_dim
        x = self.input_proj(concat_encodings)  # (B*T, I, attention_dim) or (B, I, attention_dim)
        
        # Expand mask for element-wise multiplication
        mask_expanded = attention_mask[:, :, jnp.newaxis]  # (B*T, I, 1) or (B, I, 1)
        x = x * mask_expanded
        
        # where True/1 means attend, False/0 means don't attend
        # MultiHeadDotProductAttention expects mask shape (batch, num_heads, q_length, kv_length)
        attn_mask = attention_mask > 0.5  # Convert to boolean: True if valid, False if masked

        # # This ensures masked positions don't attend to anything and nothing attends to them
        # attn_mask_2d = attn_mask[:, jnp.newaxis, :, jnp.newaxis] & attn_mask[:, jnp.newaxis, jnp.newaxis, :]
        attn_mask_2d = attn_mask[:, jnp.newaxis, jnp.newaxis, :]
        
        # Apply multi-layer multi-head attention with residual connections
        for i, (attn_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Self-attention with masking
            attn_output = attn_layer(
                x, x,  # query and key-value are the same for self-attention
                mask=attn_mask_2d,
                deterministic=not train
            )
            
            attn_output = attn_output * mask_expanded
            # Residual connection and layer normalization
            x = layer_norm(x + attn_output)
            x = x * mask_expanded
        
        # Project back to output space
        attended_encodings = self.output_proj(x)  # (B*T, I, attention_dim) or (B, I, attention_dim)
        
        # Apply mask again to zero out masked positions
        attended_encodings = attended_encodings * mask_expanded
        
        if self.aggregate == "task_instruct":
            # Take attended encoding corresponding to original instruction (index 0)
            aggregated_encoding = attended_encodings[:, 0]  # (B*T, attention_dim) or (B, attention_dim)
        else:
            # Aggregate across views (mean pooling over valid positions)
            # Sum over views and divide by number of valid views
            sum_encodings = jnp.sum(attended_encodings, axis=1)  # (B*T, attention_dim) or (B, attention_dim)
            num_valid = jnp.sum(attention_mask, axis=1, keepdims=True)  # (B*T, 1) or (B, 1)
            num_valid = jnp.maximum(num_valid, 1.0)  # Avoid division by zero
            aggregated_encoding = sum_encodings / num_valid  # (B*T, attention_dim) or (B, attention_dim)
        
        # Reshape back if obs_horizon is not None
        if obs_horizon is not None:
            aggregated_encoding = rearrange(
                aggregated_encoding, "(B T) D -> B (T D)",
                B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            aggregated_encoding = jnp.concatenate([aggregated_encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            aggregated_encoding = jax.lax.stop_gradient(aggregated_encoding)

        return aggregated_encoding

class DiTEncoder(nn.Module):
    encoder: nn.Module
    num_similar_instructions_used: int = 4
    embed_lang_in_observation: bool = True
    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 6:
            # obs history case: (B, T, I_total, H, W, C)
            batch_size, obs_horizon, total_views = observations["image"].shape[:3]
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :, :num_views, :, :, :]  # (B, T, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]    # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]          # (B, num_views)

            obs_image = rearrange(obs_image_restricted, "B T I H W C -> (B T I) H W C")
            

            language = repeat(
                language_restricted, "B I E -> B T I E", T=obs_horizon
            )

            language = rearrange(language, "B T I E -> (B T I) E")
            
            # Get validity mask: (B, I) -> (B, T, I) -> (B*T*I,)
            valid_mask = repeat(
                mask_restricted, "B I -> B T I", T=obs_horizon
            )
            valid_mask = rearrange(valid_mask, "B T I -> (B T I)")
        else:
            # No history case: (B, I_total, H, W, C)
            batch_size = observations["image"].shape[0]
            total_views = observations["image"].shape[1]
            obs_horizon = None
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :num_views, :, :, :]  # (B, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]  # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]        # (B, num_views)
            
            obs_image = rearrange(obs_image_restricted, "B I H W C -> (B I) H W C")
            language = rearrange(language_restricted, "B I E -> (B I) E")
            
            # Get validity mask: (B, I) -> (B*I,)
            valid_mask = rearrange(mask_restricted, "B I -> (B I)")


        if self.embed_lang_in_observation:
            encoding = self.encoder(obs_image, train=train, cond_var=language)
        else:
            encoding = self.encoder(obs_image, train=train)

        
        # Apply validity mask by zeroing out encodings for padded views
        valid_mask_expanded = valid_mask[:, jnp.newaxis]
        encoding = encoding * valid_mask_expanded

        # Reshape to separate batch, time, and view dimensions
        if obs_horizon is not None:

            encoding = rearrange(
                encoding, "(B T I) F -> B T I F", 
                B=batch_size, T=obs_horizon, I=num_views
            )
            language = rearrange(
                language, "(B T I) E -> B T I E",
                B=batch_size, T=obs_horizon, I=num_views
            )
            
            concat_encodings = jnp.concatenate([encoding, language], axis=-1)  # (B, T, I/num_similar+1, F+E)
            
            concat_encodings = rearrange(concat_encodings, "B T I D -> B I (T D)")
            attention_mask = mask_restricted # (B, I)
        else:
            # (B*I, F) -> (B, I, F)
            encoding = rearrange(
                encoding, "(B I) F -> B I F", 
                B=batch_size, I=num_views
            )
            language = rearrange(
                language, "(B I) E -> B I E",
                B=batch_size, I=num_views
            )
            
            concat_encodings = jnp.concatenate([encoding, language], axis=-1)  # (B, I, F+E)
            attention_mask = mask_restricted  # (B, I)
        
        return concat_encodings, attention_mask


## TODO
class Imaginations_LCEncodingWrapper_MMDiT(nn.Module):
    """
    Encodes observations, imaginations, and language instructions using MMDiT architecture.
    
    This wrapper uses the InContextMMDiT model to process image observations and language
    instructions as separate modalities with joint attention. It supports two modes:
    
    1. concat_language_output=True: Language embeddings are concatenated with image encodings
       after the MMDiT encoder (late fusion).
    2. concat_language_output=False: Language is used as a separate multimodal input to MMDiT,
       and only the image modality output is used (true multimodal processing).
    
    Args:
        encoder: The base image encoder network (e.g., ResNet) to extract image features.
        mmdit_encoder: The InContextMMDiT model for multimodal processing.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
        num_similar_instructions_used: Number of similar instructions to use (default: 4).
        concat_language_output: If True, concatenate language embeddings with image output.
                                If False, use language as separate modality input to MMDiT.
        image_tokens_per_view: Number of tokens to use per image view (default: 16).
        language_projection_dim: Dimension to project language embeddings to (default: 512).
    """

    encoder: nn.Module  # Base image encoder (e.g., ResNet)
    mmdit_encoder: eqx.Module  # InContextMMDiT model
    use_proprio: bool
    stop_gradient: bool
    num_similar_instructions_used: int = 4
    concat_language_output: bool = True
    image_tokens_per_view: int = 16
    language_projection_dim: int = 512

    def setup(self):
        # Projection layers for converting encoder outputs to tokens
        # These will be determined based on encoder output dimension
        pass

    @nn.compact
    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 6:
            # obs history case: (B, T, I_total, H, W, C)
            batch_size, obs_horizon, total_views = observations["image"].shape[:3]
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :, :num_views, :, :, :]  # (B, T, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]    # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]          # (B, num_views)
            
            # fold batch_size and obs_horizon into a single dimension
            obs_image = rearrange(obs_image_restricted, "B T I H W C -> (B T I) H W C")
            
            # Handle language with proper shape
            # repeat for obs_horizon: (B, T, I, E)
            language = repeat(
                language_restricted, "B I E -> B T I E", T=obs_horizon
            )
            # Flatten to match obs_image: (B*T*I, E)
            language = rearrange(language, "B T I E -> (B T I) E")
            
            # Get validity mask: (B, I) -> (B, T, I) -> (B*T*I,)
            valid_mask = repeat(
                mask_restricted, "B I -> B T I", T=obs_horizon
            )
            valid_mask = rearrange(valid_mask, "B T I -> (B T I)")
        else:
            # No history case: (B, I_total, H, W, C)
            batch_size = observations["image"].shape[0]
            total_views = observations["image"].shape[1]
            obs_horizon = None
            
            # Restrict to num_similar_instructions_used + 1 (original obs + N similar instructions)
            num_views = self.num_similar_instructions_used + 1
            obs_image_restricted = observations["image"][:, :num_views, :, :, :]  # (B, num_views, H, W, C)
            language_restricted = goals["language_with_similar"][:, :num_views, :]  # (B, num_views, E)
            mask_restricted = goals["susie_goal_valid_mask"][:, :num_views]        # (B, num_views)
            
            obs_image = rearrange(obs_image_restricted, "B I H W C -> (B I) H W C")
            language = rearrange(language_restricted, "B I E -> (B I) E")
            
            # Get validity mask: (B, I) -> (B*I,)
            valid_mask = rearrange(mask_restricted, "B I -> (B I)")

        # Encode images using base encoder (e.g., ResNet)
        # This produces a flat encoding per image
        image_encoding = self.encoder(obs_image, train=train)  # (B*T*I, F) or (B*I, F)
        
        # Get dimensions
        encoding_dim = image_encoding.shape[-1]
        language_dim = language.shape[-1]
        
        # Project image encodings to tokens
        # Create multiple tokens per image for richer representation
        image_proj = nn.Dense(
            encoding_dim * self.image_tokens_per_view,
            name="image_token_proj"
        )(image_encoding)
        image_tokens = rearrange(
            image_proj, "B (T D) -> B T D",
            T=self.image_tokens_per_view, D=encoding_dim
        )  # (B*T*I, num_tokens, encoding_dim) or (B*I, num_tokens, encoding_dim)
        
        if self.concat_language_output:
            # Mode 1: Use language as conditioning, concatenate with output
            # In this mode, we don't use MMDiT's multimodal capabilities
            # Instead, we just process image tokens and concatenate language later
            
            # For simplicity in concat mode, we can use a simpler processing
            # or just return the base encoder output with language concatenated
            # Apply validity mask
            valid_mask_expanded = valid_mask[:, jnp.newaxis]
            image_encoding = image_encoding * valid_mask_expanded
            language = language * valid_mask_expanded
            
            # Reshape and aggregate
            if obs_horizon is not None:
                # (B*T*I, F) -> (B, T, I, F)
                image_encoding = rearrange(
                    image_encoding, "(B T I) F -> B T I F",
                    B=batch_size, T=obs_horizon, I=num_views
                )
                language = rearrange(
                    language, "(B T I) E -> B T I E",
                    B=batch_size, T=obs_horizon, I=num_views
                )
                
                # Take original view (index 0) and concatenate with language
                original_image = image_encoding[:, :, 0, :]  # (B, T, F)
                original_lang = language[:, :, 0, :]  # (B, T, E)
                
                # Concatenate
                combined = jnp.concatenate([original_image, original_lang], axis=-1)  # (B, T, F+E)
                combined = rearrange(combined, "B T D -> B (T D)")
            else:
                # (B*I, F) -> (B, I, F)
                image_encoding = rearrange(
                    image_encoding, "(B I) F -> B I F",
                    B=batch_size, I=num_views
                )
                language = rearrange(
                    language, "(B I) E -> B I E",
                    B=batch_size, I=num_views
                )
                
                # Take original view (index 0) and concatenate with language
                original_image = image_encoding[:, 0, :]  # (B, F)
                original_lang = language[:, 0, :]  # (B, E)
                
                # Concatenate
                combined = jnp.concatenate([original_image, original_lang], axis=-1)  # (B, F+E)
            
            encoding = combined
            
        else:
            # Mode 2: Use language as separate modality input to MMDiT
            # Project language to appropriate dimension
            language_proj = nn.Dense(
                self.language_projection_dim,
                name="language_proj"
            )(language)  # (B*T*I, lang_dim) or (B*I, lang_dim)
            
            # Add sequence dimension to language (treat as single token per view)
            language_tokens = language_proj[:, jnp.newaxis, :]  # (B*T*I, 1, lang_dim) or (B*I, 1, lang_dim)
            
            # Apply validity mask to tokens
            valid_mask_expanded = valid_mask[:, jnp.newaxis, jnp.newaxis]
            image_tokens = image_tokens * valid_mask_expanded
            language_tokens = language_tokens * valid_mask_expanded
            
            # Process through MMDiT
            # MMDiT expects tuple of (seq_len, dim) for each modality
            # We need to vmap over the batch dimension
            def process_single_view(img_tok, lang_tok, mask):
                # img_tok: (num_image_tokens, encoding_dim)
                # lang_tok: (1, lang_dim)
                # mask: scalar
                
                # Create attention mask if needed
                total_seq = img_tok.shape[0] + lang_tok.shape[0]
                attn_mask = jnp.ones((total_seq, total_seq), dtype=bool)
                
                # Call MMDiT
                img_out, lang_out = self.mmdit_encoder(
                    (img_tok, lang_tok),
                    attention_mask=attn_mask
                )
                
                # Return image output (aggregate tokens)
                # Take mean over tokens
                img_encoding = jnp.mean(img_out, axis=0) * mask  # (encoding_dim,)
                return img_encoding
            
            # vmap over batch dimension
            encoding = jax.vmap(process_single_view)(
                image_tokens, language_tokens, valid_mask
            )  # (B*T*I, encoding_dim) or (B*I, encoding_dim)
            
            # Reshape and aggregate
            if obs_horizon is not None:
                # (B*T*I, F) -> (B, T, I, F)
                encoding = rearrange(
                    encoding, "(B T I) F -> B T I F",
                    B=batch_size, T=obs_horizon, I=num_views
                )
                # Take original view (index 0)
                encoding = encoding[:, :, 0, :]  # (B, T, F)
                encoding = rearrange(encoding, "B T F -> B (T F)")
            else:
                # (B*I, F) -> (B, I, F)
                encoding = rearrange(
                    encoding, "(B I) F -> B I F",
                    B=batch_size, I=num_views
                )
                # Take original view (index 0)
                encoding = encoding[:, 0, :]  # (B, F)

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding


# class Imaginations_LCEncodingWrapper(nn.Module):
#     """
#     Encodes observations, imaginations, and language instructions into a single flat encoding.

#     Takes a tuple (observations, goals) as input, where goals contains the language instruction.

#     Args:
#         encoder: The encoder network for observations.
#         use_proprio: Whether to concatenate proprioception (after encoding).
#         stop_gradient: Whether to stop the gradient after the encoder.
#     """

#     encoder: nn.Module
#     use_proprio: bool
#     stop_gradient: bool

#     def __call__(
#         self,
#         observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
#     ) -> jnp.ndarray:
#         observations, goals = observations_and_goals

#         if len(observations["image"].shape) == 6:
#             # obs history case
#             batch_size, obs_horizon = observations["image"].shape[:2]
#             # fold batch_size into obs_horizon to encode each frame separately
#             obs_image = rearrange(observations["image"], "B T I H W C -> (B T I) H W C")
#             # repeat language so that there's an instruction for each frame
#             language = repeat(
#                 goals["language_with_similar"], "B I E -> (B repeat) I E", repeat=obs_horizon
#             )
#             language = rearrange(language, "B I E -> (B I) E")
#         else: ## No History Case
#             obs_image = rearrange(observations["image"], "B I H W C -> (B I) H W C")
#             language = rearrange(goals["language_with_similar"], "B I E -> (B I) E")

#         encoding = self.encoder(obs_image, cond_var=language)

#         if len(observations["image"].shape) == 6:
#             # unfold obs_horizon from batch_size
#             encoding = rearrange(
#                 encoding, "(B T I) F -> B (T I F)", B=batch_size, T=obs_horizon
#             )

#         if self.use_proprio:
#             encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

#         if self.stop_gradient:
#             encoding = jax.lax.stop_gradient(encoding)

#         return encoding

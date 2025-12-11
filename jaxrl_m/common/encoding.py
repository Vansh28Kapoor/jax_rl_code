from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


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

    def __call__(self, observations: Dict[str, jnp.ndarray], train: bool = False) -> jnp.ndarray:
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
    num_similar_instructions_used: int = 2

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
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

    def setup(self):
        # Attention projection layers
        self.query_proj = nn.Dense(self.attention_dim, name="query_proj")
        self.key_proj = nn.Dense(self.attention_dim, name="key_proj")
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool = False,
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
        
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)  # (B, T, num_similar) or (B, num_similar)
        
        # Apply dropout to attention weights for regularization (only active during training)
        attention_weights = self.dropout(attention_weights, deterministic=not train)
        
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

        return combined_encoding
    
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

import jax
import jax.numpy as jnp
import flax.linen as nn
import equinox as eqx
from jaxrl_m.common.typing import Dict
from typing import Optional, Tuple
from einops import rearrange, repeat

# # Try importing MMDiT - it should be available
# try:
#     from mmdit_jax import MMDiT
#     MMDIT_AVAILABLE = True
# except ImportError:
#     MMDIT_AVAILABLE = False
#     print("Warning: mmdit_jax not available for MMDiTScoreActor")


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(beta_start, beta_end, timesteps)
    return betas


def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


class ScoreActor(nn.Module):
    encoder: nn.Module
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module

    def __call__(self, observations, actions, time, train=False, return_attention_weights=False):
        # flatten actions
        flat_actions = actions.reshape([actions.shape[0], -1])

        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        
        # Call encoder - it may return (encoding, attention_weights) if return_attention_weights is enabled
        encoder_output = self.encoder(observations, train=train, return_attention_weights=return_attention_weights)
        
        # Check if encoder returned attention weights
        if isinstance(encoder_output, tuple):
            obs_enc, attention_weights = encoder_output
        else:
            obs_enc = encoder_output
            attention_weights = None
            
        reverse_input = jnp.concatenate([cond_enc, obs_enc, flat_actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, train=train)

        # un-flatten pred sequence
        eps_pred_reshaped = eps_pred.reshape(actions.shape)
        
        if return_attention_weights and attention_weights is not None:
            return eps_pred_reshaped, attention_weights
        else:
            return eps_pred_reshaped


class MMDiTScoreActor(nn.Module):
    """
    MMDiT-based Score Actor for diffusion policy.
    
    Uses MMDiT to process observations and actions as two modalities with
    timestep conditioning via AdaLN modulation. This is a natural fit since
    MMDiT was originally designed for diffusion transformers.
    
    Architecture:
        Observations → Encoder → Observation Tokens
        Actions → Flatten → Action Tokens
        (Obs Tokens, Action Tokens) → MMDiT(timestep) → Action Output
    
    Args:
        encoder: Encoder for observations (e.g., GCEncodingWrapper)
        mmdit_params_template: Template parameters for MMDiT (arrays only)
        mmdit_static: Static structure of MMDiT (non-arrays, functions, etc.)
        
    NOTE: MMDiT array parameters are stored in Flax params
    """
    
    encoder: nn.Module
    mmdit_params_template: eqx.Module  # MMDiT parameters (arrays only) for initialization
    mmdit_static: eqx.Module  # MMDiT static structure (non-trainable parts)

    @nn.compact
    def __call__(
        self, 
        observations: Dict, 
        actions: jnp.ndarray, 
        time: jnp.ndarray, 
        train: bool = False,
        return_attention_weights: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass through MMDiT score actor.
        
        Args:
            observations: Dictionary of observations
            actions: Noisy actions to denoise, shape (B, action_horizon, action_dim)
            time: Diffusion timestep, shape (B,) or (B, 1)
            train: Training mode flag
            return_attention_weights: Whether to return attention weights (not implemented for MMDiT)
        
        Returns:
            Predicted noise, shape (B, action_horizon, action_dim)
        """
        batch_size = actions.shape[0]
        action_horizon = actions.shape[1]
        action_dim_per_step = actions.shape[2]
        
        # Flatten actions: (B, action_horizon, action_dim) -> (B, 1, action_horizon * action_dim)
        action_tokens = actions.reshape([batch_size, 1, -1])
        
        # DiTEncoder
        obs_tokens, attention_mask = self.encoder(observations, train=train)
        attention_mask = attention_mask > 0.5  # (B, I)
        padded = jnp.pad(attention_mask, pad_width=((0, 0), (0, 1)), mode = "constant", constant_values=True)
        attention_mask = repeat(padded, "B I -> B T I", T=padded.shape[-1])

        
        # Ensure time is scalar per batch element
        if time.ndim > 1:
            time = time.squeeze(-1)  # (B,)
        
        # Store MMDiT array parameters in Flax params
        # These are the trainable weights
        mmdit_params = self.param('mmdit', lambda rng: self.mmdit_params_template)
        
        # Reconstruct the full MMDiT model by combining params with static structure
        mmdit_model = eqx.combine(mmdit_params, self.mmdit_static)

        # MMDiT expects tuple of (seq_len, dim) for each modality
        # We need to vmap over batch dimension
        def process_single_batch(obs_tok, act_tok, attn, t):
            """Process single batch element through MMDiT"""
            obs_out, act_out = mmdit_model(
                modality_tokens=(obs_tok, act_tok),
                timestep=t,
                attention_mask=attn, 
            )
            
            return act_out 
        
        # vmap over batch
        eps_pred = jax.vmap(process_single_batch)(
            obs_tokens, action_tokens, attention_mask, time
        ).squeeze(1)  # (B, flattened_action_dim)
        
        
        eps_pred_reshaped = eps_pred.reshape(actions.shape)
        return eps_pred_reshaped

class InContextMMDiTScoreActor(nn.Module):
    encoder: nn.Module
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module
    mmdit_params_template: eqx.Module  # MMDiT parameters (arrays only) for initialization
    mmdit_static: eqx.Module  # MMDiT static structure (non-trainable parts)

    @nn.compact
    def __call__(
        self, 
        observations: Dict, 
        actions: jnp.ndarray, 
        time: jnp.ndarray = None,
        train: bool = False,
        return_attention_weights: bool = False
    ) -> jnp.ndarray:

        flat_actions = actions.reshape([actions.shape[0], -1])

        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)

        # DiTEncoder
        obs_tokens, all_attention_mask = self.encoder(observations, train=train)
        all_attention_mask = all_attention_mask > 0.5  # (B, I)
        transf_attention_mask = repeat(all_attention_mask, "B I -> B T I", T=all_attention_mask.shape[-1])
        
        # Store MMDiT array parameters in Flax params
        #self.param is a Flax nn.Module API for declaring and retrieving parameters (Flax requires parameter initializers to have the signature: init_fn(rng) -> value)
        # These are the trainable weights
        mmdit_params = self.param('mmdit', lambda rng: self.mmdit_params_template)
        
        # Reconstruct the full MMDiT model by combining params with static structure
        mmdit_model = eqx.combine(mmdit_params, self.mmdit_static)

        # MMDiT expects tuple of (seq_len, dim) for each modality
        # We need to vmap over batch dimension
        def process_single_batch(obs_tok, attn_mask):
            """Process single batch element through MMDiT"""
            obs_out = mmdit_model(
                modality_tokens=(obs_tok,),  # Single-element tuple
                attention_mask=attn_mask, 
            )
            # Unpack single-element tuple for single modality
            return obs_out[0] 
        
        # vmap over batch
        obs_encoding = jax.vmap(process_single_batch)(
            obs_tokens, transf_attention_mask
        ) # (B, num_views, obs_dim)

        original_encoding = obs_encoding[:, 0, :]  # (B, F)
        similar_encodings = obs_encoding[:, 1:, :]  # (B, num_similar, F)
        sim_attention_mask = all_attention_mask[:, 1:, jnp.newaxis]  # (B, num_similar, 1)
        similar_encodings_mean = jnp.sum(similar_encodings * sim_attention_mask, axis=1) / jnp.sum(sim_attention_mask, axis=1)  # (B, F)

        concat_encodings = jnp.concatenate([original_encoding, similar_encodings_mean], axis=-1)

        reverse_input = jnp.concatenate([cond_enc, concat_encodings, flat_actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, train=train)

        # un-flatten pred sequence
        eps_pred_reshaped = eps_pred.reshape(actions.shape)
        return eps_pred_reshaped


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

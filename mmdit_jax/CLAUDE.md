# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install package
pip install -e .

# Run all tests (87 tests)
pytest mmdit_jax/tests/ -v

# Run specific test file
pytest mmdit_jax/tests/test_model.py

# Run specific test
pytest mmdit_jax/tests/test_model.py::test_mmdit_basic_forward
```

## Architecture Overview

This is a JAX/Equinox implementation of Multi-Modal Diffusion Transformer (MMDiT). The library provides two model variants:

1. **MMDiT** — SD3.5-aligned with AdaLN modulation (for diffusion/flow matching)
2. **InContextMMDiT** — Pure token processor with in-context conditioning (for imitation learning)

### Core Data Flow

```
# MMDiT (AdaLN modulation)
Input Tokens + Timestep → [MMDiTBlock × depth] → ModulatedFinalLayer → Output

# InContextMMDiT (in-context conditioning)
Input Tokens (with time tokens prepended) → [InContextMMDiTBlock × depth] → SimpleFinalLayer → Output
```

### Key Design Decisions

1. **Single-example processing**: All modules process individual examples without batch dimensions. Use `jax.vmap` or `eqx.filter_vmap` for batching.

2. **adaLN-Zero initialization**: All modulation parameters (gates, scales, shifts) are zero-initialized for stable training. This makes each block start as identity.

3. **Mandatory QK normalization**: Following SD3.5, RMS normalization is always applied to queries and keys in attention.

4. **SwiGLU activation**: Feedforward uses `Linear_out(Linear_in(x) * swish(Linear_gate(x)))`.

---

## Model Variants

### MMDiT (AdaLN Modulation)

Standard MMDiT using Adaptive Layer Normalization for conditioning. Best for diffusion/flow matching where conditioning is global.

**Module Structure:**
- `mmdit_jax/models.py` - `MMDiT`: Main model with timestep embedding
- `mmdit_jax/blocks.py` - `MMDiTBlock`: Block with AdaLN + gated residuals
- `mmdit_jax/layers.py` - `AdaptiveLayerNorm`, `TimestepEmbedding`
- `mmdit_jax/final_layer.py` - `ModulatedFinalLayer`

**Usage:**
```python
from mmdit_jax import MMDiT

model = MMDiT(
    depth=12,
    dim_modalities=(512, 256),
    dim_outs=(512, 256),
    dim_cond=1024,
    timestep_embed_dim=256,
    dim_head=64,
    heads=8,
    key=key,
)

# Timestep is a required parameter
outputs = model(modality_tokens, timestep=0.5)
```

### InContextMMDiT (Token-Based Conditioning)

Pure token processor without internal conditioning logic. Time/conditioning is handled externally by prepending tokens to the input sequence. Best for imitation learning where different modalities need different conditioning.

**Module Structure:**
- `mmdit_jax/in_context_models.py` - `InContextMMDiT`: Pure token processor
- `mmdit_jax/in_context_blocks.py` - `InContextMMDiTBlock`: RMSNorm + standard residuals
- `mmdit_jax/in_context_final_layer.py` - `SimpleFinalLayer`, `RMSNorm`

**Key Differences from MMDiT:**

| Aspect | MMDiT | InContextMMDiT |
|--------|-------|----------------|
| Conditioning | AdaLN modulation (global) | User prepends time tokens |
| LayerNorm | AdaptiveLayerNorm | RMSNorm |
| Residuals | Gated: `x + gate * out` | Standard: `x + out` |
| Final layer | ModulatedFinalLayer | SimpleFinalLayer |
| API | `model(tokens, timestep)` | `model(tokens)` |

---

## InContextMMDiT for Imitation Learning

The `InContextMMDiT` model is designed for imitation learning scenarios where:
- Observations don't need timestep conditioning
- Actions need timestep conditioning for denoising
- User has full control over where/how time tokens are injected

### Complete Example: Imitation Learning Setup

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from mmdit_jax import InContextMMDiT, TimestepEmbedding

# Configuration
obs_dim = 512       # Observation embedding dimension
action_dim = 256    # Action embedding dimension
obs_seq_len = 100   # Number of observation tokens
action_seq_len = 16 # Number of action tokens
num_time_tokens = 2 # Time tokens to prepend to actions

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 4)

# === Step 1: Create time embedding (user's responsibility) ===
time_embed = TimestepEmbedding(
    dim_embed=256,      # Sinusoidal embedding dimension
    dim_out=256,        # Output dimension after MLP
    key=keys[0]
)
# Project time embedding to action token dimension
time_to_token = eqx.nn.Linear(256, action_dim, key=keys[1])

# === Step 2: Create the model ===
model = InContextMMDiT(
    depth=12,
    dim_modalities=(obs_dim, action_dim),
    dim_outs=(obs_dim, action_dim),
    dim_head=64,
    heads=8,
    ff_mult=4,
    key=keys[2],
)

# === Step 3: Prepare inputs ===
# Observation tokens (no time conditioning needed)
obs_tokens = jax.random.normal(keys[3], (obs_seq_len, obs_dim))

# Action tokens (will be prepended with time tokens)
action_tokens = jax.random.normal(keys[3], (action_seq_len, action_dim))

# Embed timestep and create time tokens
t = jnp.array(0.5)  # Flow matching timestep
time_cond = time_embed(t)  # (256,)
time_token = time_to_token(time_cond)  # (action_dim,)
time_tokens = jnp.tile(time_token[None, :], (num_time_tokens, 1))  # (2, action_dim)

# Prepend time tokens to actions
action_with_time = jnp.concatenate([time_tokens, action_tokens], axis=0)
# Shape: (num_time_tokens + action_seq_len, action_dim) = (18, 256)

# === Step 4: Create attention mask ===
# Observations: full attention among themselves
# Actions: can see all observations + causal among action tokens
total_seq = obs_seq_len + num_time_tokens + action_seq_len

mask = jnp.ones((total_seq, total_seq), dtype=bool)

# Make action tokens causal (including time tokens)
action_total = num_time_tokens + action_seq_len
action_causal = jnp.tril(jnp.ones((action_total, action_total), dtype=bool))
mask = mask.at[obs_seq_len:, obs_seq_len:].set(action_causal)

# Optional: prevent observations from seeing actions
# mask = mask.at[:obs_seq_len, obs_seq_len:].set(False)

# === Step 5: Forward pass ===
obs_out, action_out = model(
    (obs_tokens, action_with_time),
    attention_mask=mask,
)

# === Step 6: Strip time tokens from output ===
action_out = action_out[num_time_tokens:]  # (action_seq_len, action_dim)

print(f"obs_out shape: {obs_out.shape}")      # (100, 512)
print(f"action_out shape: {action_out.shape}")  # (16, 256)
```

### Batched Processing

```python
# Wrap model with vmap for batching
batched_model = eqx.filter_vmap(model, in_axes=0)

batch_size = 8

# Batched inputs
obs_tokens = jax.random.normal(key, (batch_size, obs_seq_len, obs_dim))
action_with_time = jax.random.normal(key, (batch_size, num_time_tokens + action_seq_len, action_dim))
mask = jnp.ones((batch_size, total_seq, total_seq), dtype=bool)

# Batched forward
obs_out, action_out = batched_model((obs_tokens, action_with_time), mask)
# obs_out: (batch_size, obs_seq_len, obs_dim)
# action_out: (batch_size, num_time_tokens + action_seq_len, action_dim)
```

### Multiple Conditioning Variables (e.g., t and noise level r)

```python
# Create separate embeddings for each conditioning variable
time_embed_t = TimestepEmbedding(dim_embed=256, dim_out=256, key=keys[0])
time_embed_r = TimestepEmbedding(dim_embed=128, dim_out=256, key=keys[1])
time_to_token = eqx.nn.Linear(256, action_dim, key=keys[2])

# Embed and sum conditioning variables
t, r = jnp.array(0.5), jnp.array(0.3)
time_cond = time_embed_t(t) + time_embed_r(r)
time_token = time_to_token(time_cond)
time_tokens = jnp.tile(time_token[None, :], (num_time_tokens, 1))
```

### Three-Modality Example (Observations, States, Actions)

```python
# Only actions get time conditioning
model = InContextMMDiT(
    depth=12,
    dim_modalities=(512, 128, 256),  # obs, state, action
    dim_outs=(512, 128, 256),
    key=key,
)

obs_tokens = ...    # (obs_seq, 512)
state_tokens = ...  # (state_seq, 128)
action_with_time = jnp.concatenate([time_tokens, action_tokens], axis=0)

outputs = model((obs_tokens, state_tokens, action_with_time), attention_mask=mask)
obs_out, state_out, action_out = outputs

# Strip time tokens from action output
action_out = action_out[num_time_tokens:]
```

---

## Original MMDiT Details

### Modulation Pattern (SD3.5 aligned)

All adaptive layer norms use shift+scale modulation: `LN(x) * (1 + scale) + shift`. Both shift and scale are predicted from the timestep conditioning and zero-initialized (adaLN-Zero).

### Multiple Conditioning Variables

The model supports multiple conditioning variables (e.g., `t` and `r` for improved meanflow). Pass a tuple of dimensions to `timestep_embed_dim`:

```python
# Single conditioning variable (default)
model = MMDiT(..., timestep_embed_dim=256)
output = model(tokens, timestep=0.5)

# Multiple conditioning variables
model = MMDiT(..., timestep_embed_dim=(256, 128))
output = model(tokens, timestep=(t, r))
```

Each conditioning variable gets its own `TimestepEmbedding`, and their outputs are summed before modulation.

---

## Common Patterns

### Attention Mask Convention

Attention masks are 2D boolean arrays `(total_seq, total_seq)` where `True` means the query can attend to the key. Total sequence length is the sum of all modality sequence lengths.

```python
# Full bidirectional attention
mask = jnp.ones((total_seq, total_seq), dtype=bool)

# Causal attention for a modality
causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
mask = mask.at[start:end, start:end].set(causal)

# Block one modality from seeing another
mask = mask.at[mod1_start:mod1_end, mod2_start:mod2_end].set(False)
```

### Gradient Computation

```python
def loss_fn(model, tokens, targets):
    outputs = model(tokens)
    return jnp.mean((outputs[0] - targets[0])**2)

loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tokens, targets)
```

---

## Dependencies

- JAX ≥ 0.7.0
- Equinox ≥ 0.13.0
- einops ≥ 0.8.0

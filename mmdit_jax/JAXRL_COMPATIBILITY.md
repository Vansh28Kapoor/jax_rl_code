# mmdit-jax Compatibility with jaxrl Environment

## Summary

mmdit-jax has been successfully configured to work with the existing `jaxrl` conda environment without upgrading any core dependencies (JAX, numpy, scipy, etc.).

## Changes Made

### 1. Updated `pyproject.toml`
- **JAX requirement**: Changed from `>=0.7.0` to `>=0.4.13`
- **Equinox requirement**: Changed from `>=0.13.0` to `>=0.11.0`
- **Moved ipdb to optional dependencies**: To reduce required dependencies

### 2. Added JAX 0.4.13 Compatibility Layer
Created `dot_product_attention_compat()` in `mmdit_jax/helpers.py`:
- Detects if `jax.nn.dot_product_attention` is available (added in JAX 0.4.25+)
- Falls back to manual scaled dot-product attention for older versions
- Maintains identical behavior across JAX versions

### 3. Updated `attention.py`
- Replaced `jax.nn.dot_product_attention` with `dot_product_attention_compat`
- Ensures attention mechanism works with JAX 0.4.13

### 4. Updated `README.md`
- Added specific installation instructions for jaxrl environment
- Documented verified package versions

## Installation

```bash
conda activate jaxrl
cd jax_rl_code/mmdit_jax

# Install equinox 0.11.0 (compatible with JAX 0.4.13)
pip install "equinox==0.11.0" --no-deps
pip install jaxtyping "typeguard==2.13.3"

# Install mmdit-jax
pip install -e . --no-deps
```

## Verified Package Versions

After installation, your environment will have:
- **jax**: 0.4.13 (unchanged)
- **jaxlib**: 0.4.13 (unchanged)
- **numpy**: 1.24.3 (unchanged)
- **scipy**: 1.11.4 (unchanged)
- **einops**: 0.8.1 (unchanged)
- **equinox**: 0.11.0 (newly installed)
- **mmdit-jax**: 0.0.1 (newly installed)

## Tested Features

All major features have been tested and verified working with JAX 0.4.13:

✅ Basic forward pass  
✅ Batched inference with `jax.vmap`  
✅ Attention masks (including causal masking)  
✅ Multiple conditioning variables  
✅ InContextMMDiT models  
✅ Multi-modal processing (observations + actions)  

## Technical Details

The main compatibility issue was `jax.nn.dot_product_attention`, which was introduced in JAX 0.4.25. Our compatibility layer implements the **official JAX algorithm** based on the source code from `jax/_src/nn/functions.py`:

```python
# Official JAX scaled dot-product attention algorithm
# From: https://github.com/google/jax/blob/main/jax/_src/nn/functions.py

# Compute attention logits: Q @ K^T
attn_logits = einsum('...qhd,...khd->...hqk', query, key)

# Scale (default: 1/sqrt(head_dim))
attn_logits = attn_logits * scale

# Apply bias (if provided)
if bias is not None:
    attn_logits = attn_logits + bias

# Apply mask (True = attend, False = mask out)
if mask is not None:
    attn_logits = where(mask, attn_logits, dtype_min)

# Softmax and compute output
attn_weights = softmax(attn_logits, axis=-1)
output = einsum('...hqk,...khd->...qhd', attn_weights, value)
```

This implementation is **identical** to the official JAX version, ensuring exact numerical equivalence across JAX versions from 0.4.13 onwards.

## Usage Example

```python
import jax
import jax.numpy as jnp
from mmdit_jax import MMDiT

key = jax.random.PRNGKey(0)
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

# Single example
obs_tokens = jax.random.normal(key, (50, 512))
action_tokens = jax.random.normal(key, (20, 256))
timestep = jnp.array(0.5)

obs_out, action_out = model(
    modality_tokens=(obs_tokens, action_tokens),
    timestep=timestep,
)

# Batching
batched_model = jax.vmap(model, in_axes=(0, 0))
```

## Notes

- This configuration is specifically designed for the jaxrl environment
- If you encounter any issues, ensure all package versions match those listed above
- The compatibility layer adds minimal overhead and will automatically use the native JAX implementation when available in newer versions


# MMDiT-JAX

Multi-Modal Diffusion Transformer implementation in JAX/Equinox, aligned with Stable Diffusion 3.5 architecture.

## Installation

### With existing jaxrl conda environment

#### For Python 3.9+ (recommended)

```bash
conda activate jaxrl
cd jax_rl_code/mmdit_jax

# Install equinox 0.11.0 (compatible with JAX 0.4.13)
pip install "equinox==0.11.0" --no-deps
pip install jaxtyping "typeguard==2.13.3"

# Install mmdit-jax
pip install -e . --no-deps
```

#### For Python 3.8

```bash
conda activate jaxrl
cd jax_rl_code/mmdit_jax

# Install equinox 0.10.4 (last version supporting Python 3.8)
pip install "equinox==0.10.4" --no-deps
pip install jaxtyping "typeguard==2.13.3"

# Install mmdit-jax
pip install -e . --no-deps
```

This installs equinox (0.10.4 for Python 3.8, 0.11.0 for Python 3.9+) without upgrading your existing JAX, numpy, scipy, or other packages.

**Verified to work with:**
- Python 3.8+ ✓
- JAX 0.4.13 ✓
- Equinox 0.10.4+ (0.10.4 for Python 3.8, 0.11.0 for Python 3.9+) ✓
- einops 0.8.1 ✓
- numpy 1.24.3 ✓

### Standalone installation

```bash
pip install -e .
```

**Requirements:** 
- Python ≥ 3.8
- JAX ≥ 0.4.13
- Equinox ≥ 0.10.4 (0.10.4 for Python 3.8, 0.11.0 for Python 3.9+)
- einops ≥ 0.8.0

See [PYTHON_VERSION_SUPPORT.md](PYTHON_VERSION_SUPPORT.md) for detailed version compatibility information.

**Tested with:** JAX 0.4.13, Equinox 0.11.0+, einops 0.8.1

**Verify installation:**
```bash
python verify_installation.py
```
This runs comprehensive tests to ensure everything is working correctly.

## Quick Start

```python
import jax
import jax.numpy as jnp
from mmdit_jax import MMDiT

key = jax.random.PRNGKey(0)
model = MMDiT(
    depth=12,
    dim_modalities=(512, 256),   # input dimensions per modality
    dim_outs=(512, 256),         # output dimensions per modality
    dim_cond=1024,
    timestep_embed_dim=256,
    dim_head=64,
    heads=8,
    key=key,
)

# Single example (no batch dimension)
obs_tokens = jax.random.normal(key, (50, 512))
action_tokens = jax.random.normal(key, (20, 256))
timestep = jnp.array(0.5)

obs_out, action_out = model(
    modality_tokens=(obs_tokens, action_tokens),
    timestep=timestep,
)

# For batching, use vmap
batched_model = jax.vmap(model, in_axes=(0, 0, None))
```

### With Attention Masks

Attention masks are 2D boolean arrays of shape `(total_seq, total_seq)` where `True` means the query can attend to the key:

```python
seq_len1, seq_len2 = 50, 20
total_seq = seq_len1 + seq_len2

# Full attention by default; make second modality causal
attention_mask = jnp.ones((total_seq, total_seq), dtype=bool)
causal = jnp.tril(jnp.ones((seq_len2, seq_len2), dtype=bool))
attention_mask = attention_mask.at[seq_len1:, seq_len1:].set(causal)

outputs = model(
    modality_tokens=(obs_tokens, action_tokens),
    timestep=timestep,
    attention_mask=attention_mask,
)
```

## Architecture

```
Input Tokens → [MMDiTBlock × depth] → ModulatedFinalLayer → Output

MMDiTBlock:
├── AdaptiveLayerNorm (scale modulation)
├── JointAttention (with mandatory QK RMS norm)
├── Gate (zero-initialized)
├── AdaptiveLayerNorm (scale modulation)
├── SwiGLU FeedForward
└── Gate (zero-initialized)
```

**Key features:**
- **adaLN-Zero**: All modulation parameters zero-initialized for stable training
- **Mandatory QK normalization**: Always applied in attention (SD3.5 standard)
- **SwiGLU**: `Linear_out(Linear_in(x) * swish(Linear_gate(x)))`
- **Single-example processing**: Use `jax.vmap` or `eqx.filter_vmap` for batching
- **JAX 0.4.13+ compatible**: Uses official `jax.nn.dot_product_attention` implementation with automatic fallback for older versions

## API

```python
MMDiT(
    depth: int,                                    # Number of transformer blocks
    dim_modalities: Tuple[int],                    # Input dimension per modality
    dim_outs: Tuple[int],                          # Output dimension per modality
    dim_cond: int = 1024,                          # Conditioning dimension
    timestep_embed_dim: int | Tuple[int] = 256,   # Sinusoidal embedding dim(s)
    key: PRNGKey,
    # Passed to MMDiTBlock:
    dim_head: int = 64,
    heads: int = 8,
    ff_mult: int = 4,
)

model(
    modality_tokens: Tuple[Array, ...],           # Each (seq_len, dim)
    timestep: Array | Tuple[Array, ...],          # Scalar(s) in [0, 1]
    attention_mask: Optional[Array],              # (total_seq, total_seq) boolean
)
```

### Multiple Conditioning Variables

For algorithms like improved meanflow that require multiple conditioning variables (e.g., `t` and `r`), pass a tuple of dimensions to `timestep_embed_dim`:

```python
model = MMDiT(
    depth=12,
    dim_modalities=(512, 256),
    dim_outs=(512, 256),
    timestep_embed_dim=(256, 128),  # Two conditioning variables
    key=key,
)

t = jnp.array(0.5)  # timestep
r = jnp.array(0.3)  # second conditioning variable

outputs = model(
    modality_tokens=(obs_tokens, action_tokens),
    timestep=(t, r),  # Pass tuple of conditioning values
)
```

Each conditioning variable gets its own `TimestepEmbedding` module, and their outputs are summed before being used for modulation.

## Testing

```bash
# Run all tests (67 tests)
pytest mmdit_jax/tests/ -v

# Run specific test file
pytest mmdit_jax/tests/test_model.py

# Run specific test
pytest mmdit_jax/tests/test_model.py::test_mmdit_basic_forward
```

## Training Example

```python
import optax
import equinox as eqx

model = MMDiT(depth=12, dim_modalities=(512, 256), dim_outs=(512, 256), key=key)
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

def loss_fn(model, obs, actions, timesteps, targets):
    predictions = model((obs, actions), timesteps)
    return jnp.mean((predictions[1] - targets)**2)

@jax.jit
def train_step(model, opt_state, batch):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
```

## References

- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206)
- [SD3.5 Code](https://github.com/Stability-AI/sd3.5)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Equinox Documentation](https://docs.kidger.site/equinox/)

## License

MIT

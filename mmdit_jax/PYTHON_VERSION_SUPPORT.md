# Python Version Support for mmdit-jax

## Summary

mmdit-jax supports **Python 3.8+** when using the appropriate Equinox version.

## Version Matrix

| Python Version | Equinox Version | JAX Version | Status | Notes |
|---------------|-----------------|-------------|---------|-------|
| 3.8.x | 0.10.4 | 0.4.13 | ✅ Supported | Use Equinox 0.10.4 |
| 3.9+ | 0.11.0 | 0.4.13 | ✅ Supported | Recommended |
| 3.10+ | 0.11.0 | 0.4.13 | ✅ Supported | Fully tested |

## Installation Instructions

### Python 3.8

For Python 3.8 environments, install Equinox 0.10.4 (the last version supporting Python 3.8):

```bash
conda activate your_env  # or activate your Python 3.8 environment
cd jax_rl_code/mmdit_jax

# Install dependencies
pip install "equinox==0.10.4" --no-deps
pip install "jaxtyping>=0.2.15" "typeguard==2.13.3"

# Install mmdit-jax
pip install -e . --no-deps
```

### Python 3.9+

For Python 3.9+ environments (recommended), use Equinox 0.11.0:

```bash
conda activate your_env  # or activate your Python 3.9+ environment
cd jax_rl_code/mmdit_jax

# Install dependencies
pip install "equinox==0.11.0" --no-deps
pip install jaxtyping "typeguard==2.13.3"

# Install mmdit-jax
pip install -e . --no-deps
```

## Compatibility Details

### Actual PyPI Requirements (verified)

| Package | Version | Python Requirement |
|---------|---------|-------------------|
| JAX | 0.4.13 | >=3.8 |
| Equinox | 0.10.4 | ~=3.8 (3.8 only) |
| Equinox | 0.11.0 | ~=3.9 (3.9+ only) |
| einops | 0.8.0 | >=3.8 |
| jaxtyping | 0.2.15+ | >=3.8 |

### Why Two Equinox Versions?

Equinox dropped Python 3.8 support starting with version 0.10.5. To support both Python 3.8 and 3.9+, we need to use different Equinox versions:

- **Equinox 0.10.0 - 0.10.4**: Support Python 3.8
- **Equinox 0.10.5+**: Require Python 3.9+
- **Equinox 0.11.0+**: Require Python 3.9+

The mmdit-jax codebase is compatible with both Equinox 0.10.4 and 0.11.0 since we only use stable core APIs:
- `eqx.Module`
- `eqx.nn.Linear`, `eqx.nn.Sequential`, `eqx.nn.Lambda`, `eqx.nn.LayerNorm`
- `eqx.filter_vmap`, `eqx.tree_at`, `eqx.partition`, `eqx.combine`
- `eqx.is_array`, `eqx.filter`, `eqx.filter_value_and_grad`

## Testing Your Installation

After installation, verify everything works:

```bash
python verify_installation.py
```

This will check:
- Package versions
- Attention compatibility layer
- MMDiT model functionality
- Gradient computation
- All advanced features

## Important Notes

### Python 3.8 End of Life

**Note:** Python 3.8 reached end-of-life in October 2024 and no longer receives security updates. While mmdit-jax supports Python 3.8 for compatibility with existing environments (like jaxrl), we recommend upgrading to Python 3.9+ for new projects.

### JAX Version Compatibility

JAX 0.4.13 is the last version officially supporting Python 3.8. JAX 0.4.14+ require Python 3.9+.

If you're using Python 3.8:
- ✅ JAX 0.4.13: Supported
- ❌ JAX 0.4.14+: NOT supported (requires Python 3.9+)

## Troubleshooting

### "equinox requires Python ~=3.9"

If you see this error on Python 3.8, you're trying to install Equinox 0.10.5+. Use Equinox 0.10.4 instead:

```bash
pip uninstall equinox
pip install "equinox==0.10.4" --no-deps
```

### Verification Failed

If `verify_installation.py` fails, check:

1. **Python version**: Run `python --version`
   - Python 3.8.x → Use Equinox 0.10.4
   - Python 3.9+ → Use Equinox 0.11.0

2. **Package versions**: Run `pip list | grep -E "(jax|equinox|einops)"`
   - Should match the version matrix above

3. **Installation method**: Make sure you used `--no-deps` to avoid version conflicts

## Examples

### Example: Python 3.8 with JAX 0.4.13

```python
import sys
import jax
import equinox as eqx
from mmdit_jax import MMDiT

print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}")
print(f"Equinox: {eqx.__version__}")

# Expected output:
# Python: 3.8.20 ...
# JAX: 0.4.13
# Equinox: 0.10.4

# Create model (works identically on Python 3.8 and 3.9+)
key = jax.random.PRNGKey(0)
model = MMDiT(
    depth=12,
    dim_modalities=(512, 256),
    dim_outs=(512, 256),
    key=key,
)
```

### Example: Python 3.10 with JAX 0.4.13

```python
import sys
import jax
import equinox as eqx
from mmdit_jax import MMDiT

print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}")
print(f"Equinox: {eqx.__version__}")

# Expected output:
# Python: 3.10.18 ...
# JAX: 0.4.13
# Equinox: 0.11.0

# Same code works on Python 3.10
model = MMDiT(
    depth=12,
    dim_modalities=(512, 256),
    dim_outs=(512, 256),
    key=key,
)
```

## Support

For version-specific issues:
1. Check this document for correct version combinations
2. Run `verify_installation.py` to diagnose problems
3. Ensure you're using `--no-deps` during installation to prevent automatic upgrades







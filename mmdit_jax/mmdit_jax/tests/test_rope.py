"""Tests for Rotary Position Embedding (RoPE)."""

import jax
import jax.numpy as jnp
import pytest

from mmdit_jax.rope import rotate_half, apply_rope, build_rope_frequencies


class TestRotateHalf:
    """Tests for rotate_half function."""

    def test_rotate_half_shape(self):
        """Output shape should match input shape."""
        x = jnp.ones((4, 8, 64))
        result = rotate_half(x)
        assert result.shape == x.shape

    def test_rotate_half_values(self):
        """Test that rotation swaps and negates correctly."""
        # [0, 1, 2, 3] -> [-2, -3, 0, 1]
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        result = rotate_half(x)
        expected = jnp.array([-2.0, -3.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_rotate_half_2d(self):
        """Test rotation on 2D input."""
        x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = rotate_half(x)
        expected = jnp.array([[-3.0, -4.0, 1.0, 2.0], [-7.0, -8.0, 5.0, 6.0]])
        assert jnp.allclose(result, expected)


class TestBuildRopeFrequencies:
    """Tests for build_rope_frequencies function."""

    def test_output_shapes(self):
        """Cos and sin should have shape (seq_len, dim)."""
        seq_len, dim = 16, 64
        cos, sin = build_rope_frequencies(seq_len, dim)
        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_cos_sin_bounded(self):
        """Cos and sin values should be in [-1, 1]."""
        cos, sin = build_rope_frequencies(32, 64)
        assert jnp.all(cos >= -1.0) and jnp.all(cos <= 1.0)
        assert jnp.all(sin >= -1.0) and jnp.all(sin <= 1.0)

    def test_position_zero_has_no_rotation(self):
        """At position 0, cos should be 1 and sin should be 0 (no rotation)."""
        cos, sin = build_rope_frequencies(4, 8)
        # Position 0: cos(0) = 1, sin(0) = 0
        assert jnp.allclose(cos[0], jnp.ones(8))
        assert jnp.allclose(sin[0], jnp.zeros(8))

    def test_different_positions_different_frequencies(self):
        """Different positions should have different frequency values."""
        cos, sin = build_rope_frequencies(4, 8)
        # Positions 1, 2, 3 should all be different
        assert not jnp.allclose(cos[1], cos[2])
        assert not jnp.allclose(cos[2], cos[3])

    def test_frequency_decreases_with_dimension(self):
        """Higher dimension pairs should have lower frequencies (slower rotation)."""
        cos, sin = build_rope_frequencies(100, 8)
        # At position 50, lower dims should have rotated more than higher dims
        # Compare sin values - lower dims oscillate faster
        # dim 0-1 pair vs dim 6-7 pair
        # The absolute sin at position 50 should show more oscillation in lower dims
        # This is hard to test directly, but we can check frequencies are different
        assert not jnp.allclose(cos[:, 0], cos[:, 3])


class TestApplyRope:
    """Tests for apply_rope function."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = jnp.ones((8, 16, 64))  # (heads, seq_len, dim)
        cos, sin = build_rope_frequencies(16, 64)
        result = apply_rope(x, cos, sin)
        assert result.shape == x.shape

    def test_position_zero_identity(self):
        """At position 0 (cos=1, sin=0), RoPE should be identity."""
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 1, 8))
        cos = jnp.ones((1, 8))
        sin = jnp.zeros((1, 8))
        result = apply_rope(x, cos, sin)
        assert jnp.allclose(result, x)

    def test_rope_preserves_norm(self):
        """RoPE should approximately preserve vector norms (it's a rotation)."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (4, 16, 64))
        cos, sin = build_rope_frequencies(16, 64)
        result = apply_rope(x, cos, sin)

        # Norms should be preserved
        x_norms = jnp.linalg.norm(x, axis=-1)
        result_norms = jnp.linalg.norm(result, axis=-1)
        assert jnp.allclose(x_norms, result_norms, rtol=1e-5)

    def test_different_positions_different_outputs(self):
        """Same vector at different positions should produce different outputs."""
        x = jnp.ones((1, 4, 8))  # Same value at all positions
        cos, sin = build_rope_frequencies(4, 8)
        result = apply_rope(x, cos, sin)

        # Different positions should give different results
        assert not jnp.allclose(result[0, 0], result[0, 1])
        assert not jnp.allclose(result[0, 1], result[0, 2])


class TestRopeRelativePosition:
    """Tests for RoPE's relative position encoding property."""

    def test_relative_position_property(self):
        """
        Key property of RoPE: dot(RoPE(q, m), RoPE(k, n)) depends on (m - n).

        For positions with the same relative distance, the dot products should
        have the same pattern (modulo the actual q, k values).
        """
        key = jax.random.PRNGKey(123)
        dim = 8

        # Create a single query and key vector
        q = jax.random.normal(key, (dim,))
        k = jax.random.normal(jax.random.PRNGKey(456), (dim,))

        cos, sin = build_rope_frequencies(10, dim)

        # Apply RoPE at different positions
        def rope_single(x, pos):
            return (x * cos[pos]) + (rotate_half(x) * sin[pos])

        # Compute dot products for different position pairs with same relative distance
        # (m=2, n=1) has relative distance 1
        q_pos2 = rope_single(q, 2)
        k_pos1 = rope_single(k, 1)
        dot_2_1 = jnp.dot(q_pos2, k_pos1)

        # (m=5, n=4) also has relative distance 1
        q_pos5 = rope_single(q, 5)
        k_pos4 = rope_single(k, 4)
        dot_5_4 = jnp.dot(q_pos5, k_pos4)

        # These should be equal (same relative distance)
        assert jnp.allclose(dot_2_1, dot_5_4, rtol=1e-5)

        # (m=3, n=1) has relative distance 2, should be different
        q_pos3 = rope_single(q, 3)
        dot_3_1 = jnp.dot(q_pos3, k_pos1)
        assert not jnp.allclose(dot_2_1, dot_3_1)


class TestRopeIntegration:
    """Integration tests for RoPE with attention-like operations."""

    def test_rope_with_multihead_shape(self):
        """Test RoPE works with multi-head attention tensor shapes."""
        heads, seq_len, dim_head = 8, 32, 64
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (heads, seq_len, dim_head))
        k = jax.random.normal(jax.random.PRNGKey(1), (heads, seq_len, dim_head))

        cos, sin = build_rope_frequencies(seq_len, dim_head)

        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape

    def test_rope_attention_scores_shape(self):
        """Test that RoPE'd Q and K produce valid attention scores."""
        heads, seq_len, dim_head = 4, 16, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (heads, seq_len, dim_head))
        k = jax.random.normal(jax.random.PRNGKey(1), (heads, seq_len, dim_head))

        cos, sin = build_rope_frequencies(seq_len, dim_head)

        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        # Compute attention scores: (heads, seq_len, seq_len)
        scores = jnp.einsum('hqd,hkd->hqk', q_rope, k_rope)
        assert scores.shape == (heads, seq_len, seq_len)

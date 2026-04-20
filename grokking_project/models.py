import jax
import jax.numpy as jnp
from flax import nnx


def apply_rope(x, base=1e6):
    """
    Apply Rotary Positional Embeddings (RoPE) to Q,K.
    x shape: [batch, seq, heads, head_dim].
    """
    b, seq, n_heads, dim = x.shape
    half = dim // 2
    if half * 2 != dim:
        raise ValueError("Head dimension must be even for RoPE.")

    # Frequencies for rotation
    i = jnp.arange(half)
    theta = 1.0 / (base ** (2 * i / dim))  # shape [half]

    # Positions
    pos = jnp.arange(seq)
    angles = pos[:, None] * theta[None, :]  # [seq, half]

    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    # We'll broadcast cos, sin to [batch, seq, heads, half]
    # Tile over batch & heads, then swapaxes back
    cos = jnp.tile(cos[None, :, None, :], (b, 1, n_heads, 1))
    sin = jnp.tile(sin[None, :, None, :], (b, 1, n_heads, 1))

    # x = [b, seq, heads, dim]
    x1, x2 = jnp.split(x, 2, axis=-1)  # each [b, seq, heads, half]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    return jnp.concatenate([x1_rot, x2_rot], axis=-1)


class MultiHeadSelfAttention(nnx.Module):

    def __init__(self, dim: int, n_heads: int, dropout: float, rngs: nnx.Rngs):
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads

        self.norm = nnx.RMSNorm(dim, rngs=rngs)
        self.Wq = nnx.Linear(dim, n_heads * self.dim_head, use_bias=False, rngs=rngs)
        self.Wk = nnx.Linear(dim, n_heads * self.dim_head, use_bias=False, rngs=rngs)
        self.Wv = nnx.Linear(dim, n_heads * self.dim_head, use_bias=False, rngs=rngs)
        self.Wo = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.dropout_layer = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, rngs=None, training=True):
        b, seq, d = x.shape
        # Pre-norm
        x_norm = self.norm(x)

        q = self.Wq(x_norm).reshape(b, seq, self.n_heads, self.dim_head)
        k = self.Wk(x_norm).reshape(b, seq, self.n_heads, self.dim_head)
        v = self.Wv(x_norm).reshape(b, seq, self.n_heads, self.dim_head)

        # Apply RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # Causal mask: disallow attention to future tokens
        causal_mask = jnp.triu(
            jnp.full((seq, seq), -jnp.inf, dtype=jnp.float32), k=1
        )  # shape [seq, seq]

        # Convert to shape [b, n_heads, seq, seq] for broadcasting
        causal_mask = causal_mask[None, None, :, :]

        # Scaled dot-product
        attn_scores = jnp.einsum('bthd,bshd->bhts', q, k) / jnp.sqrt(self.dim_head)
        attn_scores = attn_scores + causal_mask
        attn_weights = nnx.softmax(attn_scores, axis=-1)

        # Weighted sum
        out = jnp.einsum('bhts,bshd->bthd', attn_weights, v)
        out = out.reshape(b, seq, self.dim)
        out = self.Wo(out)
        out = self.dropout_layer(out, rngs=rngs, deterministic=not training)
        return out


class FeedForward(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(dim, rngs=rngs)
        self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(hidden_dim, dim, use_bias=False, rngs=rngs)
        self.dropout_layer = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, rngs=None, training=True):
        x_norm = self.norm(x)
        x1 = self.w1(x_norm)          # [b, seq, hidden_dim]
        x_silu = nnx.silu(x1)
        # Gating branch
        x2 = self.w3(x_norm)          # [b, seq, hidden_dim]
        gated = x_silu * x2
        gated = self.dropout_layer(gated, rngs=rngs, deterministic=not training)
        out = self.w2(gated)          # [b, seq, dim]
        return out


class Transformer(nnx.Module):

    def __init__(
            self,
            n_tokens: int,
            dim: int,
            heads: int,
            dropout: float,
            depth: int,
            rngs: nnx.Rngs,
            pool: str = 'cls',
    ):
        self._pool = pool

        self.embedding = nnx.Embed(num_embeddings=n_tokens, features=dim, rngs=rngs)

        self.blocks = nnx.List([
            nnx.List([
                MultiHeadSelfAttention(dim, heads, dropout, rngs=rngs),
                FeedForward(dim, 4 * dim, dropout, rngs=rngs)
            ]) for _ in range(depth)
        ])
            
        self.final_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.output_linear = nnx.Linear(dim, n_tokens, use_bias=False, rngs=rngs)

    def __call__(self, x, rngs=None, training=True):
        # Token embeddings
        x = self.embedding(x)  # [batch, seq_len, dim]

        # Transformer blocks
        for attn, ffn in self.blocks:
            x = x + attn(x, rngs=rngs, training=training)
            x = x + ffn(x, rngs=rngs, training=training)

        x = self.final_norm(x)

        # Pool
        if self._pool == 'mean':
            x = jnp.mean(x, axis=1)  # [b, dim]
        else:
            # last token
            x = x[:, -1, :]          # [b, dim]

        # Classifier
        logits = self.output_linear(x)  # [b, n_tokens]
        return logits

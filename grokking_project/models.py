import jax
import jax.numpy as jnp
from flax import nnx


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
                nnx.MultiHeadAttention(
                    num_heads=heads,
                    in_features=dim,
                    dropout_rate=dropout,
                    decode=False,
                    normalize_qk=True,
                    rngs=rngs
                ),
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

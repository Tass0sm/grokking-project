from collections.abc import Callable
import functools
from typing import Any, Optional, Union
import warnings

import jax
import jax.numpy as jnp

import optax
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax.transforms import _clipping


def scale_by_lissa(
    S1: jax.typing.ArrayLike = 1,
    S2: jax.typing.ArrayLike = 2,
    loss_fn: callable,
) -> base.GradientTransformation:
    assert S1 > 0 and S2 > 1, "S1 and S2 must be greater than 0"

    def update_fn(updates, state, params=None, batch=None, rng=None):
        """Based on Algorithm 1 in https://www.jmlr.org/papers/volume18/16-491/16-491.pdf"""

        assert params is not None, "LiSSA requires params to compute HVP of loss_fn"

        def fill_row():
            X_i0 = updates

            def f(X_ij):
                _, h_v = jax.jvp(lambda p: loss_fn(p, batch), (params,), (X_prev,))

                X_ij_plus_1 = jax.tree.map(
                    lambda g, v, hv: g + (v - hv),
                    X_ij_plus_1, X_ij, H_ij_F
                )
                return updates + (I - H_sample_at_x_j) @ g
            
            X_i = jax.lax.scan(f, X_i0, xs=None, length=S2)

            return X

        fill_rows = jax.vmap(fill_row)
        X = fill_rows(X)
        
        # X_t
        last_col = jax.tree.map(lambda x: x.at[:, -1].get(), X)
        updates = jax.tree.map(lambda x: (1.0 / S1) * x.sum(axis=0), last_col)
        return updates, state

    return base.GradientTransformation(base.init_empty_state, update_fn)


def lissa(
    learning_rate: base.ScalarOrSchedule,
    S1: int = 1,
    S2: int = 2,
) -> base.GradientTransformationExtraArgs:
    r"""The Linear (time) Stochastic Second-Order Algorithm (LiSSA).
    """

    return combine.chain(
        scale_by_lissa(
            S1=S1,
            S2=S2,
        ),
        transform.scale_by_learning_rate(learning_rate),
    )

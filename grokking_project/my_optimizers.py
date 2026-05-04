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
from optax._src.linesearch import _extract_fns_kwargs
from optax.transforms import _clipping


def scale_by_lissa(
        S1: jax.typing.ArrayLike = 5,
        S2: jax.typing.ArrayLike = 5,
        alpha: float = 1.0,
        use_richardson_iteration: bool = False
) -> base.GradientTransformation:
    assert S1 > 0 and S2 > 1, "S1 and S2 must be greater than 0"

    def update_fn(updates, state, params=None, value_fn=None, lissa_batch_x=None, lissa_batch_y=None, **extra_args):
        """Based on Algorithm 1 in https://www.jmlr.org/papers/volume18/16-491/16-491.pdf"""

        assert params is not None, "LiSSA requires params to compute HVP of loss_fn"
        assert value_fn is not None, "LiSSA requires value_fn for which to compute HVPs"
        assert lissa_batch_x is not None and lissa_batch_y is not None, \
            "LISSA requires lissa_batch_x, lissa_batch_y with which to estimate HVPs for the value_fn at different data points."
        assert lissa_batch_x.shape[0] >= S1 * S2, \
            f"LISSA requires (for x) batch_size >= S1 * S2, got {lissa_batch_x.shape[0]} < {S1} * {S2}"
        assert lissa_batch_y.shape[0] >= S1 * S2, \
            f"LISSA requires (for y) batch_size >= S1 * S2, got {lissa_batch_y.shape[0]} < {S1} * {S2}"

        # Fetch arguments to be fed to value_fn from the extra_args
        (fn_kwargs,), remaining_kwargs = _extract_fns_kwargs(
            (value_fn,), extra_args
        )

        if remaining_kwargs:
            # TODO: Maybe print a warning here
            pass

        def ith_h_inverse_times_grad(ith_batch_x, ith_batch_y):
            if use_richardson_iteration:
                X_i0 = torch.zeros_like(updates)
            else:
                X_i0 = updates

            #
            def f(X_ij, xy):
                # add batch dimension
                x_j, y_j = jax.tree.map(lambda x: jnp.expand_dims(x, 0), xy)

                # jacobian vector product of the gradient of the loss function (hessian) and the gradient of the
                # \tilde{\nabla}^2 f[i, j](x_t) @ X[i, j]
                _, H_ij_F = jax.jvp(
                    jax.grad(lambda p: value_fn(p, x=x_j, y=y_j, **fn_kwargs)),
                    (params,),
                    (X_ij,)
                )

                if use_richardson_iteration:
                    # richardson iteration, based on author's implementation
                    # rather than pseudocode in paper
                    X_ij_plus_1 = jax.tree.map(
                        lambda g, v, hv: v + alpha * (g - hv),
                        updates, X_ij, H_ij_F
                    )
                else:
                    # X[i, j+1] = \nabla f(x_t) + (I - \tilde{\nabla}^2 f[i, j](x_t))X[i, j]
                    # X[i, j+1] = \nabla f(x_t) + (X[i, j] - \tilde{\nabla}^2 f[i, j](x_t) @ X[i, j])
                    # X[i, j+1] = g + (v - h @ v)
                    # X[i, j+1] = g + (v - hv)
                    X_ij_plus_1 = jax.tree.map(
                        lambda g, v, hv: g + (v - alpha * hv),
                        updates, X_ij, H_ij_F
                    )

                return X_ij_plus_1, None

            final_X, _ = jax.lax.scan(f, X_i0, xs=(ith_batch_x, ith_batch_y))

            return final_X

        lissa_batch_x = lissa_batch_x[:S1*S2].reshape((S1, S2, *lissa_batch_x.shape[1:]))
        lissa_batch_y = lissa_batch_y[:S1*S2].reshape((S1, S2, *lissa_batch_y.shape[1:]))

        h_inverse_times_grad_samples = jax.vmap(ith_h_inverse_times_grad)(lissa_batch_x, lissa_batch_y)
        h_inverse_times_grad_estimate = jax.tree.map(
            lambda x: (1.0 / S1) * x.sum(axis=0),
            h_inverse_times_grad_samples
        )

        return h_inverse_times_grad_estimate, state

    return base.GradientTransformationExtraArgs(base.init_empty_state, update_fn)


def lissa(
    learning_rate: base.ScalarOrSchedule,
    S1: int = 1,
    S2: int = 2,
    alpha: float = 1.0,
) -> base.GradientTransformationExtraArgs:
    r"""The Linear (time) Stochastic Second-Order Algorithm (LiSSA).
    """

    return combine.chain(
        scale_by_lissa(
            S1=S1,
            S2=S2,
            alpha=alpha,
        ),
        transform.scale_by_learning_rate(learning_rate),
    )

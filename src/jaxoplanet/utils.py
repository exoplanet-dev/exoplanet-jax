__all__ = ["get_dtype_eps", "zero_safe_sqrt"]

import jax
import jax.numpy as jnp


def get_dtype_eps(*args):
    return jnp.finfo(jax.dtypes.result_type(*args)).eps


@jax.custom_jvp
def zero_safe_sqrt(x):
    return jnp.sqrt(x)


@zero_safe_sqrt.defjvp
def zero_safe_sqrt_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = jnp.sqrt(x)
    cond = jnp.less(x, 10 * get_dtype_eps(x))
    denom = jnp.where(cond, jnp.ones_like(x), x)
    tangent_out = 0.5 * x_dot * primal_out / denom
    return primal_out, tangent_out

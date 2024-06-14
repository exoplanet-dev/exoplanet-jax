from functools import partial

import jax
import jax.numpy as jnp


@jax.custom_jvp
def cel(k2, p, a, b):
    return _cel(k2, jnp.sqrt(1.0 - k2), p, a, b)


@cel.defjvp
def _(primals, tangents):
    k2, p, a, b = primals
    k2_dot, p_dot, a_dot, b_dot = tangents

    kc = jnp.sqrt(1 - k2)
    kc2 = jnp.square(kc)
    ap = a * p
    bp = b * p
    lam = kc2 * (b + ap - 2.0 * bp) + p * (3.0 * bp - ap * p - 2.0 * b)

    primal_out = _cel(k2, kc, p, a, b)

    dk = -kc * (_cel(k2, kc, kc2, a, b) - primal_out) / (p - kc2)
    dk2 = -0.5 * dk / kc
    dp = (
        _cel(k2, kc, p, jnp.zeros_like(a), lam)
        + (b - ap) * _cel(k2, kc, jnp.ones_like(p), 1 - p, kc2 - p)
    ) / (2.0 * p * (1.0 - p) * (p - kc2))
    da = _cel(k2, kc, p, jnp.ones_like(a), jnp.zeros_like(b))
    db = _cel(k2, kc, p, jnp.zeros_like(a), jnp.ones_like(b))

    tangent_out = k2_dot * dk2 + p_dot * dp + a_dot * da + b_dot * db

    return primal_out, tangent_out


@partial(jax.jit, static_argnames=("max_iter",))
def _cel(k2, kc, p, a, b, *, max_iter=10):
    eps = jnp.finfo(jax.dtypes.result_type(k2)).eps
    ca = jnp.sqrt(eps)
    kc = jnp.maximum(kc, eps)

    def pos(op):
        _, p, a, b = op
        p = jnp.sqrt(p)
        b = b / p
        return p, a, b

    def neg(op):
        k2, p, a, b = op
        q = k2
        g = 1.0 - p
        f = g - k2
        q *= b - a * p
        p = jnp.sqrt(f / g)
        a = (a - b) / g
        b = -q / (g * g * p) + a * p
        return p, a, b

    p, a, b = jax.lax.cond(
        jnp.greater(p, 0.0),
        pos,
        neg,
        operand=(k2, p, a, b),
    )

    ee = kc
    f = a
    a += b / p
    g = ee / p
    b += f * g
    b += b
    p += g
    g = jnp.ones_like(kc)
    m = 1.0 + kc

    def loop_cond(args):
        i, kc, p, a, b, ee, g, m = args
        del p, a, b, ee, m
        return jnp.logical_and(jnp.abs(g - kc) > g * ca, i < max_iter)

    def loop_body(args):
        i, kc, p, a, b, ee, g, m = args
        kc = jnp.sqrt(ee)
        kc += kc
        ee = kc * m
        f = a
        pinv = 1.0 / p
        a += b * pinv
        g = ee * pinv
        b += f * g
        b += b
        p += g
        g = m
        m += kc

        return i + 1, kc, p, a, b, ee, g, m

    _, _, p, a, b, _, _, m = jax.lax.while_loop(
        loop_cond, loop_body, (0, kc, p, a, b, ee, g, m)
    )

    return 0.5 * jnp.pi * (a * m + b) / (m * (m + p))

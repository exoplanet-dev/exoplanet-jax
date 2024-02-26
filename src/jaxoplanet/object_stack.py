from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Generic, Optional, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
from jax.interpreters import batching
from jax.tree_util import tree_flatten

try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu  # type: ignore

Obj = TypeVar("Obj")


class ObjectStack(eqx.Module, Generic[Obj]):
    objects: tuple[Obj, ...]
    stack: Optional[Obj]

    def __init__(self, *objects: Obj):
        self.objects = objects

        # If all the objects have matching Pytree structure then we save a
        # stacked version that we can use for vmaps below. This allows for more
        # efficient evaluations in the case of multiple objects.
        self.stack = None
        if len(self.objects):
            spec = list(map(jax.tree_util.tree_structure, self.objects))
            if spec.count(spec[0]) == len(spec):
                self.stack = jax.tree_util.tree_map(
                    lambda *x: jnp.stack(x, axis=0), *self.objects
                )

    def __len__(self) -> int:
        return len(self.objects)

    def vmap(
        self,
        func: Callable,
        in_axes: Union[int, None, Sequence[Any]] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        """Map a function over the bodies of this system

        If possible, this method will apply the appropriate ``jax.vmap`` to the input
        function, but if the Pytree structure of the bodies don't match, this requires
        a loop over bodies, applying the function separately to each body, and stacking
        the results.

        Args:
            func: The function to map. It's first positional argument must accept a
                Keplerian :class:`Body` object.
            in_axes: The input axis specifications for all arguments after the first.
                The semantics should match ``jax.vmap``.
            out_axes: The output axis specifications, matching ``jax.vmap``.

        Returns:
            The vectorized version of ``func`` mapped over bodies in this system.

        For example, if (for some reason) we wanted to compute the $x$ positions of all
        the bodies in a system at a particular time, in units of the body radius, we
        could use the following:

        >>> from jaxoplanet.orbits.keplerian import Central, System
        >>> sys = System(Central())
        >>> sys = sys.add_body(period=1.0, radius=0.1)
        >>> sys = sys.add_body(period=2.0, radius=0.2)
        >>> pos = sys.body_vmap(
        ...     lambda body, t: body.position(t)[0] / body.radius,
        ...     in_axes=None,
        ... )
        >>> pos(0.2)
        <Quantity([40.0231   19.632687], 'dimensionless')>
        """

        @wraps(func)
        def impl(*args):
            # First, normalize the "in_axes" argument so we always have an iterable
            if isinstance(in_axes, Sequence):
                in_axes_ = tuple(in_axes)
            else:
                in_axes_ = tuple(in_axes for _ in args)

            # If we have a "body_stack" we can just vmap and be done
            if self.stack is not None:
                return jax.vmap(func, in_axes=(0,) + in_axes_, out_axes=out_axes)(
                    self.stack, *args
                )

            # Otherwise we need to loop over the bodies and apply the function once for
            # each body

            # Here we flatten the input arguments and `in_axes` so that we don't have
            # to deal with Pytree logic for the `in_axes` ourselves below.
            args_flat, in_tree = tree_flatten(args, is_leaf=batching.is_vmappable)
            in_axes_flat = jax.api_util.flatten_axes(  # type: ignore
                "body_vmap in_axes", in_tree, in_axes_
            )

            # Then loop over the bodies and accumulate the function results
            results = []
            out_tree = None
            for n, body in enumerate(self.objects):
                f = lu.wrap_init(func)
                f, out_tree_ = flatten_func_for_object_vmap(f, in_tree, in_axes_flat, n)
                results.append(f.call_wrapped(body, *args_flat))  # type: ignore
                out_tree_ = out_tree_()  # type: ignore
                if out_tree is not None and out_tree_ != out_tree:
                    raise ValueError(
                        "Input function does not return consistent Pytree structure;\n"
                        f"expected: {out_tree}\n"
                        f"found: {out_tree_}\n"
                    )
                out_tree = out_tree_

            out_axes_flat = jax.api_util.flatten_axes(  # type: ignore
                "body_vmap out_axes", out_tree, out_axes
            )
            return out_tree.unflatten(  # type: ignore
                parts[0] if a is None else jnpu.stack(parts, axis=a)
                for a, *parts in zip(out_axes_flat, *results)  # type: ignore
            )

        return impl


def index_helper(n, arg, axis):
    if axis is None:
        return arg
    else:
        idx = (slice(None),) * axis + (n,)
        return arg[idx]


@lu.transformation_with_aux  # type: ignore
def flatten_func_for_object_vmap(in_tree, in_axes_flat, index, body, *args_flat):
    args_indexed = (index_helper(index, *args) for args in zip(args_flat, in_axes_flat))
    ans = yield (body,) + in_tree.unflatten(args_indexed), {}
    yield tree_flatten(ans, is_leaf=batching.is_vmappable)

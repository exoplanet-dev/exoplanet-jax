from collections.abc import Callable, Iterable, Sequence
from typing import Any

from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.orbits.keplerian import Body, Central, OrbitalBody, System
from jaxoplanet.starry.surface import Surface


class SurfaceBody(Body):
    surface: Surface | None = None


class SurfaceSystem(System):
    central_surface: Surface | None
    _body_surface_stack: ObjectStack[Surface]

    def __init__(
        self,
        central: Central | None = None,
        central_surface: Surface | None = None,
        *,
        bodies: Iterable[tuple[Body | OrbitalBody | SurfaceBody, Surface | None]] = (),
    ):
        self.central = Central() if central is None else central

        if central_surface is None:
            central_surface = Surface()

        self.central_surface = central_surface

        orbital_bodies = []
        body_surfaces = []
        for body, surface in bodies:
            if isinstance(body, OrbitalBody):
                orbital_bodies.append(body)
                body_surfaces.append(surface)
            else:
                orbital_bodies.append(OrbitalBody(self.central, body))
                if surface is None:
                    body_surfaces.append(getattr(body, "surface", None))
                else:
                    body_surfaces.append(surface)

        self._body_stack = ObjectStack(*orbital_bodies)
        self._body_surface_stack = ObjectStack(*body_surfaces)

    @property
    def body_surfaces(self) -> tuple[Surface, ...]:
        return self._body_surface_stack.objects

    def add_body(
        self,
        body: Body | SurfaceBody | None = None,
        surface: Surface | None = None,
        **kwargs: Any,
    ) -> "SurfaceSystem":
        if body is None:
            body = Body(**kwargs)
        if surface is None:
            surface = getattr(body, "surface", None)
        bodies = list(zip(self.bodies, self.body_surfaces, strict=False)) + [
            (body, surface)
        ]
        return SurfaceSystem(
            central=self.central,
            central_surface=self.central_surface,
            bodies=bodies,
        )

    def surface_vmap(
        self,
        func: Callable,
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        return self._body_surface_stack.vmap(func, in_axes=in_axes, out_axes=out_axes)

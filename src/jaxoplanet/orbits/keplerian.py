"""A module to define Keplerian systems of bodies.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.core.kepler import kepler
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg


class Central(eqx.Module):
    """A central body in an orbital system

    Args:
        mass (Optional[Quantity]): Mass of central body [mass unit].
        radius (Optional[Quantity]): Radius of central body [length unit].
        density (Optional[Quantity]): Density of central body [mass/length**3 unit].
        map (Optional[Map]): Map of the central body. If None a uniform map with
            intensity 1 is used.
    """

    mass: Quantity = units.field(units=ureg.M_sun)
    radius: Quantity = units.field(units=ureg.R_sun)
    density: Quantity = units.field(units=ureg.M_sun / ureg.R_sun**3)

    @units.quantity_input(
        mass=ureg.M_sun, radius=ureg.R_sun, density=ureg.M_sun / ureg.R_sun**3
    )
    def __init__(
        self,
        *,
        mass: Quantity | None = None,
        radius: Quantity | None = None,
        density: Quantity | None = None,
    ):
        if radius is None and mass is None:
            radius = 1.0 * ureg.R_sun
            if density is None:
                mass = 1.0 * ureg.M_sun

        # Check that all the input values are scalars; we don't support Scalars
        # here
        if any(
            jnp.ndim(arg) != 0 for arg in (mass, radius, density) if arg is not None
        ):
            raise ValueError("All parameters of a KeplerianCentral must be scalars")

        # Compute all three parameters based on the input values
        error_msg = (
            "Values must be provided for exactly two of mass, radius, and density"
        )
        if density is None:
            if mass is None or radius is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = radius
            self.density = 3 * mass / (4 * jnp.pi * radius**3)
        elif radius is None:
            if mass is None or density is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = (3 * mass / (4 * jnp.pi * density)) ** (1 / 3)
            self.density = density
        elif mass is None:
            if radius is None or density is None:
                raise ValueError(error_msg)
            self.mass = 4 * jnp.pi * radius**3 * density / 3.0
            self.radius = radius
            self.density = density

    @classmethod
    @units.quantity_input(
        period=ureg.day, semimajor=ureg.R_sun, radius=ureg.R_sun, body_mass=ureg.M_sun
    )
    def from_orbital_properties(
        cls,
        *,
        period: Quantity,
        semimajor: Quantity,
        radius: Quantity | None = None,
        body_mass: Quantity | None = None,
    ) -> "Central":
        """Initialize the central body (e.g. a star) of an orbital system using
        orbital parameters to derive radius and mass.

        Args:
            period: The orbital period of the orbiting body [time unit].
            semimajor: The semi-major axis of the orbiting body [length unit].
            radius (Optional[Quantity]): Radius of central body [length unit].
            body_mass (Optional[Quantity]): Mass of orbiting body [mass unit].

        Returns:
            Central object
        """
        # Check that inputs are scalar
        if any(
            jnp.ndim(arg) != 0
            for arg in (semimajor, period, body_mass)
            if arg is not None
        ):
            raise ValueError(
                "All parameters of 'KeplerianCentral.from_orbital_properties' must be "
                "scalars; for multi-planet systems, use 'jax.vmap'"
            )

        radius = 1.0 * ureg.R_sun if radius is None else radius

        mass = 4 * jnp.pi**2 * semimajor**3 / (ureg.gravitational_constant * period**2)
        if body_mass is not None:
            mass -= body_mass

        return cls(mass=mass, radius=radius)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mass.shape


class Body(eqx.Module):
    """Initialize an orbiting body (e.g. a planet) using orbital parameters

    See https://docs.exoplanet.codes/en/latest/tutorials/data-and-models/ for a
    description of the orbital geometry.

    Args:
        central (Optional[Central]): The Central object that this Body orbits
            [Central].
        time_transit (Optional[Quantity]): The epoch of a reference transit
            [time unit].
        time_peri (Optional[Quantity]): The epoch of a reference periastron passage
            [time unit].
        period (Optional[Quantity]): Orbital period [time unit].
        semimajor (Optional[Quantity]): Semi-major axis in [length unit].
        inclination (Optional[Quantity]): Inclination of orbital plane in
            [angular unit].
        impact_param (Optional): Impact parameter.
        eccentricity (Optional): Eccentricity, must be ``0 <= eccentricity < 1``
            where 0 = circular orbit.
        omega_peri (Optional[Quantity]): Argument of periastron [angular unit].
        sin_omega_peri (Optional): sin(argument of periastron).
        cos_omega_peri (Optional): cos(argument of periastron).
        asc_node (Optional[Quantity]): Longitude of ascending node [angular unit].
        sin_asc_node (Optional): sin(longitude of ascending node).
        cos_asc_node (Optional): cos(longitude of ascending node).
        mass (Optional[Quantity]): Mass of orbiting body [mass unit].
        radius (Optional[Quantity]): Radius of orbiting body [length unit].
        central_radius (Optional[Quantity]): Radius of central body [length unit].
        radial_velocity_semiamplitude (Optional[Quantity]): The radial velocity
            semi-amplitude [length/time unit].
        parallax (Optional[Quantity]): Parallax (to convert position/velocity into
            arcsec). [length unit].
    """

    time_transit: Quantity | None = units.field(default=None, units=ureg.d)
    time_peri: Quantity | None = units.field(default=None, units=ureg.d)
    period: Quantity | None = units.field(default=None, units=ureg.d)
    semimajor: Quantity | None = units.field(default=None, units=ureg.R_sun)
    inclination: Quantity | None = units.field(default=None, units=ureg.radian)
    impact_param: Quantity | None = units.field(default=None, units=ureg.dimensionless)
    eccentricity: Quantity | None = units.field(default=None, units=ureg.dimensionless)
    omega_peri: Quantity | None = units.field(default=None, units=ureg.radian)
    sin_omega_peri: Quantity | None = units.field(
        default=None, units=ureg.dimensionless
    )
    cos_omega_peri: Quantity | None = units.field(
        default=None, units=ureg.dimensionless
    )
    asc_node: Quantity | None = units.field(default=None, units=ureg.radian)
    sin_asc_node: Quantity | None = units.field(default=None, units=ureg.dimensionless)
    cos_asc_node: Quantity | None = units.field(default=None, units=ureg.dimensionless)
    mass: Quantity | None = units.field(default=None, units=ureg.M_sun)
    radius: Quantity | None = units.field(default=None, units=ureg.R_sun)
    radial_velocity_semiamplitude: Quantity | None = units.field(
        default=None, units=ureg.R_sun / ureg.d
    )
    parallax: Quantity | None = units.field(default=None, units=ureg.arcsec)

    def __check_init__(self) -> None:
        if not ((self.period is None) ^ (self.semimajor is None)):
            raise ValueError("Exactly one of period or semimajor must be specified")

        # Check that all the input arguments have the right shape
        provided_input_arguments = [
            arg
            for arg in (
                self.time_transit,
                self.time_peri,
                self.period,
                self.semimajor,
                self.inclination,
                self.impact_param,
                self.eccentricity,
                self.omega_peri,
                self.sin_omega_peri,
                self.cos_omega_peri,
                self.asc_node,
                self.sin_asc_node,
                self.cos_asc_node,
                self.mass,
                self.radius,
                self.radial_velocity_semiamplitude,
                self.parallax,
            )
            if arg is not None
        ]
        if any(jnp.ndim(arg) != 0 for arg in provided_input_arguments):
            raise ValueError(
                "All input arguments to 'Body' must be scalars; "
                "for multi-planet systems, use a 'System'"
            )

        if self.omega_peri is not None and (
            self.sin_omega_peri is not None or self.cos_omega_peri is not None
        ):
            raise ValueError(
                "Cannot specify both omega_peri and sin_omega_peri or cos_omega_peri"
            )
        if (self.sin_omega_peri is not None) ^ (self.cos_omega_peri is not None):
            raise ValueError("Both sin_omega_peri and cos_omega_peri must be specified")

        if self.asc_node is not None and (
            self.sin_asc_node is not None or self.cos_asc_node is not None
        ):
            raise ValueError(
                "Cannot specify both asc_node and sin_asc_node or cos_asc_node"
            )
        if (self.sin_asc_node is not None) ^ (self.cos_asc_node is not None):
            raise ValueError("Both sin_asc_node and cos_asc_node must be specified")

        has_omega_peri = (
            self.omega_peri is not None
            or self.sin_omega_peri is not None
            or self.cos_omega_peri is not None
        )
        if (self.eccentricity is not None) ^ has_omega_peri:
            raise ValueError(
                "Both or neither of eccentricity and omega_peri must be specified"
            )

        if self.impact_param is not None and self.inclination is not None:
            raise ValueError(
                "Only one of impact_param and inclination can be specified"
            )

        if self.time_transit is not None and self.time_peri is not None:
            raise ValueError("Only one of time_transit or time_peri can be specified")


class OrbitalBody(eqx.Module):
    """A computational tool"""

    central: Central
    time_ref: Quantity = units.field(units=ureg.d)
    time_transit: Quantity = units.field(units=ureg.d)
    period: Quantity = units.field(units=ureg.d)
    semimajor: Quantity = units.field(units=ureg.R_sun)
    sin_inclination: Quantity = units.field(units=ureg.dimensionless)
    cos_inclination: Quantity = units.field(units=ureg.dimensionless)
    impact_param: Quantity = units.field(units=ureg.dimensionless)
    mass: Quantity | None = units.field(units=ureg.M_sun)
    radius: Quantity | None = units.field(units=ureg.R_sun)
    eccentricity: Quantity | None = units.field(units=ureg.dimensionless)
    sin_omega_peri: Quantity | None = units.field(units=ureg.dimensionless)
    cos_omega_peri: Quantity | None = units.field(units=ureg.dimensionless)
    sin_asc_node: Quantity | None = units.field(units=ureg.dimensionless)
    cos_asc_node: Quantity | None = units.field(units=ureg.dimensionless)
    radial_velocity_semiamplitude: Quantity | None = units.field(
        units=ureg.R_sun / ureg.d
    )
    parallax: Quantity | None = units.field(units=ureg.arcsec)

    def __init__(self, central: Central, body: Body):
        self.central = central

        # Save the input mass and radius
        self.radius = body.radius
        self.mass = body.mass
        self.radial_velocity_semiamplitude = body.radial_velocity_semiamplitude
        self.parallax = body.parallax

        # Work out the period and semimajor axis to be consistent
        mass_factor = ureg.gravitational_constant * self.total_mass
        if body.semimajor is None:
            assert body.period is not None
            self.semimajor = jnpu.cbrt(mass_factor * body.period**2 / (4 * jnp.pi**2))
            self.period = body.period
        elif body.period is None:
            assert body.semimajor is not None
            self.semimajor = body.semimajor
            self.period = (
                2 * jnp.pi * body.semimajor * jnpu.sqrt(body.semimajor / mass_factor)
            )

        # Handle treatment and normalization of angles
        if body.omega_peri is not None:
            self.sin_omega_peri = jnpu.sin(body.omega_peri)
            self.cos_omega_peri = jnpu.cos(body.omega_peri)
        else:
            self.sin_omega_peri = body.sin_omega_peri
            self.cos_omega_peri = body.cos_omega_peri

        if body.asc_node is not None:
            self.sin_asc_node = jnpu.sin(body.asc_node)
            self.cos_asc_node = jnpu.cos(body.asc_node)
        else:
            self.sin_asc_node = body.sin_asc_node
            self.cos_asc_node = body.cos_asc_node

        # Handle eccentric and circular orbits
        self.eccentricity = body.eccentricity
        if self.eccentricity is None:
            M0 = jnpu.full_like(self.period, 0.5 * jnp.pi)  # type: ignore
            incl_factor = 1
        else:
            assert self.sin_omega_peri is not None
            assert self.cos_omega_peri is not None
            opsw = 1 + self.sin_omega_peri
            E0 = 2 * jnpu.arctan2(
                jnpu.sqrt(1 - self.eccentricity) * self.cos_omega_peri,
                jnpu.sqrt(1 + self.eccentricity) * opsw,
            )
            M0 = E0 - self.eccentricity * jnpu.sin(E0)

            ome2 = 1 - self.eccentricity**2
            incl_factor = (1 + self.eccentricity * self.sin_omega_peri) / ome2

        # Handle inclined orbits
        dcosidb = incl_factor * central.radius / self.semimajor
        if body.impact_param is not None:
            self.impact_param = body.impact_param
            self.cos_inclination = dcosidb * body.impact_param
            self.sin_inclination = jnpu.sqrt(1 - self.cos_inclination**2)
        elif body.inclination is not None:
            self.cos_inclination = jnpu.cos(body.inclination)
            self.sin_inclination = jnpu.sin(body.inclination)
            self.impact_param = self.cos_inclination / dcosidb
        else:
            z = jnp.zeros_like(self.period.magnitude) * ureg.dimensionless
            self.impact_param = z
            self.cos_inclination = z
            self.sin_inclination = (
                jnp.ones_like(self.period.magnitude) * ureg.dimensionless
            )

        # Work out all the relevant reference times
        self.time_ref = -M0 * self.period / (2 * jnp.pi)
        if body.time_transit is not None:
            self.time_transit = body.time_transit
        elif body.time_peri is not None:
            self.time_transit = body.time_peri - self.time_ref
        else:
            self.time_transit = jnpu.zeros_like(self.time_ref)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.period.shape

    @property
    def central_radius(self) -> Quantity:
        return self.central.radius

    @property
    def time_peri(self) -> Quantity:
        return self.time_transit + self.time_ref  # type: ignore

    @property
    def inclination(self) -> Quantity:
        return jnpu.arctan2(self.sin_inclination, self.cos_inclination)

    @property
    def omega_peri(self) -> Quantity | None:
        if self.eccentricity is None:
            return None
        assert self.sin_omega_peri is not None
        assert self.cos_omega_peri is not None
        return jnpu.arctan2(self.sin_omega_peri, self.cos_omega_peri)

    @property
    def total_mass(self) -> Quantity:
        return self.central.mass if self.mass is None else self.mass + self.central.mass

    def position(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's position in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        semimajor = -self.semimajor * self.central.mass / self.total_mass
        return self._get_position_and_velocity(
            t, semimajor=semimajor, parallax=parallax
        )[0]

    def central_position(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """The central's position in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        semimajor = self.semimajor * self.mass / self.total_mass
        return self._get_position_and_velocity(
            t, semimajor=semimajor, parallax=parallax
        )[0]

    def relative_position(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's position relative to the central in the X,Y,Z frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        return self._get_position_and_velocity(
            t,
            semimajor=-self.semimajor,
            parallax=parallax,  # type: ignore
        )[0]

    def relative_angles(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity]:
        """This body's relative position to the central in the sky plane, in
        separation, position angle coordinates

        Args:
            t: The times where the angles should be evaluated.

        Returns:
            The separation (arcseconds) and position angle (radians, measured
            east of north) of the planet relative to the star.
        """
        X, Y, _ = self.relative_position(t, parallax=parallax)
        rho = jnpu.sqrt(X**2 + Y**2)
        theta = jnpu.arctan2(Y, X)
        return rho, theta

    def velocity(
        self, t: Quantity, semiamplitude: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's velocity in the barycentric frame

        Args:
            t: The times where the velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. By convention, this
                is half of the peak-to-peak amplitude of the central's velocity.
                If provided, the ``mass`` and ``inclination`` parameters will be
                ignored and this amplitude will be used instead.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``R_sun/day``.
        """
        if semiamplitude is None:
            mass: Quantity = -self.central.mass  # type: ignore
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.central.mass / self.mass
        return self._get_position_and_velocity(t, semiamplitude=k)[1]

    def central_velocity(
        self, t: Quantity, semiamplitude: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """The central's velocity in the barycentric frame

        Args:
            t: The times where the velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. By convention, this
                is half of the peak-to-peak amplitude of the central's velocity.
                If provided, the ``mass`` and ``inclination`` parameters will be
                ignored and this amplitude will be used instead.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.
        """
        if semiamplitude is None:
            return self._get_position_and_velocity(t, mass=self.mass)[1]
        _, v = self._get_position_and_velocity(t, semiamplitude=semiamplitude)
        return v

    def relative_velocity(
        self, t: Quantity, semiamplitude: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's velocity relative to the central

        Args:
            t: The times where the velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. By convention, this
                is half of the peak-to-peak amplitude of the central's velocity.
                If provided, the ``mass`` and ``inclination`` parameters will be
                ignored and this amplitude will be used instead.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.
        """
        if semiamplitude is None:
            mass: Quantity = -self.total_mass  # type: ignore
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.total_mass / self.mass
        _, v = self._get_position_and_velocity(t, semiamplitude=k)
        return v

    def radial_velocity(
        self, t: Quantity, semiamplitude: Quantity | None = None
    ) -> Quantity:
        """Get the radial velocity of the central

        .. note:: The convention is that positive `z` points *towards* the
            observer. However, for consistency with radial velocity literature
            this method returns values where positive radial velocity
            corresponds to a redshift as expected.

        Args:
            t: The times where the radial velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. If provided, the
                ``mass`` and ``inclination`` parameters will be ignored and
                this amplitude will be used instead.

        Returns:
            The reflex radial velocity evaluated at ``t``.
        """
        return -self.central_velocity(t, semiamplitude=semiamplitude)[2]  # type: ignore

    def _warp_times(self, t: Quantity) -> Quantity:
        return t - self.time_transit  # type: ignore

    @units.quantity_input(t=ureg.d)
    def _get_true_anomaly(self, t: Quantity) -> tuple[Quantity, Quantity]:
        M = 2 * jnp.pi * (self._warp_times(t) - self.time_ref) / self.period
        if self.eccentricity is None:
            return jnpu.sin(M), jnpu.cos(M)
        return kepler(M.magnitude, self.eccentricity.magnitude)

    def _rotate_vector(
        self, x: Quantity, y: Quantity, *, include_inclination: bool = True
    ) -> tuple[Quantity, Quantity, Quantity]:
        """Apply the rotation matrices to go from orbit to observer frame

        In order,
        1. rotate about the z axis by an amount omega -> x1, y1, z1
        2. rotate about the x1 axis by an amount -incl -> x2, y2, z2
        3. rotate about the z2 axis by an amount Omega -> x3, y3, z3

        Args:
            x: A tensor representing the x coodinate in the plane of the
                orbit.
            y: A tensor representing the y coodinate in the plane of the
                orbit.

        Returns:
            Three tensors representing ``(X, Y, Z)`` in the observer frame.
        """

        # 1) rotate about z0 axis by omega
        if self.eccentricity is None:
            x1 = x
            y1 = y
        else:
            x1 = self.cos_omega_peri * x - self.sin_omega_peri * y
            y1 = self.sin_omega_peri * x + self.cos_omega_peri * y

        # 2) rotate about x1 axis by -incl
        # z3 = z2, subsequent rotation by Omega doesn't affect it
        if include_inclination:
            x2 = x1
            y2 = self.cos_inclination * y1
            Z = -self.sin_inclination * y1
        else:
            x2 = x1
            y2 = y1
            Z = -y1

        # 3) rotate about z2 axis by Omega
        if self.cos_asc_node is None:
            return x2, y2, Z  # type: ignore

        X = self.cos_asc_node * x2 - self.sin_asc_node * y2
        Y = self.sin_asc_node * x2 + self.cos_asc_node * y2
        return X, Y, Z  # type: ignore

    @units.quantity_input(
        t=ureg.d,
        semimajor=ureg.R_sun,
        mass=ureg.M_sun,
        parallax=ureg.arcsec,
    )
    def _get_position_and_velocity(
        self,
        t: Quantity,
        semimajor: Quantity | None = None,
        mass: Quantity | None = None,
        semiamplitude: Quantity | None = None,
        parallax: Quantity | None = None,
    ) -> tuple[
        tuple[Quantity, Quantity, Quantity], tuple[Quantity, Quantity, Quantity]
    ]:
        if self.shape != ():
            raise ValueError(
                "Cannot evaluate the position or velocity of a Keplerian 'Body' "
                "with multiple planets. Use 'jax.vmap' instead."
            )

        if semiamplitude is None:
            semiamplitude = self.radial_velocity_semiamplitude

        if parallax is None:
            parallax = self.parallax

        if semiamplitude is None:
            if self.radial_velocity_semiamplitude is None:
                m = 1 * ureg.M_sun if mass is None else mass
                k0 = 2 * jnp.pi * self.semimajor * m / (self.total_mass * self.period)
                if self.eccentricity is not None:
                    k0 /= jnpu.sqrt(1 - self.eccentricity**2)
            else:
                k0 = self.radial_velocity_semiamplitude

            if parallax is not None:
                k0 = k0.to(ureg.au / ureg.d).magnitude * parallax / ureg.day
        else:
            k0 = semiamplitude

        r0 = 1
        if semimajor is not None:
            if parallax is None:
                r0 = semimajor
            else:
                r0 = semimajor.to(ureg.au).magnitude * parallax

        sinf, cosf = self._get_true_anomaly(t)
        if self.eccentricity is None:
            v1, v2 = -k0 * sinf, k0 * cosf
        else:
            v1, v2 = -k0 * sinf, k0 * (cosf + self.eccentricity)
            r0 *= (1 - self.eccentricity**2) / (1 + self.eccentricity * cosf)

        x, y, z = self._rotate_vector(r0 * cosf, r0 * sinf)
        vx, vy, vz = self._rotate_vector(
            v1, v2, include_inclination=semiamplitude is None
        )

        # In the case a semiamplitude was passed without units, we strip the
        # units of the output
        if semiamplitude is not None and not (
            hasattr(semiamplitude, "_magnitude") and hasattr(semiamplitude, "_units")
        ):
            vx = vx.magnitude
            vy = vy.magnitude
            vz = vz.magnitude

        return (x, y, z), (vx, vy, vz)


class System(eqx.Module):
    """A Keplerian orbital system"""

    central: Central
    _body_stack: ObjectStack[OrbitalBody]

    def __init__(
        self,
        central: Central | None = None,
        *,
        bodies: Iterable[Body | OrbitalBody] = (),
    ):
        self.central = Central() if central is None else central
        self._body_stack = ObjectStack(
            *(
                b if isinstance(b, OrbitalBody) else OrbitalBody(self.central, b)
                for b in bodies
            )
        )

    def __repr__(self) -> str:
        return eqx.tree_pformat(
            self, truncate_leaf=lambda obj: isinstance(obj, ObjectStack)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._body_stack),)

    @property
    def bodies(self) -> tuple[OrbitalBody, ...]:
        return self._body_stack.objects

    @property
    def radius(self) -> Quantity:
        return self.body_vmap(lambda body: body.radius)()

    @property
    def central_radius(self) -> Quantity:
        return self.body_vmap(lambda body: body.central_radius)()

    def add_body(
        self,
        body: Body | None = None,
        central: Central | None = None,
        **kwargs: Any,
    ) -> "System":
        """Add a body to the system and return a new system

        Args:
            body (Body | None, optional): body to add. Defaults to None.
            central (Central | None, optional): TODO. Defaults to None.

        Returns:
            System: :py:class:`~jaxoplanet.orbits.keplerian.System` with the added body
        """
        body_: Body | OrbitalBody | None = body
        if body_ is None:
            body_ = Body(**kwargs)
        if central is not None:
            body_ = OrbitalBody(central, body_)
        return System(central=self.central, bodies=self.bodies + (body_,))

    def body_vmap(
        self,
        func: Callable,
        in_axes: int | None | Sequence[Any] = 0,
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
        return self._body_stack.vmap(func, in_axes=in_axes, out_axes=out_axes)

    def position(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        return self.body_vmap(OrbitalBody.position, in_axes=None)(t)

    def central_position(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        return self.body_vmap(OrbitalBody.central_position, in_axes=None)(t)

    def relative_position(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        return self.body_vmap(OrbitalBody.relative_position, in_axes=None)(t)

    def velocity(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        return self.body_vmap(OrbitalBody.velocity, in_axes=None)(t)

    def central_velocity(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        return self.body_vmap(OrbitalBody.central_velocity, in_axes=None)(t)

    def relative_velocity(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        return self.body_vmap(OrbitalBody.relative_velocity, in_axes=None)(t)

    def radial_velocity(self, t: Quantity) -> Quantity:
        return self.body_vmap(OrbitalBody.radial_velocity, in_axes=None)(t)

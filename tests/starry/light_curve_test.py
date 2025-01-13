import jax
import numpy as np
import pytest

from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.light_curves.emission import light_curve as emission_light_curve
from jaxoplanet.orbits import keplerian
from jaxoplanet.starry import Surface, Ylm
from jaxoplanet.starry.light_curves import light_curve, surface_light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.test_utils import assert_allclose
from jaxoplanet.units import unit_registry as ureg


@pytest.mark.parametrize("deg", [2, 5, 10])
@pytest.mark.parametrize("u", [[], [0.1], [0.2, 0.1]])
def test_compare_starry_limb_dark(deg, u):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # map
    inc = np.pi / 2
    np.random.seed(deg)
    y = Ylm.from_dense(np.random.randn((deg + 1) ** 2), normalize=True)
    map = Surface(y=y, u=u, inc=inc)

    # occultor
    yo = np.linspace(-3, 3, 1000)
    ro = 0.01
    zo = 2.0
    xo = 0.0

    # starry
    ms = starry.Map(ydeg=deg, udeg=len(u), inc=np.rad2deg(inc))
    ms[:, :] = y.todense()
    if len(u) > 0:
        ms[1:] = u
    expected = ms.flux(xo=xo, yo=yo, ro=ro, zo=zo)

    # jaxoplanet
    calc = jax.vmap(surface_light_curve, in_axes=(None, None, None, 0, None, None))(
        map, ro, xo, yo, zo, 0.0
    )

    assert_allclose(calc, expected)


@pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_flux(r, l_max=5, order=500):
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.multiprecision import flux as mp_flux

    # We know that these are were the errors are the highest
    b = 1 - r if r < 1 else r

    n = (l_max + 1) ** 2
    expect = np.zeros(n)
    calc = np.zeros(n)
    ys = np.eye(n, dtype=np.float64)
    ys[:, 0] = 1.0

    @jax.jit
    def light_curve(y):
        surface = Surface(y=Ylm.from_dense(y, normalize=False), normalize=False)
        return surface_light_curve(surface, y=b, z=10.0, r=r, order=order)

    for i, y in enumerate(ys):
        expect[i] = float(mp_flux.flux(ydeg=l_max)(b, r, y=y))
        calc[i] = light_curve(y)

    for n in range(expect.size):
        # For logging/debugging purposes, work out the case id
        l = np.floor(np.sqrt(n)).astype(int)  # noqa
        m = n - l**2 - l
        mu = l - m
        nu = l + m
        if mu == 1 and l == 1:
            case = 1
        elif mu % 2 == 0 and (mu // 2) % 2 == 0:
            case = 2
        elif mu == 1 and l % 2 == 0:
            case = 3
        elif mu == 1:
            case = 4
        elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
            case = 5
        else:
            case = 0

        assert_allclose(
            calc[n],
            expect[n],
            err_msg=f"n={n}, l={l}, m={m}, mu={mu}, nu={nu}, case={case}",
            atol=1e-6,
        )


@pytest.fixture(
    params=[
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
            ),
            "bodies": [{"radius": 0.5, "mass": 0.1, "period": 1.5}],
        },
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
            ),
            "central_surface": Surface(
                y=Ylm.from_dense(
                    [1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03, 0.04, 0.4, 0.2, 0.1]
                ),
                inc=0.9,
                obl=0.3,
                period=1.2,
                u=(0.1, 0.1),
            ),
            "bodies": [
                {
                    "radius": 0.5,
                    "mass": 0.1,
                    "period": 1.5,
                    "surface": Surface(
                        y=Ylm.from_dense([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03]),
                        inc=-0.3,
                        period=0.8,
                        u=(0.2, 0.3),
                    ),
                }
            ],
        },
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
            ),
            "central_surface": Surface(
                y=Ylm.from_dense(
                    [1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03, 0.04, 0.4, 0.2, 0.1]
                ),
                inc=0.9,
                obl=0.3,
                period=1.2,
                u=(0.1, 0.1),
            ),
            "bodies": [
                {
                    "radius": 0.5,
                    "mass": 0.1,
                    "period": 1.5,
                },
                {
                    "radius": 0.2,
                    "mass": 0.3,
                    "period": 0.4,
                },
            ],
        },
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
            ),
            "central_surface": Surface(
                y=Ylm.from_dense(np.hstack([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03])),
                period=1.2,
                u=(0.1, 0.1),
            ),
            "bodies": [
                {
                    "radius": 0.5,
                    "mass": 0.1,
                    "period": 1.5,
                }
            ],
        },
        {
            "central": keplerian.Central(
                mass=0.3,
                radius=0.1,
            ),
            "bodies": [
                {
                    "radius": 1.5,
                    "mass": 1.1,
                    "period": 1.5,
                    "surface": Surface(
                        y=Ylm.from_dense([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03]),
                        period=1.2,
                        u=(0.1, 0.1),
                    ),
                }
            ],
        },
    ]
)
def keplerian_system(request):
    system = SurfaceSystem(
        request.param["central"],
        central_surface=request.param.get("central_surface", None),
    )
    for body in request.param["bodies"]:
        system = system.add_body(**body)

    return system


def test_compare_starry_system(keplerian_system):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    def jaxoplanet2starry(body, surface_map=None):
        cls = (
            starry.Primary if isinstance(body, keplerian.Central) else starry.Secondary
        )
        if surface_map is None or surface_map.period is None:
            prot = 1e15
        else:
            prot = surface_map.period

        if surface_map is None:
            map_kwargs = dict(ydeg=0, rv=False, reflected=False)
        else:
            map_kwargs = dict(
                ydeg=surface_map.ydeg,
                udeg=surface_map.udeg,
                inc=np.rad2deg(surface_map.inc),
                obl=np.rad2deg(surface_map.obl),
                amp=surface_map.amplitude,
            )

        body_kwargs = dict(
            r=body.radius.magnitude,
            m=body.mass.magnitude,
            prot=prot,
        )

        if isinstance(body, keplerian.OrbitalBody):
            body_kwargs["porb"] = body.period.magnitude
            map_kwargs["amp"] = surface_map.amplitude if surface_map else 0.0

        starry_body = cls(starry.Map(**map_kwargs), **body_kwargs)

        if surface_map and surface_map.u:
            starry_body.map[1:] = surface_map.u
        if surface_map and surface_map.deg > 0:
            starry_body.map[1:, :] = np.asarray(surface_map.y.todense())[1:]

        return starry_body

    time = np.linspace(-1.5, 1.0, 300)

    # jaxoplanet
    jaxoplanet_flux = light_curve(keplerian_system)(time)

    # starry
    primary = jaxoplanet2starry(
        keplerian_system.central, keplerian_system.central_surface
    )
    secondaries = [
        jaxoplanet2starry(body, surface_map)
        for body, surface_map in zip(
            keplerian_system.bodies, keplerian_system.body_surfaces, strict=False
        )
    ]

    starry_system = starry.System(primary, *secondaries)
    starry_flux = starry_system.flux(time, total=False)

    assert_allclose(jaxoplanet_flux.T, np.array(starry_flux))

    # # compare summed jaxponet fluxes with the corresponding starry fluxes
    # if body_map is None:
    #     assert_allclose(jax.numpy.squeeze(jaxoplanet_flux), starry_flux[0])
    # elif central_map is None:
    #     assert_allclose(jax.numpy.squeeze(jaxoplanet_flux), starry_flux[1])
    # else:
    #     assert_allclose(jax.numpy.squeeze(jaxoplanet_flux),
    # np.sum(starry_flux, axis=0))

    # # compare individual jaxpolanet fluxes with the corresponding starry fluxes
    # for jaxoplanet_flux_item, starry_flux_item in zip(jaxoplanet_flux, starry_flux):
    #     if jaxoplanet_flux_item is not None:
    #         assert_allclose(jax.numpy.squeeze(jaxoplanet_flux_item), starry_flux_item)


def test_map_light_curves_none_occultor():
    surface = Surface(
        y=Ylm.from_dense(np.hstack([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03])),
        period=1.2,
        u=(0.1, 0.1),
    )

    expected = surface_light_curve(surface, theta=0.5)
    calc = surface_light_curve(surface, 0.0, 2.0, 2.0, 2.0, theta=0.5)

    assert_allclose(calc, expected)


@pytest.mark.parametrize("deg", [2, 5, 10])
def test_compare_starry_rot(deg):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # map
    inc = np.pi / 2
    np.random.seed(deg)
    y = Ylm.from_dense(np.random.randn((deg + 1) ** 2), normalize=True)
    map = Surface(y=y, inc=inc)

    # phase
    theta = np.linspace(0, 2 * np.pi, 200)

    # starry
    ms = starry.Map(ydeg=deg, inc=np.rad2deg(inc))
    ms[:, :] = y.todense()
    expected = ms.flux(theta=np.rad2deg(theta))
    calc = jax.vmap(surface_light_curve, in_axes=(None, None, None, None, None, 0))(
        map, None, None, None, None, theta
    )

    assert_allclose(calc, expected)


def test_EB():

    from jaxoplanet.units import unit_registry as ureg

    params = {
        "primary_mass": 2.0,
        "secondary_mass": 1.0,
        "primary_radius": 2.0,
        "secondary_radius": 1.0,
        "inclination": np.pi / 2,
        "period": 1.0,
        "t0": 0.0,
        "s": 0.35,
    }

    primary = keplerian.Central(
        radius=params["primary_radius"] * ureg.R_sun,
        mass=params["primary_mass"] * ureg.M_sun,
    )

    primary_amplitude = 1.1
    primary_surface = Surface(amplitude=primary_amplitude, normalize=False)

    secondary = keplerian.Body(
        radius=params["secondary_radius"] * ureg.R_sun,
        mass=params["secondary_mass"] * ureg.M_sun,
        period=params["period"] * ureg.day,
        time_transit=params["t0"] * ureg.day,
        inclination=params["inclination"] * ureg.rad,
    )

    secondary_amplitude = params["s"]
    secondary_surface = Surface(amplitude=secondary_amplitude, normalize=False)

    system = SurfaceSystem(primary, primary_surface).add_body(
        secondary, secondary_surface
    )

    def flux_function(time):
        light_curve(system)(time)

    # no occultation
    assert_allclose(
        flux_function(params["period"] / 4), [primary_amplitude, secondary_amplitude]
    )

    # primary occultation
    assert_allclose(flux_function(params["period"] / 2), [primary_amplitude, 0.0])

    # secondary occultation
    depth = (params["secondary_radius"] / params["primary_radius"]) ** 2
    assert_allclose(
        flux_function(0.0), [primary_amplitude * (1 - depth), secondary_amplitude]
    )


def test_compare_limb_dark_light_curve():
    time = np.linspace(-0.1, 0.1, 200)

    params = {
        "stellar_mass": 0.3,
        "stellar_radius": 0.3,
        "planet_radius": 0.1,
        "u": (0.1, 0.1),
        "planet_period": 15.0,
    }

    system = keplerian.System(
        keplerian.Central(mass=params["stellar_mass"], radius=params["stellar_radius"]),
    )

    system = system.add_body(
        radius=params["planet_radius"], period=params["planet_period"]
    )

    expected = limb_dark_light_curve(system, params["u"])(time)[:, 0] + 1.0

    surface_system = SurfaceSystem(
        keplerian.Central(mass=params["stellar_mass"], radius=params["stellar_radius"]),
        Surface(u=params["u"]),
    )
    surface_system = surface_system.add_body(
        radius=params["planet_radius"], period=params["planet_period"]
    )

    calc = light_curve(surface_system)(time)[:, 0]

    assert_allclose(calc, expected)


def test_EB_interchanged():

    params = {
        "primary_mass": 2.0,
        "secondary_mass": 1.0,
        "primary_radius": 2.0,
        "secondary_radius": 1.0,
        "inclination": np.pi / 2,
        "primary_amplitude": 1.1,
        "secondary_amplitude": 0.35,
        "period": 1.0,
        "t0": 0.0,
        "s": 0.35,
    }

    system_1 = SurfaceSystem(
        keplerian.Central(
            radius=params["primary_radius"] * ureg.R_sun,
            mass=params["primary_mass"] * ureg.M_sun,
        ),
        Surface(amplitude=params["primary_amplitude"], normalize=False),
    ).add_body(
        keplerian.Body(
            radius=params["secondary_radius"] * ureg.R_sun,
            mass=params["secondary_mass"] * ureg.M_sun,
            period=params["period"] * ureg.day,
            time_transit=params["t0"] * ureg.day,
            inclination=params["inclination"] * ureg.rad,
        ),
        Surface(amplitude=params["secondary_amplitude"], normalize=False),
    )

    system_2 = SurfaceSystem(
        keplerian.Central(
            radius=params["secondary_radius"] * ureg.R_sun,
            mass=params["secondary_mass"] * ureg.M_sun,
        ),
        Surface(amplitude=params["secondary_amplitude"], normalize=False),
    ).add_body(
        keplerian.Body(
            radius=params["primary_radius"] * ureg.R_sun,
            mass=params["primary_mass"] * ureg.M_sun,
            period=params["period"] * ureg.day,
            time_transit=params["t0"] * ureg.day,
            inclination=params["inclination"] * ureg.rad,
        ),
        Surface(amplitude=params["primary_amplitude"], normalize=False),
    )

    # switching primary and secondary
    time = np.linspace(0, params["period"], 200)
    flux_ordered = light_curve(system_1)(time).sum(1)
    flux_reversed = light_curve(system_2)(time + params["period"] / 2).sum(1)

    # for some reason the assert_allclose wasn't catching error here
    np.testing.assert_allclose(
        flux_ordered,
        flux_reversed,
        atol=1e-6 if flux_ordered.dtype.name == "float32" else 1e-12,
    )


@pytest.mark.parametrize("order", [20, 100, 500])
def test_light_curves_orders(order):

    central = keplerian.Central(
        radius=1.0 * ureg.R_sun,
        mass=1.0 * ureg.M_sun,
    )

    np.random.seed(42)
    y = Ylm.from_dense([1.00, *np.random.normal(0.0, 2e-2, size=15)])
    surface = Surface(inc=1.0, obl=0.2, period=27.0, u=(0.1, 0.1), y=y)

    system = SurfaceSystem(central, surface).add_body(
        semimajor=40.0 * ureg.au,
        radius=20.0 * ureg.R_earth,
        mass=1.0 * ureg.M_earth,
        impact_param=0.2,
    )

    _ = light_curve(system, order=order)(0.0)


@pytest.mark.parametrize("deg", [2, 5, 10])
def test_compare_y_from_u(deg):
    """In this test we convert the limb darkening coefficients to spherical harmonic
    coefficients and compare the light curve providing one or the other.
    """

    u = np.ones(deg)
    r = 0.1
    b = np.linspace(0, 1 + r, 20)
    surface_u = Surface(u=u)
    surface_y = Surface(y=Ylm.from_limb_darkening(u))

    light_curve_function = jax.vmap(
        lambda surface, b: surface_light_curve(surface, r, y=b), (None, 0)
    )

    flux_u = light_curve_function(surface_u, b)
    flux_y = light_curve_function(surface_y, b)

    assert_allclose(flux_u, flux_y)


@pytest.mark.parametrize(
    "params",
    [
        {
            "r": [1.0, 0.2],  # radii
            "u": [(0.1, 0.2), ()],  # limb darkening
            "a": [1.0, 0.2],  # amplitudes
        },
        {
            "r": [0.2, 1.5],  # radii
            "u": [(0.1, 0.2), ()],  # limb darkening
            "a": [0.1, 0.2],  # amplitudes
        },
        {
            "r": [1.0, 0.1, 0.5],  # radii
            "u": [(0.1, 0.2), (), (0.3, 0.4)],  # limb darkening
            "a": [1.0, 0.2, 0.5],  # amplitudes
        },
    ],
)
def test_emission_light_curve(params):

    central = keplerian.Central(radius=params["r"][0], mass=0.5)
    system = SurfaceSystem(central, Surface(u=params["u"][0], amplitude=params["a"][0]))

    for r, u, a in zip(params["r"][1:], params["u"][1:], params["a"][1:], strict=False):
        system = system.add_body(
            radius=r, surface=Surface(u=u, amplitude=a), period=1.0
        )

    period = system.bodies[0].period.magnitude
    time = np.linspace(-0.5 * period, 0.5 * period, 1000)

    expected = light_curve(system)(time)
    calc = emission_light_curve(system)(time)

    assert_allclose(calc, expected)

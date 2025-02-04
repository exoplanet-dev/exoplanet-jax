import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg


def ensure_quantity(x, unit):
    """
    Convert x to a JAX array with the given unit if it isn’t already a quantity.

    Args:
        x: A value that may or may not already be a `Quantity`.
        unit: The `pint` unit to which x should be converted.

    Returns:
        A `jnp.ndarray` with the magnitude in the specified unit.
    """
    if hasattr(x, "to"):
        return x
    else:
        return jnp.asarray(x) * unit


@units.quantity_input(
    period=ureg.d,
    duration=ureg.d,
    speed=1 / ureg.d,
    time_transit=ureg.d,
    impact_param=ureg.dimensionless,
    radius_ratio=ureg.dimensionless,
)
class TTVOrbit(TransitOrbit):
    """
    An extension of TransitOrbit that allows for transit timing variations (TTVs).

    The TTVs can be specified by:
      1. Providing a list of TTV offset arrays (`ttvs`).
      2. Providing a list of observed transit time arrays (`transit_times`),
         with optional transit indices (`transit_inds`) to label each transit.

    Either `ttvs` or `transit_times` must be provided, not both.

    Attributes:
        ttvs (list of jnp.ndarray):
            Per-planet arrays of TTV offsets (in days) or derived from linear fits.
        transit_times (list of jnp.ndarray):
            Observed or computed transit times (in days).
        transit_inds (list of jnp.ndarray):
            Transit indices corresponding to each set of transit times.
        t0 (jnp.ndarray):
            Reference transit time(s) (days, dimensionless).
        ttv_period (jnp.ndarray):
            Effective period(s) derived from transit times or supplied (days, dimensionless).
        all_transit_times (list of jnp.ndarray):
            Complete transit times (including expected and warped) for each planet.
        _bin_edges (list of jnp.ndarray):
            Internal bin edges for time-warp lookups.
        _bin_values (list of jnp.ndarray):
            Internal bin values (actual transit times) for time-warp lookups.
    """

    ttvs: list[jnp.ndarray]
    transit_times: list[jnp.ndarray]
    transit_inds: list[jnp.ndarray]
    t0: jnp.ndarray  # Reference transit time(s), stored dimensionless (days).
    ttv_period: jnp.ndarray  # The effective period(s), dimensionless (days).
    all_transit_times: list[jnp.ndarray]
    _bin_edges: list[jnp.ndarray]  # dimensionless arrays (days).
    _bin_values: list[jnp.ndarray]  # dimensionless arrays (days).

    def __init__(
        self,
        *,
        period: Quantity | None = None,
        duration: Quantity | None = None,
        speed: Quantity | None = None,
        time_transit: Quantity | None = None,
        impact_param: Quantity | None = None,
        radius_ratio: Quantity | None = None,
        # TTV-specific arguments:
        ttvs: list[Quantity] | None = None,
        transit_times: list[Quantity] | None = None,
        transit_inds: list[jnp.ndarray] | None = None,
        delta_log_period: float | None = None,
    ):
        """
        Initialize a TTVOrbit object, extending TransitOrbit to include transit timing variations.

        Args:
            period (Quantity, optional):
                The base orbital period(s) in days for each planet. If not provided and
                transit_times is given, it is inferred from a linear fit plus any
                `delta_log_period` offset.
            duration (Quantity, optional):
                The transit duration(s) in days for each planet.
            speed (Quantity, optional):
                The orbital speed in 1/days. Typically derived if `duration` is given.
            time_transit (Quantity, optional):
                The reference transit time(s) in days.
            impact_param (Quantity, optional):
                Dimensionless impact parameter.
            radius_ratio (Quantity, optional):
                Dimensionless planet/star radius ratio.
            ttvs (list of Quantity, optional):
                List of arrays with TTV offsets (days). Either `ttvs` or `transit_times`
                must be specified, not both.
            transit_times (list of Quantity, optional):
                List of arrays with observed transit times (days). If provided, a linear
                regression is done to infer period(s) and reference time(s).
            transit_inds (list of jnp.ndarray, optional):
                Per-planet indices labeling the transit number for each observed transit
                time. If not provided, they default to a simple range.
            delta_log_period (float, optional):
                Offset in natural log to be added to the derived period(s) from transit_times.
        """
        # --- Auto-convert inputs to correct units ---
        period = ensure_quantity(period, ureg.d) if period is not None else None
        duration = ensure_quantity(duration, ureg.d) if duration is not None else None
        time_transit = (
            ensure_quantity(time_transit, ureg.d) if time_transit is not None else None
        )
        impact_param = (
            ensure_quantity(impact_param, ureg.dimensionless)
            if impact_param is not None
            else None
        )
        radius_ratio = (
            ensure_quantity(radius_ratio, ureg.dimensionless)
            if radius_ratio is not None
            else None
        )

        # Convert transit_times to a list of arrays in days
        if transit_times is not None:
            if not isinstance(transit_times, (list, tuple)):
                transit_times = [transit_times]
            transit_times = [ensure_quantity(tt, ureg.d) for tt in transit_times]

        if ttvs is None and transit_times is None:
            raise ValueError("One of 'ttvs' or 'transit_times' must be provided.")

        # Prepare containers
        self.ttvs = []
        self.transit_times = []
        self.transit_inds = []

        # --- CASE 1: transit_times are provided ---
        if transit_times is not None:
            t0_list = []
            period_list = []
            # Possibly fill in transit_inds
            if transit_inds is None:
                transit_inds = [None] * len(transit_times)
            for i, times in enumerate(transit_times):
                arr_times = jnp.atleast_1d(
                    times.magnitude
                )  # dimensionless array of shape (n,)

                if transit_inds[i] is None:
                    inds = jnp.arange(arr_times.shape[0], dtype=jnp.int64)
                else:
                    inds = jnp.asarray(transit_inds[i], dtype=jnp.int64)

                # Linear regression: times ~ intercept + slope * inds
                X = jnp.vstack([jnp.ones_like(inds), inds])
                beta = jnp.linalg.solve(X @ X.T, X @ arr_times)
                intercept, slope = beta

                expect = intercept + slope * inds
                period_list.append(slope)
                t0_list.append(intercept)
                self.transit_times.append(arr_times)  # keep dimensionless
                self.transit_inds.append(inds)
                self.ttvs.append(arr_times - expect)

            # Stack up the reference times and periods
            self.t0 = jnp.stack(t0_list)  # shape (n_planets,)
            self.ttv_period = jnp.stack(period_list)

            # If single planet, make them scalars
            if self.t0.shape[0] == 1:
                self.t0 = self.t0[0]
                self.ttv_period = self.ttv_period[0]

            # If period wasn't provided, use the fitted period (+ delta_log_period if any)
            if period is None:
                if delta_log_period is not None:
                    period = jnp.exp(jnp.log(self.ttv_period) + delta_log_period)
                else:
                    period = self.ttv_period

            # time_transit is set to t0:
            time_transit = self.t0 * ureg.d

        # --- CASE 2: ttvs are provided ---
        else:
            # Convert them to dimensionless in days
            self.ttvs = [jnp.atleast_1d(ttv.magnitude) for ttv in ttvs]
            # Possibly fill in transit_inds
            if transit_inds is None:
                self.transit_inds = [jnp.arange(ttv.shape[0]) for ttv in self.ttvs]
            else:
                self.transit_inds = [
                    jnp.asarray(inds, dtype=jnp.int64) for inds in transit_inds
                ]

            if time_transit is None:
                time_transit = 0.0 * ureg.d
            self.t0 = jnp.atleast_1d(time_transit.magnitude)  # dimensionless

            # Must provide period in this mode
            if period is None:
                raise ValueError("When using ttvs, 'period' must be provided.")
            self.ttv_period = jnp.atleast_1d(period.magnitude)

        # If either t0 or ttv_period is a 1D array with length 1, squeeze it to a scalar:
        if self.t0.size == 1:
            self.t0 = self.t0.item()
        if self.ttv_period.size == 1:
            self.ttv_period = self.ttv_period.item()

        # --- Initialize the base TransitOrbit (computes speed or duration) ---
        super().__init__(
            period=period,
            duration=duration,
            speed=speed,
            time_transit=time_transit,
            impact_param=impact_param,
            radius_ratio=radius_ratio,
        )

        # If only ttvs were provided, compute the transit_times = t0 + period * inds + ttv
        if transit_times is None:
            self.transit_times = []
            for i, ttv_arr in enumerate(self.ttvs):
                if jnp.ndim(self.t0) > 0:
                    t0_i = self.t0[i]
                    period_i = self.period.magnitude[i]
                else:
                    t0_i = self.t0
                    period_i = self.period.magnitude
                inds = self.transit_inds[i]
                times = t0_i + period_i * inds + ttv_arr
                self.transit_times.append(times)

        # --- Build lookup tables (bin edges, bin values), dimensionless ---
        self.all_transit_times = []
        self._bin_edges = []
        self._bin_values = []

        # Helper to index single- or multi-planet parameters
        def get_t0(i):
            """Return the appropriate t0 for planet i, handling shape differences."""
            return self.t0[i] if self.t0.ndim and self.t0.shape[0] > 1 else self.t0

        def get_period(i):
            """Return the appropriate period for planet i, handling shape differences."""
            if hasattr(self.period, "magnitude"):
                p = self.period.magnitude
            else:
                p = self.period
            return p[i] if jnp.ndim(p) and p.shape[0] > 1 else p

        def get_ttv_period(i):
            """Return the appropriate TTV-fitted period for planet i."""
            return (
                self.ttv_period[i]
                if jnp.ndim(self.ttv_period) and self.ttv_period.shape[0] > 1
                else self.ttv_period
            )

        for i, inds in enumerate(self.transit_inds):
            t0_i = get_t0(i)
            period_i = get_period(i)
            ttv_p_i = get_ttv_period(i)

            max_ind = inds.shape[0] - 1
            expected_times = t0_i + period_i * jnp.arange(max_ind + 1, dtype=inds.dtype)

            full_transits = expected_times.at[inds].set(self.transit_times[i])
            self.all_transit_times.append(full_transits)

            # Build the bin edges
            half_p = 0.5 * ttv_p_i
            if full_transits.shape[0] > 1:
                midpoints = 0.5 * (full_transits[1:] + full_transits[:-1])
                bin_edges = jnp.concatenate(
                    (
                        jnp.array([full_transits[0] - half_p]),
                        midpoints,
                        jnp.array([full_transits[-1] + half_p]),
                    )
                )
                bin_values = jnp.concatenate(
                    (
                        jnp.array([full_transits[0]]),
                        full_transits,
                        jnp.array([full_transits[-1]]),
                    )
                )
            else:
                # Single transit index => trivial bins
                bin_edges = jnp.array(
                    [full_transits[0] - half_p, full_transits[0] + half_p]
                )
                bin_values = jnp.array([full_transits[0], full_transits[0]])

            # Store dimensionless
            self._bin_edges.append(bin_edges)
            self._bin_values.append(bin_values)

    @property
    def linear_t0(self):
        """
        Return reference transit time(s) in days (dimensionless array).

        If `self.t0` is scalar, it is wrapped in `jnp.atleast_1d`.
        """
        return jnp.atleast_1d(self.t0)

    @property
    def linear_period(self):
        """
        Return the TTV period(s) in days (dimensionless array).

        If `self.ttv_period` is scalar, it is wrapped in `jnp.atleast_1d`.
        """
        return jnp.atleast_1d(self.ttv_period)

    def _get_model_dt(self, t: Quantity) -> jnp.ndarray:
        """
        Determine the mapped transit time for each planet by searching
        which transit bin the input time belongs to.

        Args:
            t (Quantity):
                Observation times (days).

        Returns:
            jnp.ndarray:
                Dimensionless array (in days) of shape (..., n_planets) giving
                the transit time that each planet is currently “closest” to.
        """
        t_magnitude = jnp.asarray(t.to(ureg.d).magnitude)

        dt_list = []
        for i in range(len(self._bin_edges)):
            inds = jnp.searchsorted(self._bin_edges[i], t_magnitude)
            dt_i = self._bin_values[i][inds]  # dimensionless
            dt_list.append(dt_i)

        return jnp.stack(dt_list, axis=-1)  # shape (..., n_planets)

    def _warp_times(self, t: Quantity) -> Quantity:
        """
        Warp the time axis to incorporate TTVs.

        This uses the lookup from `_get_model_dt` to shift each input time
        to the “nearest” transit time for each planet.

        Args:
            t (Quantity):
                The times at which to evaluate the TTV-warped orbit (days).

        Returns:
            Quantity:
                The warped times (days) accounting for TTVs.
        """
        dt = self._get_model_dt(t)  # dimensionless shape (..., n_planets)
        # If only one planet, drop the trailing axis
        if dt.shape[-1] == 1:
            dt = dt[..., 0]

        # Convert self.t0 to dimensionless array
        t0 = jnp.atleast_1d(self.t0)
        return t - (dt - t0) * ureg.d

    @units.quantity_input(t=ureg.d, parallax=ureg.arcsec)
    def relative_position(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """
        Override the base relative_position method to warp the time axis first.

        Args:
            t (Quantity):
                Times at which to compute the orbit (days).
            parallax (Quantity, optional):
                Parallax angle if relevant.

        Returns:
            (x, y, z): tuple of Quantity
                The 3D position in the star‐centric frame, with shape matching `t`.
        """
        warped_t = self._warp_times(t)
        x, y, z = super().relative_position(warped_t, parallax=parallax)
        return jnpu.squeeze(x), jnpu.squeeze(y), jnpu.squeeze(z)


def compute_expected_transit_times(min_time, max_time, period, t0):
    """
    Compute the expected transit times within a dataset.

    Args:
        min_time (float):
            The start time of the dataset (days).
        max_time (float):
            The end time of the dataset (days).
        period (array-like):
            The orbital period(s) for one or more planets (days).
        t0 (array-like):
            The reference transit times for those planets (days).

    Returns:
        A list of arrays, where each array contains the transit times
        between `min_time` and `max_time` for the corresponding planet.
    """
    periods = jnp.atleast_1d(period)
    t0s = jnp.atleast_1d(t0)
    transit_times = []
    for p, t0_ in zip(periods, t0s, strict=False):
        min_ind = jnp.floor((min_time - t0_) / p)
        max_ind = jnp.ceil((max_time - t0_) / p)
        times = t0_ + p * jnp.arange(min_ind, max_ind, 1)
        times = times[(min_time <= times) & (times <= max_time)]
        transit_times.append(times)
    return transit_times

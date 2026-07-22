"""
Wraps a pretrained color-magnitude (CMD) normalizing flow (see
``pretrain_bkg_cmd_flow.py``) for direct use as a coordinate distribution in a
``stream_membership.ModelComponent``, operating on raw (unstandardized) data.

``pretrain_bkg_cmd_flow.py`` trains ``FlowDensity`` (from
``stream_membership.distributions``) on *standardized* (zero mean, unit
variance) color/magnitude/phi1 values, for numerical stability during
training. ``CalibratedFlowDensity`` below composes that ``FlowDensity`` with
the saved standardization statistics so it can be constructed and called
directly on raw (BP-RP, G) values and raw phi1, like every other coordinate
distribution in this codebase. It also applies the log-Jacobian correction
for the standardization, so the returned log-density is on the same physical
scale (mag^-1 dex^-1) as if the flow had been fit directly to raw data --
this matters because the mixture model combines this term additively (in log
space) with the other, already-physical-unit coordinate terms (pm1, pm2, rv,
...) when computing each star's relative membership probability.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from stream_membership.distributions import FlowDensity


class CalibratedFlowDensity(dist.Distribution):
    """
    A ``FlowDensity`` wrapped with affine standardization of both the modeled
    (event) variable and the conditioning variable, so it can be used
    directly on raw-unit data.

    Follows the same conditioning convention as the rest of this codebase's
    distributions (e.g. ``NormalSpline``, ``FlowDensity``): ``x`` is fixed at
    construction time but can be overridden by passing a new ``x`` to
    ``.log_prob()`` / ``.sample()``.

    Parameters
    ----------
    flow
        A ``flowjax`` distribution trained on standardized inputs (see
        ``pretrain_bkg_cmd_flow.py``).
    norm_stats
        Dictionary as saved by ``pretrain_bkg_cmd_flow.save_flow`` (in
        ``<out>_meta.json``), i.e.
        ``{"bp_rp": {"mean": ..., "std": ...}, "phot_g_mean_mag": {...}, "phi1": {...}}``.
    x
        Raw (unstandardized) conditioning array, e.g. phi1 in degrees.
    event_columns
        Names (keys into ``norm_stats``) of the modeled event dimensions, in
        the same order as the flow's event axis. Defaults to
        ``("bp_rp", "phot_g_mean_mag")``, matching ``pretrain_bkg_cmd_flow``.
    cond_column
        Name (key into ``norm_stats``) of the conditioning variable.
    """

    def __init__(
        self,
        flow,
        norm_stats: dict[str, dict[str, float]],
        x: ArrayLike | None = None,
        *,
        event_columns: tuple[str, ...] = ("g0", "r0"),
        cond_column: str = "phi1",
        validate_args: bool | None = None,
    ) -> None:
        self.norm_stats = norm_stats
        self.event_columns = event_columns
        self.cond_column = cond_column

        self._loc = jnp.array([norm_stats[c]["mean"] for c in event_columns])
        self._scale = jnp.array([norm_stats[c]["std"] for c in event_columns])
        # log|d(raw)/d(standardized)|, constant since the map is affine.
        self._log_jac = jnp.sum(jnp.log(self._scale))

        x_std = None if x is None else self._standardize_cond(jnp.asarray(x))
        self._flow_density = FlowDensity(flow, x=x_std, validate_args=validate_args)

        super().__init__(
            batch_shape=self._flow_density.batch_shape,
            event_shape=self._flow_density.event_shape,
            validate_args=validate_args,
        )

    def _standardize_cond(self, x: jax.Array) -> jax.Array:
        s = self.norm_stats[self.cond_column]
        return (x - s["mean"]) / s["std"]

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """Draws raw-unit (magnitude, magnitude) samples from the flow."""
        x_std = None if x is None else self._standardize_cond(jnp.asarray(x))
        std_sample = self._flow_density.sample(key, sample_shape=sample_shape, x=x_std)
        return std_sample * self._scale + self._loc

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """Evaluates the log probability density for raw-unit (magnitude, magnitude) values."""
        value = jnp.asarray(value)
        std_value = (value - self._loc) / self._scale
        x_std = None if x is None else self._standardize_cond(jnp.asarray(x))
        return self._flow_density.log_prob(std_value, x=x_std) - self._log_jac

    @property
    def support(self):
        return self._flow_density.support

import functools
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
from stream_membership import ComponentMixtureModel, ModelComponent
from stream_membership.model import FromDist
from stream_membership.distributions import (
    IndependentGMM,
    IsochroneCMD,
    KROUPA_SLOPES,
    KROUPA_BREAKS,
    SALPETER_SLOPES,
    SALPETER_BREAKS,
    TruncatedNormalSpline,
    TruncatedNormalGMMConditional,
    DirichletSpline,
    TruncatedNormal1DSplineMixture,
    LinearGradientSplineWithGap,
    QuadraticGradientSplineWithGap,
)
from scipy.stats import binned_statistic
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from cmd_flow_density import CalibratedFlowDensity, FrozenLogProb

from cats.inputs import stream_inputs

# Coordinate key used for the (color, magnitude) CMD term. A tuple key is
# used because this is a joint 2D distribution over (mag1, mag2),
# following the same convention as e.g. IndependentGMM's ("phi1", "phi2") joint
# coordinate in make_offtrack_model_component.
CMD_COORD = ("PS_g", "PS_r")


def _phi1_key(coord_name):
    """Convert a coordinate name (a plain string like "pm1", or a joint/
    tuple coordinate like ``CMD_COORD``) into the ``data`` dict key used to
    look up *that coordinate's own* phi1-conditioning subsample -- e.g.
    "pm1" -> "pm1_phi1", ``("PS_g", "PS_r")`` -> "PS_g-PS_r_phi1".

    This is the other half of the per-coordinate-plate architecture change
    in ``stream_membership.model.ComponentMixtureModel.__call__``: since not
    every star has proper motions/radial velocities/photometry, each
    coordinate's spline/mixture terms need to be conditioned on ("x"'d
    against) *that coordinate's own* valid-star phi1 values, not one shared
    full-sample phi1 array -- otherwise the conditioning array and the
    observed values passed to `__call__` would have mismatched lengths.
    """
    name = coord_name if isinstance(coord_name, str) else "-".join(coord_name)
    return f"{name}_phi1"


def _coord_x(data, coord_name):
    """Get the phi1 array to condition ``coord_name``'s distribution on.

    Uses the coordinate's own valid-subsample phi1 array if present in
    ``data`` under the ``<coord>_phi1`` convention (see `_phi1_key`);
    otherwise falls back to the full ``data["phi1"]`` array. The fallback
    preserves the old, single-shared-N behavior for any caller that hasn't
    been updated to pass per-coordinate subsamples, so this is backwards
    compatible.
    """
    return data.get(_phi1_key(coord_name), data["phi1"])


def _phi2_key(coord_name):
    """Mirror of `_phi1_key`, but for *phi2* -- e.g. "pm1" -> "pm1_phi2".

    Needed alongside `_phi1_key`/`_coord_x` once a coordinate's
    `conditioning_dist` is D=2 (a joint (phi1,phi2) `IndependentGMM`, e.g.
    `make_offtrack_model_component`'s spatial grid) rather than D=1: the
    `TruncatedNormalGMMConditional`'s own `x` then needs each coordinate's
    own valid-star *(phi1, phi2) pairs*, not phi1 alone -- see `_coord_x2`.
    """
    name = coord_name if isinstance(coord_name, str) else "-".join(coord_name)
    return f"{name}_phi2"


def _coord_x2(data, coord_name):
    """Get the phi2 array to condition ``coord_name``'s distribution on.

    Mirror of `_coord_x`: uses the coordinate's own valid-subsample phi2
    array if present in ``data`` under the ``<coord>_phi2`` convention (see
    `_phi2_key`); otherwise falls back to the full ``data["phi2"]`` array.
    """
    return data.get(_phi2_key(coord_name), data["phi2"])


def _valid_mask_key(coord_name):
    """Convert a coordinate name into the ``data`` dict key used to look up
    *that coordinate's own* full-length boolean valid-star mask -- e.g.
    "pm1" -> "pm1_valid".

    Needed to scatter a ragged coordinate's (shorter) per-star log-probs
    back into a full-length (``len(data["phi1"])``-shaped) array -- e.g. in
    ``make_lp_dict``/``_make_lp_dict`` when computing total per-star
    membership log-posteriors across a mixture's components, some of which
    have coordinates (pm1/pm2/radial_velocity) that aren't measured for
    every star. See ``_scatter_ragged_lp``.
    """
    name = coord_name if isinstance(coord_name, str) else "-".join(coord_name)
    return f"{name}_valid"


def _scatter_ragged_lp(full_len, mask, ragged_values, fill_value=0.0):
    """Scatter a ragged coordinate's per-star values back into a full-length
    array, aligned with the original (unfiltered) star ordering.

    Parameters
    ----------
    full_len : int
        Length of the full (unraveled) sample, e.g. ``len(data["phi1"])``.
    mask : array of bool, shape (full_len,)
        This coordinate's own valid-star mask (see `_valid_mask_key`) --
        ``True`` at the positions (in the full sample) that survived into
        the ragged subsample.
    ragged_values : array, shape (mask.sum(),)
        The ragged coordinate's own per-star values (e.g. log-probs),
        already aligned with the *order* of ``True`` entries in ``mask``.
    fill_value : float
        Value used for stars that don't have this coordinate at all (e.g.
        no radial-velocity measurement). Defaults to 0.0 so that a missing
        coordinate simply contributes nothing to a summed total
        log-posterior, rather than being (incorrectly) compared against a
        value it doesn't have.

    Returns
    -------
    array, shape (full_len,)
    """
    full = jnp.full(full_len, fill_value, dtype=ragged_values.dtype)
    valid_idx = jnp.flatnonzero(mask, size=ragged_values.shape[0])
    return full.at[valid_idx].set(ragged_values)


def load_stream_isochrone(json_path):
    """
    Load the fixed SSP isochrone track produced by
    ``scripts/generate_stream_isochrone.py`` (age=12 Gyr, [Fe/H]=-2.5,
    matching ``cats.inputs.stream_inputs["GD-1"]``), for use as the
    ``IsochroneCMD`` stream CMD term in ``make_stream_model_component``.

    Note: ``generate_stream_isochrone.py`` requires network access (to fetch
    MIST grid data) that isn't available in every environment -- see that
    script's docstring. If ``json_path`` doesn't exist, run it elsewhere and
    copy the resulting ``*.json`` file here.

    Returns
    -------
    dict with keys "track_abs_mag", "track_color", "distmod_coeffs" (jnp
    arrays), ready to be splatted into ``IsochroneCMD``'s fixed constructor
    arguments (see how it's used below, bound via ``functools.partial``).
    """
    with open(json_path) as f:
        meta = json.load(f)
    iso = {
        "track_abs_mag": jnp.array(meta["abs_mag1"]),
        "track_color": jnp.array(meta["abs_mag1"]) - jnp.array(meta["abs_mag2"]),
        "distmod_coeffs": jnp.array(meta["distmod_coeffs"]), # QUITE SPECIFIC TO GD-1
    }
    # Initial stellar masses along the track, used to build a real
    # (IMF-weighted) luminosity function in ``IsochroneCMD`` -- see
    # ``make_stream_model_component``'s ``luminosity_function`` argument.
    if "initial_mass" in meta:
        iso["track_mass"] = jnp.array(meta["initial_mass"])
    return iso


def make_mag_limit_selection_function(
    data, mag_key="PS_g", phi1_bin=10.0, quantile=0.98, width=0.15,
    min_stars=50,
):
    """
    Build a ``MagLimSelectionFunction`` (for ``make_stream_model_component``'s
    ``selection_function=``) from the sample itself: the limiting magnitude
    ``m_lim(phi1)`` (the 50%-completeness point of a logistic completeness
    curve) is estimated per ``phi1_bin``-degree bin as the ``quantile`` of
    ``data[mag_key]`` -- i.e. where the observed magnitude distribution ends.
    This is an approximation for a sample with a hard faint cut (e.g. your
    Gaia G < 20.5 cut, which is a soft edge in PS_g): ``quantile=0.98`` puts
    ``m_lim`` just inside the edge, and ``width`` (mag) is the edge's
    softness. m_lim is taken independent of phi2. Bins with fewer than
    ``min_stars`` stars fall back to the global quantile. Check the printed
    values against the cut you expect, or pass your own
    ``MagLimSelectionFunction`` if you have a depth map.
    """
    import numpy as np
    from stream_membership.distributions import MagLimSelectionFunction

    phi1 = np.asarray(_coord_x(data, CMD_COORD))
    phi2 = np.asarray(_coord_x2(data, CMD_COORD))
    mag = np.asarray(data[mag_key])
    edges = np.arange(phi1.min(), phi1.max() + phi1_bin, phi1_bin)
    centers = 0.5 * (edges[:-1] + edges[1:])
    glob = np.quantile(mag, quantile)
    mlim = np.array([
        np.quantile(mag[(phi1 >= lo) & (phi1 < hi)], quantile)
        if ((phi1 >= lo) & (phi1 < hi)).sum() >= min_stars else glob
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    print(f"make_mag_limit_selection_function: m_lim(phi1) in "
          f"[{mlim.min():.2f}, {mlim.max():.2f}] over {len(centers)} phi1 bins "
          f"(global {glob:.2f}), width={width}")
    if len(centers) < 2:  # need >= 2 nodes for the bilinear grid
        centers = np.array([phi1.min(), phi1.max() + 1e-3]); mlim = np.repeat(mlim[:1], 2)
    phi2_nodes = np.array([phi2.min(), phi2.max() + 1e-3])
    return MagLimSelectionFunction(
        phi1_nodes=centers, phi2_nodes=phi2_nodes,
        mlim_grid=np.repeat(mlim[:, None], 2, axis=1), width=width,
        mag_min=float(mag.min()) - 1.0, mag_max=float(mag.max()) + 1.0,
    )

def get_bounds_and_grids(data, pawprint):
    
    phi1_lim = (jnp.min(data['phi1']), jnp.max(data['phi1']))
    phi2_lim = (jnp.min(data['phi2']), jnp.max(data['phi2']))

    # NOTE: pm1_lim/pm2_lim used to be set from the (rough, coarse) proper
    # motion polygon vertices alone. That polygon is only meant for an early
    # rough on-track/off-track cut, so it can be narrower than the actual
    # pm1/pm2 range spanned by `data` (which includes off-track background
    # stars, by construction more scattered in proper motion than on-track
    # stream stars). These pm1_lim/pm2_lim bounds are used downstream as the
    # *truncation* bounds (`low`/`high`) for the background model's
    # TruncatedNormal1DSplineMixture over pm1/pm2.
    #
    # This matters for the "FloatingPointError: invalid value (nan)
    # encountered in mul" SVI crash: the real root cause (fixed at its
    # source in stream_membership/distributions/normal_spline.py's
    # TruncatedNormalSpline._make_helper_dist) is that the spline's
    # interpolated `loc` can drift far outside [low, high] during
    # optimization, which underflows the TruncatedNormal's normalizing
    # constant and produces a NaN gradient. Widening pm1_lim/pm2_lim here to
    # also cover the real data range (unioned with the polygon range) is a
    # complementary mitigation, not the primary fix: it narrows the gap
    # between this truncation window and the much wider `clip_locs=(-10,
    # 10)` bound on `loc` used in make_bkg_model_component() below, making
    # it less likely `loc` needs to travel far to reach the true background
    # population's pm1/pm2 range in the first place, and it also guarantees
    # no real background data point is itself outside [low, high].
    pm1_lim = (
        jnp.minimum(jnp.min(pawprint.pmprint.vertices[:, 0]), jnp.min(data['pm1'])),
        jnp.maximum(jnp.max(pawprint.pmprint.vertices[:, 0]), jnp.max(data['pm1'])),
    )
    pm2_lim = (
        jnp.minimum(jnp.min(pawprint.pmprint.vertices[:, 1]), jnp.min(data['pm2'])),
        jnp.maximum(jnp.max(pawprint.pmprint.vertices[:, 1]), jnp.max(data['pm2'])),
    )
    rv_lim = (jnp.min(data['radial_velocity']), jnp.max(data['radial_velocity']))

    coord_bounds = {"phi1": phi1_lim, "phi2": phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim, "radial_velocity": rv_lim}
    
    plot_grids = {
    "phi1": jnp.linspace(*phi1_lim, 512),
    "phi2": jnp.linspace(*phi2_lim, 128),
    "pm1": jnp.linspace(*pm1_lim, 128),
    "pm2": jnp.linspace(*pm2_lim, 128),
    "radial_velocity": jnp.linspace(*rv_lim, 128)
}
    return coord_bounds, plot_grids

def interpolate_stream_tracks(stream_data, phi1_lim):
    # NOTE: each coordinate's phi1 values are looked up via `_coord_x`, so
    # this works whether `stream_data` has one shared "phi1" array (old,
    # single-N behavior) or per-coordinate "<coord>_phi1" subsamples (new
    # ragged-plate behavior, see `_phi1_key`/`_coord_x` above) -- the two
    # arrays passed into each `binned_statistic` call just need to be the
    # same length and co-indexed, which is guaranteed either way.
    _phi2_stat = binned_statistic(_coord_x(stream_data, "phi2"), stream_data["phi2"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 21))
    _phi2_interp = IUS(
        0.5 * (_phi2_stat.bin_edges[:-1] + _phi2_stat.bin_edges[1:]), _phi2_stat.statistic, ext=0, k=3
    )

    _pm1_stat = binned_statistic(_coord_x(stream_data, "pm1"), stream_data["pm1"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _pm1_interp = IUS(
        0.5 * (_pm1_stat.bin_edges[:-1] + _pm1_stat.bin_edges[1:]), _pm1_stat.statistic, ext=0, k=3
    )

    _pm2_stat = binned_statistic(_coord_x(stream_data, "pm2"), stream_data["pm2"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _pm2_interp = IUS(
        0.5 * (_pm2_stat.bin_edges[:-1] + _pm2_stat.bin_edges[1:]), _pm2_stat.statistic, ext=0, k=3
    )

    _rv_stat = binned_statistic(_coord_x(stream_data, "radial_velocity"), stream_data["radial_velocity"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _rv_interp = IUS(
        0.5 * (_rv_stat.bin_edges[:-1] + _rv_stat.bin_edges[1:]), _rv_stat.statistic, ext=0, k=3
    )

    interp_dict = {'phi2': _phi2_interp, 'pm1': _pm1_interp, 'pm2': _pm2_interp, 'radial_velocity': _rv_interp}

    return interp_dict

def _make_jax_gap_center_spline(x_data, y_data, phi1_lim, n_bins=21, k=3):
    """
    Same phi1-binned-mean smoothing `interpolate_stream_tracks` uses, for
    the phi2 gap-center track consumed by `LinearGradientSplineWithGap`.

    NOTE (reverted): this originally built the result with `jax_cosmo`'s
    JAX-native spline instead of scipy's `IUS`, on the reasoning that
    `gap_center_spl` gets called on `phi1` from inside
    `LinearGradientSplineWithGap.log_prob`/`.sample` -- i.e. inside
    numpyro's jitted/autodiff'd SVI loop -- and a plain scipy spline raises
    `jax.errors.TracerArrayConversionError` there under `jax.jit`/`jax.grad`
    in an isolated reproduction. In practice, on the real pipeline here,
    switching to the JAX spline broke other code downstream that expects a
    scipy `InterpolatedUnivariateSpline` object specifically (e.g. its
    scipy-only methods/attributes), and going back to plain `IUS` is what
    actually runs end to end -- so this is intentionally back to scipy for
    now. Revisit if/when the downstream scipy-specific usage is updated to
    tolerate (or is given) a JAX-native spline instead.
    """
    stat = binned_statistic(x_data, y_data, bins=jnp.linspace(phi1_lim[0], phi1_lim[1], n_bins))
    bin_centers = 0.5 * (stat.bin_edges[:-1] + stat.bin_edges[1:])
    return IUS(bin_centers, stat.statistic, k=k)

def make_bkg_model_component(coords, knot_spacings, n_mixture, coord_bounds, data, stream_data=None, cmd_flow=None, cmd_norm_stats=None, phi2_poly_degree=1, gap_renormalize=True):
    '''
    gap_renormalize : bool
        Whether the background phi2 term divides its density by ``1 - P_gap``
        (default True: original behavior, correct for FITTING to gap-excised
        ``bkg_data``). Pass False for the background component that goes into
        the bkg+stream mixture and is evaluated on ``run_data`` (which still
        contains the corridor) -- otherwise the bkg density is over-normalized
        by ``1/(1 - P_gap)`` and the fitted stream fraction is biased low. See
        ``LinearGradientSplineWithGap``'s ``gap_renormalize`` docstring.
    phi2_poly_degree : {1, 2}
        Degree of the background's phi2 density polynomial (in the Legendre
        basis, per phi1 knot). 1 (default) is the original
        ``LinearGradientSplineWithGap`` (tilt ``a_vals`` only). 2 uses
        ``QuadraticGradientSplineWithGap``, which adds a curvature spline
        ``b_vals`` (new sampled site; init to zeros to start from the linear
        model, e.g. ``bkg_init_params['phi2']['b_vals'] = jnp.zeros(n_phi2_knots)``).
    stream_data:
        Required if ``'phi2' in coords`` (the background phi2 term is a
        ``LinearGradientSplineWithGap``, which needs to know where the
        on-sky-masked corridor sits). Must be the actual stream-member
        sample -- NOT ``data`` (the background-only sample this function is
        otherwise fit on). ``data`` has, by construction, had the corridor
        already excluded from it, so using it to estimate *where the
        corridor is* silently gives the wrong location: it just tracks
        whatever background stars happen to remain on either side, which
        has no principled relationship to the actual excluded region. Using
        the real stream track (or, more precisely, the on-sky mask/pawprint
        geometry used to build ``data`` in the first place) is what
        `LinearGradientSplineWithGap`'s docstring means by deriving the gap
        location "from the known selection/mask geometry ... not from a
        fitted stream track" -- ``stream_data``'s own (observed, not SVI-fit)
        track is used here as a close, already-available proxy for that.
    cmd_flow, cmd_norm_stats (optional):
        A pretrained, phi1-conditioned flowjax flow and its accompanying
        normalization stats (as produced/saved by
        ``pretrain_bkg_cmd_flow.py`` / loaded via its ``load_flow``), used to
        add a probabilistic photometric (mag1, mag2) term to the
        background model. If either is None (the default), no CMD term is added
        and this function behaves exactly as before. When provided, ``data``
        must also include magnitude columns (aligned with "phi1", etc.).
    '''


    phi1_lim = coord_bounds['phi1']
    if 'phi2' in coords: phi2_lim = coord_bounds['phi2']
    if 'pm1' in coords: pm1_lim = coord_bounds['pm1']
    if 'pm2' in coords: pm2_lim = coord_bounds['pm2']
    if 'radial_velocity' in coords: rv_lim = coord_bounds['radial_velocity']

    phi1_knot_spacing, phi2_knot_spacing, pm1_knot_spacing, pm2_knot_spacing, rv_knot_spacing = knot_spacings

    bkg_phi1_knots = jnp.arange(jnp.around(phi1_lim[0]),# + phi1_knot_spacing/2,
                                jnp.around(phi1_lim[1])+1e-3,# - phi1_knot_spacing/2 + 1e-3,
                                phi1_knot_spacing)
    bkg_phi2_knots = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, phi2_knot_spacing)
    bkg_pm1_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm1_knot_spacing)
    bkg_pm2_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm2_knot_spacing)
    bkg_rv_knots   = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, rv_knot_spacing)

    if 'phi2' in coords:
        # TODO: make this more general so it can be used for other coordinates
        if stream_data is None:
            msg = (
                "make_bkg_model_component(coords=[..., 'phi2', ...]) needs "
                "`stream_data` (the real stream-member sample) to locate the "
                "on-sky-masked corridor for the LinearGradientSplineWithGap "
                "phi2 term -- see this function's docstring."
            )
            raise ValueError(msg)
        # phi2_gap_center_spl = _make_jax_gap_center_spline(
        #     stream_data["phi1"], stream_data["phi2"], phi1_lim
        # )
        phi2_gap_center_spl = interpolate_stream_tracks(stream_data, phi1_lim)['phi2']

    coord_distributions = {
        "phi1": IndependentGMM,
    }
    if 'phi2' in coords:
        if phi2_poly_degree not in (1, 2):
            raise ValueError(f"phi2_poly_degree must be 1 or 2, got {phi2_poly_degree!r}")
        coord_distributions["phi2"] = (
            LinearGradientSplineWithGap if phi2_poly_degree == 1 else QuadraticGradientSplineWithGap
        )
    if 'pm1' in coords: coord_distributions["pm1"] = TruncatedNormal1DSplineMixture
    if 'pm2' in coords: coord_distributions["pm2"] = TruncatedNormal1DSplineMixture
    if 'radial_velocity' in coords: coord_distributions["radial_velocity"] = TruncatedNormal1DSplineMixture

    conditional_data = {}
    if 'phi2' in coords: conditional_data["phi2"] = {"x": "phi1"}
    if 'pm1' in coords: conditional_data["pm1"] = {"x": "phi1"}
    if 'pm2' in coords: conditional_data["pm2"] = {"x": "phi1"}
    if 'radial_velocity' in coords: conditional_data["radial_velocity"] = {"x": "phi1"}


    coord_parameters = {
            "phi1": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(len(bkg_phi1_knots))),
                ),
                "locs": bkg_phi1_knots.reshape(1, -1),
                "scales": dist.TruncatedNormal(loc=0.5*phi1_knot_spacing, scale=0.5*phi1_knot_spacing,
                                               low=0.1*phi1_knot_spacing).expand([bkg_phi1_knots.shape[0]]),
                "low":  jnp.array([phi1_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1]])[:, None],                  
            }
    }
    if 'phi2' in coords:
        coord_parameters["phi2"] = {
                "a_vals": dist.TruncatedNormal(loc=0, scale=0.1,
                                               low=-0.2, high=0.2).expand([bkg_phi2_knots.shape[0]]),
                "knots": bkg_phi2_knots,
                "x": _coord_x(data, "phi2"),
                "low": phi2_lim[0],
                "high": phi2_lim[1],
                "gap_center_spl": phi2_gap_center_spl,
                "gap_half_width": stream_inputs['GD-1']['width']/2,
                "spline_k": 3,
                "clip_a": (-1+1e-3,1-1e-3),
                "gap_renormalize": gap_renormalize,
            }
        if phi2_poly_degree == 2:
            coord_parameters["phi2"]["b_vals"] = dist.TruncatedNormal(
                loc=0, scale=0.1, low=-0.2, high=0.2,
            ).expand([bkg_phi2_knots.shape[0]])
    if 'pm1' in coords:
        coord_parameters["pm1"] = {
                # NOTE: mixing_vals makes the mixing weights between the
                # n_mixture background components a smooth spline function of
                # phi1 (evaluated at the same bkg_pm1_knots used for
                # loc_vals/scale_vals below), instead of a single global
                # weight shared across all phi1. Each row is a per-component
                # sequence of knot logits; they're deterministically
                # spline-interpolated and softmaxed inside
                # TruncatedNormal1DSplineMixture (see
                # spline_mixture.py:_make_mixing_distribution). A weakly
                # informative Normal(0, 1) prior on the raw logits is used
                # here (analogous to loc_vals/scale_vals), rather than a
                # Dirichlet draw, since these are unconstrained spline control
                # points, not a probability simplex themselves.
                "mixing_vals": dist.Normal(0, 1).expand([n_mixture, bkg_pm1_knots.shape[0]]),
                "loc_vals": dist.Uniform(low=-10, high=10).expand([n_mixture, bkg_pm1_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=4,scale=3,low=2.78e-3).expand([n_mixture, bkg_pm1_knots.shape[0]]),
                "knots": bkg_pm1_knots,
                "x": _coord_x(data, "pm1"),
                "low": pm1_lim[0],
                "high": pm1_lim[1],
                "spline_k": 3,
                "clip_locs": (-20,20),
                "clip_scales": (2.78e-3, None),
                # Components are ordered/identified by loc, not a forced scale
                # ordering -- see initialization_helpers.relabel_by_loc_continuity.
                # scale_vals is already positive-constrained on its own via the
                # TruncatedNormal prior above and clip_scales, so ordered_scales
                # isn't needed for positivity, only (optionally) identifiability.
                "ordered_scales": False,
        }
    if 'pm2' in coords:
        coord_parameters["pm2"] = {
                # See the matching NOTE on "mixing_vals" in coord_parameters["pm1"]
                # above -- this makes the pm2 background mixing weights vary
                # smoothly with phi1 too.
                "mixing_vals": dist.Normal(0, 1).expand([n_mixture, bkg_pm2_knots.shape[0]]),
                "loc_vals": dist.Uniform(low=-10, high=10).expand([n_mixture, bkg_pm2_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=4,scale=3,low=2.78e-3).expand([n_mixture, bkg_pm2_knots.shape[0]]),
                "knots": bkg_pm2_knots,
                "x": _coord_x(data, "pm2"),
                "low": pm2_lim[0],
                "high": pm2_lim[1],
                "spline_k": 3,
                "clip_locs": (-10, 10),
                "clip_scales": (2.78e-3, None),
                "ordered_scales": False,
        }
    if 'radial_velocity' in coords:
        coord_parameters["radial_velocity"] = {
                # See the matching NOTE on "mixing_vals" in coord_parameters["pm1"]
                # above -- this makes the radial_velocity background mixing
                # weights vary smoothly with phi1 too.
                "mixing_vals": dist.Normal(0, 1).expand([n_mixture, bkg_rv_knots.shape[0]]),
                "loc_vals": dist.Uniform(low=-500, high=500).expand([n_mixture, bkg_rv_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=100,scale=100,low=0.1).expand([n_mixture, bkg_rv_knots.shape[0]]),
                "knots": bkg_rv_knots,
                "x": _coord_x(data, "radial_velocity"),
                "low": rv_lim[0], #-500
                "high": rv_lim[1], #500
                "spline_k": 3,
                "clip_locs": (-500, 500),
                "clip_scales": (0.1, None),
                "ordered_scales": False,
            }


    if cmd_flow is not None and cmd_norm_stats is not None:
        # SELECTION-FUNCTION NOTE: unlike the stream's IsochroneCMD term
        # (see its `selection_function` parameter), this background CMD
        # term deliberately gets no explicit selection-function correction.
        # `cmd_flow` is fit by maximum likelihood directly on real,
        # off-stream data (see pretrain_bkg_cmd_flow.py) -- i.e. on data
        # that has *already* been thinned by the survey's selection
        # function -- so a flexible density estimator trained that way
        # converges to the *observed* (selection-convolved) density
        # directly; it has no "true", pre-selection density to correct.
        # This assumes the flow is only ever evaluated on data from the
        # same field/depth it was trained on -- if it's ever reused across
        # fields with meaningfully different depths, that assumption breaks
        # and this reasoning would need revisiting.
        #
        # PERFORMANCE NOTE: unlike the stream's IsochroneCMD term, this
        # background CMD term has no learnable parameters attached at all --
        # `x` is fixed data (`data["phi1"]`), and `cmd_flow`/`cmd_norm_stats`
        # are frozen/pretrained. That means its log_prob is an exact
        # constant for the *entire* SVI run (identical every step, every
        # particle) -- but a real `CalibratedFlowDensity` still runs a full
        # forward pass through an 8-layer masked-autoregressive spline flow
        # every time `log_prob` is called, which dominates per-step cost
        # (measured 10x+ slower SVI with this term included vs. without
        # it). So we evaluate it ONCE here, up front, and swap in
        # `FrozenLogProb` -- a trivial lookup of the precomputed array --
        # for the model that SVI actually optimizes. This changes nothing
        # about the fit (the term contributes exactly zero gradient either
        # way); it just avoids recomputing it ~(num_particles * num_steps)
        # times for no benefit.
        #
        # If you need to *sample* from the background CMD flow (e.g. for
        # diagnostic CMD plots), construct a real
        # `CalibratedFlowDensity(cmd_flow, cmd_norm_stats, x=...)` directly
        # instead of pulling it out of `bkg_model` -- `FrozenLogProb`
        # doesn't implement `.sample()`.
        _cmd_value = jnp.stack([data["PS_g"], data["PS_r"]], axis=-1)
        _cmd_log_probs = CalibratedFlowDensity(
            flow=cmd_flow, norm_stats=cmd_norm_stats, x=_coord_x(data, CMD_COORD)
        ).log_prob(_cmd_value)

        coord_distributions[CMD_COORD] = functools.partial(
            FrozenLogProb, log_probs=_cmd_log_probs
        )
        coord_parameters[CMD_COORD] = {}
        conditional_data[CMD_COORD] = {}

    bkg_model = ModelComponent(
        name="background",
        coord_distributions=coord_distributions,
        coord_parameters=coord_parameters,
        conditional_data=conditional_data,
    )

    return bkg_model

def make_stream_model_component(
    knot_spacings, coord_bounds, data, rv_min, rv_max, stream_isochrone=None,
    selection_function=None, luminosity_function="kroupa",
    off_track_scale=0.3, dm_offset_bound=1.0, color_offset_bound=0.2,
    color_scale_max=0.2,
):
    '''
    off_track_scale : float | None
        ``IsochroneCMD``'s ``off_track_scale`` (mag): smooth, steep penalty for
        stars whose implied absolute magnitude is beyond the track's ends,
        instead of the old constant (free, unnormalized) density with
        straight-through gradients. Default 0.3; ``None`` = old behavior.
    dm_offset_bound, color_offset_bound, color_scale_max : float | None
        Hard bounds (mag) on the CMD nuisance parameters' priors:
        ``dm_offset`` in +/-dm_offset_bound, ``color_offset`` in
        +/-color_offset_bound, ``color_scale`` in (5e-3, color_scale_max).
        Without bounds, a stream-only fit on a contaminated sample can slide
        ``dm_offset`` to many mag and inflate ``color_scale`` into a broad
        outlier blob (observed: dm_offset = -16). ``None`` = unbounded (old).
    luminosity_function : None | "kroupa" | "salpeter" | dict
        Marginal density of the stream's CMD term in absolute magnitude (only
        has an effect if ``stream_isochrone`` is provided). ``"kroupa"``
        (default) / ``"salpeter"`` / ``dict(slopes=(...), breaks=(...))``
        weight the isochrone track by an initial mass function
        (``dN/dM = xi(m)|dm/dM|``, slopes alpha_i with ``xi ~ m^-alpha_i``;
        Kroupa = slopes (0.3, 1.3, 2.3), breaks (0.08, 0.5) Msun; Salpeter =
        slope 2.35) instead of the old flat-in-absolute-magnitude density.
        The flat density puts most of the stream's CMD mass at sparse bright
        magnitudes and under-weights the faint main sequence where nearly all
        observed stream stars are, which pushes the fitted stream fraction
        toward zero. Needs ``stream_isochrone["track_mass"]`` (re-run
        ``load_stream_isochrone`` on a json from
        ``generate_stream_isochrone.py``, which saves ``initial_mass``); if
        missing, falls back to flat with a warning. Pass ``None`` for the old
        flat behavior.
    stream_isochrone (optional):
        A dict as returned by ``load_stream_isochrone`` (fixed
        "track_abs_mag", "track_color", "distmod_coeffs" arrays), used to add
        a probabilistic single-stellar-population (SSP) isochrone CMD
        (color, absolute magnitude) term to the stream model, following the
        stream CMD approach of Starkman et al. 2025 -- a fixed isochrone
        track shape (see ``scripts/generate_stream_isochrone.py``) shifted
        along phi1 by the fixed Valluri et al. 2024 apparent
        distance-modulus track, with a fit intrinsic color scatter. If None
        (the default), no CMD term is added and this function behaves
        exactly as before. When provided, ``data`` must also include
        magnitude columns (aligned with "phi1", etc.).
    selection_function (optional):
        A fixed, pretrained ``MagLimSelectionFunction`` (see
        ``stream_membership.distributions.selection_function``), used to
        correct the ``IsochroneCMD`` stream CMD term for survey
        incompleteness at faint magnitudes -- see that class's and
        ``IsochroneCMD``'s docstrings for the full reasoning. Only has any
        effect if ``stream_isochrone`` is also provided. Defaults to None,
        in which case this function behaves exactly as before (no
        selection correction, no extra phi2 conditioning array needed) --
        so existing calls/notebooks don't need to change at all unless you
        want to turn this on.
    '''

    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']
    pm1_lim  = coord_bounds['pm1']
    pm2_lim  = coord_bounds['pm2']
    rv_lim   = coord_bounds['radial_velocity']

    phi1_knot_spacing, phi2_knot_spacing, pm1_knot_spacing, pm2_knot_spacing, rv_knot_spacing = knot_spacings

    stream_phi1_knots = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, phi1_knot_spacing)
    stream_phi2_knots = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, phi2_knot_spacing)
    stream_pm1_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm1_knot_spacing)
    stream_pm2_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm2_knot_spacing)
    stream_rv_knots   = jnp.arange(rv_min, rv_max, rv_knot_spacing)

    _interp_dict = interpolate_stream_tracks(data, phi1_lim)
    eval_interp_phi2 = jnp.array(_interp_dict['phi2'](stream_phi2_knots))
    eval_interp_pm1 = jnp.array(_interp_dict['pm1'](stream_pm1_knots))
    eval_interp_pm2 = jnp.array(_interp_dict['pm2'](stream_pm2_knots))
    eval_interp_rv = jnp.array(_interp_dict['radial_velocity'](stream_rv_knots))
    
    coord_distributions = {
        "phi1": IndependentGMM,
        "phi2": TruncatedNormalSpline,
        "pm1": TruncatedNormalSpline,
        "pm2": TruncatedNormalSpline,
        "radial_velocity": TruncatedNormalSpline,
    }
    coord_parameters = {
            "phi1": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(len(stream_phi1_knots))),
                ),
                "locs": stream_phi1_knots.reshape(1, -1),
                "scales": dist.TruncatedNormal(loc=0.5*phi1_knot_spacing, scale=0.5*phi1_knot_spacing,
                                               low=0.1*phi1_knot_spacing).expand([stream_phi1_knots.shape[0]]),
                "low": jnp.array([phi1_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1]])[:, None],
            },
            "phi2": {
                # "loc_vals": dist.Uniform(*phi2_lim).expand([stream_phi2_knots.shape[0]]),
                "loc_vals": dist.TruncatedNormal(loc=eval_interp_phi2, scale=1,low=phi2_lim[0],high=phi2_lim[1]).expand([stream_phi2_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=0.5,scale=0.5,low=0.05).expand([stream_phi2_knots.shape[0]]),
                "knots": stream_phi2_knots,
                "x": _coord_x(data, "phi2"),
                "low": phi2_lim[0],
                "high": phi2_lim[1],
                "spline_k": 3,
                "clip_scales": (0.05, None),
            },
            "pm1": {
                # "loc_vals": dist.Uniform(*pm1_lim).expand([stream_pm1_knots.shape[0]]),
                "loc_vals": dist.TruncatedNormal(loc=eval_interp_pm1, scale=2, low=pm1_lim[0],high=pm1_lim[1]).expand([stream_pm1_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=0.1,scale=0.5,low=2.78e-3, high=2).expand([stream_pm1_knots.shape[0]]),
                "knots": stream_pm1_knots,
                "x": _coord_x(data, "pm1"),
                "low": pm1_lim[0],
                "high": pm1_lim[1],
                "spline_k": 3,
                "clip_locs": pm1_lim,
                "clip_scales": (2.78e-3, 2),
            },
            "pm2": {
                # "loc_vals": dist.Uniform(*pm2_lim).expand([stream_pm2_knots.shape[0]]),
                "loc_vals": dist.TruncatedNormal(loc=eval_interp_pm2, scale=2, low=pm2_lim[0],high=pm2_lim[1]).expand([stream_pm2_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=0.1,scale=0.5,low=2.78e-3, high=2).expand([stream_pm2_knots.shape[0]]),
                "knots": stream_pm2_knots,
                "x": _coord_x(data, "pm2"),
                "low": pm2_lim[0],
                "high": pm2_lim[1],
                "spline_k": 3,
                "clip_locs": pm2_lim,
                "clip_scales": (2.78e-3, 2),
            },
            "radial_velocity": {
                "loc_vals": dist.Uniform(*rv_lim).expand([stream_rv_knots.shape[0]]),
                # "loc_vals": dist.TruncatedNormal(loc=eval_interp_rv, scale=200, low=-500,high=500).expand([stream_rv_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=1,scale=5,low=0.1, high=100).expand([stream_rv_knots.shape[0]]),
                "knots": stream_rv_knots,
                "x": _coord_x(data, "radial_velocity"),
                "low": -500, #rv_lim[0],
                "high": 500, #rv_lim[1],
                "spline_k": 1,
                "clip_locs": (-500,500), #rv_lim,
                "clip_scales": (0.1, 100),
            },
    }
    conditional_data = {"phi2": {"x": "phi1"},
                          "pm1": {"x": "phi1"},
                          "pm2": {"x": "phi1"},
                          "radial_velocity": {"x": "phi1"}}

    if stream_isochrone is not None:
        # NOTE: see the matching NOTE in make_bkg_model_component -- fixed
        # (non-numpyro) track/distmod arrays must be bound via
        # functools.partial rather than placed directly in
        # coord_parameters[CMD_COORD], since ModelComponent.make_dists
        # interprets any *dict-valued* entry as kwargs for a nested
        # numpyro.sample() call. Only "x" (a plain array) and the fit
        # nuisance priors (dm_offset, color_offset, color_scale) go in
        # coord_parameters[CMD_COORD].
        lf_kwargs = {}
        if luminosity_function is not None and selection_function is None:
            warnings.warn(
                "A real luminosity function puts most of the stream CMD model's mass at "
                "faint, unobservable magnitudes (only ~6% of it is brighter than g=20.5 "
                "for Kroupa, vs ~53% for the flat default). Without a selection_function "
                "(see make_mag_limit_selection_function) the CMD term is then under-"
                "normalized over the observed range, which drives the fitted stream "
                "fraction toward zero. Pass selection_function=..., or "
                "luminosity_function=None for the old flat behavior.",
                stacklevel=2,
            )
        if luminosity_function is not None:
            if "track_mass" not in stream_isochrone:
                warnings.warn(
                    "luminosity_function was requested but stream_isochrone has "
                    "no 'track_mass' (older isochrone dict) -- falling back to a "
                    "flat luminosity function. Reload it with "
                    "load_stream_isochrone from a json that has 'initial_mass'.",
                    stacklevel=2,
                )
            else:
                presets = {
                    "kroupa": (KROUPA_SLOPES, KROUPA_BREAKS),
                    "salpeter": (SALPETER_SLOPES, SALPETER_BREAKS),
                }
                if isinstance(luminosity_function, str):
                    slopes, breaks = presets[luminosity_function.lower()]
                else:
                    slopes = luminosity_function["slopes"]
                    breaks = luminosity_function.get("breaks", ())
                lf_kwargs = dict(
                    track_mass=stream_isochrone["track_mass"],
                    imf_slopes=tuple(slopes),
                    imf_breaks=tuple(breaks),
                )
        coord_distributions[CMD_COORD] = functools.partial(
            IsochroneCMD,
            off_track_scale=off_track_scale,
            track_abs_mag=stream_isochrone["track_abs_mag"],
            track_color=stream_isochrone["track_color"],
            distmod_coeffs=stream_isochrone["distmod_coeffs"],
            **lf_kwargs,
            # `selection_function=None` (the default) reproduces the exact
            # old behavior with zero extra computation -- see this
            # function's docstring and IsochroneCMD's docstring.
            selection_function=selection_function,
        )
        coord_parameters[CMD_COORD] = {
            "x": _coord_x(data, CMD_COORD),
            # Small, weakly-informative priors letting the joint fit
            # calibrate for any (phi1-independent) mismatch between the
            # fixed Valluri+24 distance track / MIST isochrone and the real
            # GD-1 data, without turning the distance track into a free
            # phi1-dependent spline (per the modeling choice to keep this
            # fixed, given the modest ~1.5k-star stream sample).
            "dm_offset": (
                dist.Normal(loc=0.0, scale=0.3) if dm_offset_bound is None else
                dist.TruncatedNormal(loc=0.0, scale=0.3, low=-dm_offset_bound, high=dm_offset_bound)
            ),
            "color_offset": (
                dist.Normal(loc=0.0, scale=0.05) if color_offset_bound is None else
                dist.TruncatedNormal(loc=0.0, scale=0.05, low=-color_offset_bound, high=color_offset_bound)
            ),
            "color_scale": (
                dist.TruncatedNormal(loc=0.05, scale=0.05, low=5e-3) if color_scale_max is None else
                dist.TruncatedNormal(loc=0.05, scale=0.05, low=5e-3, high=color_scale_max)
            ),
        }
        conditional_data[CMD_COORD] = {"x": "phi1"}
        if selection_function is not None:
            # Only add the phi2 conditioning array if it's actually needed
            # -- keeps the `selection_function=None` path identical to the
            # pre-existing behavior (no extra data threaded through at all).
            coord_parameters[CMD_COORD]["phi2"] = _coord_x2(data, CMD_COORD)
            conditional_data[CMD_COORD]["phi2"] = "phi2"

    stream_model = ModelComponent(
        name="stream",
        coord_distributions=coord_distributions,
        coord_parameters=coord_parameters,
        conditional_data=conditional_data,
    )

    return stream_model

def make_stream_track(stream_model, stream_params, coords=("pm1", "pm2", "radial_velocity")):
    """Build the fitted stream's phi1-dependent (loc, scale) track for each coord.

    Evaluates the stream's own `TruncatedNormalSpline` interpolants (same knots,
    spline order, endpoint handling and clipping as the model uses) from the
    fitted per-knot `loc_vals`/`scale_vals` in ``stream_params`` (e.g.
    ``no_off_params["stream"]``). Pass the result as ``stream_track`` to
    `make_offtrack_model_component`.

    Returns ``{coord: callable(phi1) -> (loc, scale)}``.
    """
    from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as JIUS

    track = {}
    for coord in coords:
        cp = stream_model.coord_parameters[coord]
        knots = jnp.asarray(cp["knots"])
        k = cp.get("spline_k", 3)
        k_loc, k_scale = (k["loc"], k["scale"]) if isinstance(k, dict) else (k, k)
        loc_vals = jnp.asarray(stream_params[coord]["loc_vals"])
        scale_vals = jnp.asarray(stream_params[coord]["scale_vals"])
        loc_spl = JIUS(knots, loc_vals, k=k_loc, endpoints="not-a-knot")
        scale_spl = JIUS(knots, scale_vals, k=k_scale)
        clip_locs = cp.get("clip_locs", (-jnp.inf, jnp.inf))
        clip_scales = cp.get("clip_scales", (0.0, jnp.inf))

        def _eval(phi1, loc_spl=loc_spl, scale_spl=scale_spl,
                  clip_locs=clip_locs, clip_scales=clip_scales):
            phi1 = jnp.asarray(phi1)
            lo = -jnp.inf if clip_locs[0] is None else clip_locs[0]
            hi = jnp.inf if clip_locs[1] is None else clip_locs[1]
            slo = 0.0 if clip_scales[0] is None else clip_scales[0]
            shi = jnp.inf if clip_scales[1] is None else clip_scales[1]
            return (jnp.clip(loc_spl(phi1), lo, hi),
                    jnp.clip(scale_spl(phi1), slo, shi))

        track[coord] = _eval
    return track


def evaluate_stream_track_at_nodes(stream_track, offtrack_phi12_locs, min_sigma=None):
    """Per-node (center, sigma) of the stream track at each off-track node's phi1.

    ``offtrack_phi12_locs`` is the (K, 2) array returned by
    `make_offtrack_model_component`. Returns ``{coord: (center (K,), sigma (K,))}``
    with the same ``min_sigma`` floor that `make_offtrack_model_component` applies
    -- use it to build in-bounds initial values for the off-track
    ``loc_vals``/``scale_vals``.
    """
    min_sigma = min_sigma or {}
    phi1 = jnp.asarray(offtrack_phi12_locs)[:, 0]
    out = {}
    for coord, fn in stream_track.items():
        center, sigma = fn(phi1)
        out[coord] = (center, jnp.maximum(sigma, min_sigma.get(coord, 0.0)))
    return out


def make_offtrack_model_component(
    offtrack_dx,
    stream_model,
    coord_bounds,
    data,
    stream_kinematics=None,
    n_sigma=3.0,
    lattice="rectangular",
    fixed_scales=False,
    sigma_frac=1.0,
    weight_prior="dirichlet",
    logit_scale=2.0,
    stream_track=None,
    track_min_sigma=None,
    track_scale_max_factor=3.0,
):
    '''
    offtrack_dx: [offtrack_phi1_dx, offtrack_phi2_dx]
        Sets the spacing (and therefore number of nodes) of the phi1 and
        phi2 grid axes, respectively. For ``lattice="rectangular"`` the full
        offtrack spatial grid is the K1 x K2 meshgrid of the two axes' node
        sets (K = K1 * K2 total nodes); each axis's own dx also seeds that
        axis's per-node width prior (how tightly each node is expected to be
        localized around its grid location) unless ``fixed_scales=True``.
        For ``lattice="diamond"`` see below.

    lattice : {"rectangular", "diamond"}
        Spatial node layout. ``"rectangular"`` (default) is the original
        meshgrid, unchanged. ``"diamond"`` is a 45-degree-rotated
        (checkerboard / staggered-row) lattice: nodes sit at
        ``(phi1_min + i*dx1/2, phi2_min + j*dx2/2)`` for all (i, j) with
        i + j even, so same-row neighbors are ``dx1`` apart (matching a
        rectangular row) but every node's nearest neighbors are the four
        diagonal ones at ``(+/-dx1/2, +/-dx2/2)`` -- twice the node density of
        the rectangular grid at the same ``offtrack_dx`` (so pass larger dx
        for the same K), and better coverage of diagonal off-track features.
        Unlike the rectangular grid, the diamond lattice spans the full
        phi1/phi2 bounds (including the edge rows/columns).
    fixed_scales : bool
        If False (default; original behavior), each node's 2D width is a
        sampled parameter with a TruncatedNormal prior centered on dx. If
        True, the widths are FIXED (not sampled) at
        ``sigma_frac * (dx1, dx2)`` for the rectangular lattice or
        ``sigma_frac * (dx1/2, dx2/2)`` for the diamond lattice (i.e. in
        units of the lattice's half-spacings per axis), leaving only the
        node weights as fitted spatial parameters. ``sigma_frac ~ 1`` gives
        roughly ripple-free summed Gaussians on the diamond lattice.
    sigma_frac : float
        See ``fixed_scales``. Ignored if ``fixed_scales`` is False.
    weight_prior : {"dirichlet", "softmax"}
        Prior on the K node weights. ``"dirichlet"`` (default; original
        behavior) is ``Dirichlet(1, ..., 1)`` (sampled through AutoNormal's
        order-dependent stick-breaking transform; poorly conditioned and
        gradient-saturating at large K). ``"softmax"`` instead samples
        iid ``Normal(0, logit_scale)`` log-amplitudes and sets
        ``weights = softmax(log_amplitudes)`` (`dist.CategoricalLogits`):
        symmetric in the nodes, no ordering dependence, and a non-saturating
        Normal restoring pull on every node. NOTE the sampled site is then
        named for logits, so any init/packed params for
        ``("phi1","phi2")``'s ``"mixing_distribution"`` must be log-amplitudes
        (e.g. zeros for uniform weights), NOT probabilities.
    logit_scale : float
        Std of the iid Normal log-amplitude prior when
        ``weight_prior="softmax"`` (weight dynamic range ~ exp(+/-2*scale)).

    data : dict
        The full per-coordinate data dict (as built by `table_to_arrays` /
        `_table_to_ragged_data`), needed here so pm1/pm2/radial_velocity can
        each be conditioned on *their own* valid-star (phi1, phi2) pairs via
        `_coord_x`/`_coord_x2` -- see the NOTE below.

    stream_kinematics : dict[str, tuple[float, float]], optional
        Maps any of "pm1", "pm2", "radial_velocity" to a (mean, std) pair
        giving the stream's own derived/fitted value for that coordinate
        (e.g. computed from an earlier no-offtrack joint bkg+stream fit --
        see `no_off_params["stream"]` in the Stage 4 notebook). When a
        coordinate is present here, every offtrack node's own loc_vals prior
        for that coordinate is a `TruncatedNormal` centered at `mean` and
        truncated to `mean +/- n_sigma*std` -- i.e. each node is encouraged
        to stay close to (but not forced identical to) the stream's fit,
        while still allowed to differ from node to node (and from the
        stream). Any of "pm1"/"pm2"/"radial_velocity" *not* present in this
        dict falls back to an uninformative prior spanning that coordinate's
        own bounds. Default None (no coordinates recentered).
    n_sigma : float
        Width (in standard deviations) of the truncation window used for any
        coordinate provided via `stream_kinematics` or `stream_track`.
        Default 3.0.

    stream_track : dict[str, callable], optional
        Output of `make_stream_track`: maps "pm1"/"pm2"/"radial_velocity" to a
        callable ``phi1 -> (loc, scale)`` giving the fitted stream's local
        track and dispersion. For every coordinate present here (this takes
        precedence over `stream_kinematics`), node k's ``loc_vals[k]`` gets a
        HARD window ``center_k +/- n_sigma * sigma_k`` where center_k/sigma_k
        are the stream's track and dispersion evaluated at the node's own
        phi1 -- so off-track kinematics can never stray from the local stream
        kinematics by more than ``n_sigma`` stream dispersions. The prior is
        ``TruncatedNormal(center_k, sigma_k)`` inside that window. Node
        ``scale_vals[k]`` is ``TruncatedNormal(sigma_k, sigma_k)`` on
        ``[0.05 sigma_k, track_scale_max_factor * sigma_k]`` (set
        ``track_scale_max_factor=None`` for no upper bound), since an arbitrarily
        broad node would otherwise reach well outside the window. Initial
        values must lie inside these per-node bounds; see
        `evaluate_stream_track_at_nodes`.
    track_min_sigma : dict[str, float], optional
        Per-coordinate floor on sigma_k for `stream_track` (e.g. when the
        stream's fitted dispersion is clipped to tiny values). Default none.
    track_scale_max_factor : float or None
        See `stream_track`. Default 3.0.

    NOTE on the model this implements: offtrack is a genuine joint 2D
    IndependentGMM grid over (phi1, phi2) -- a K1 x K2 meshgrid of spatial
    nodes -- restoring the original 2D grid's full multimodal-phi2
    representational capacity (needed since offtrack stars are *not*
    confined to a single phi2 track at fixed phi1). Unlike the original
    pre-redesign version, though, pm1/pm2/radial_velocity are each their own
    independent `TruncatedNormalGMMConditional` with fully free per-node
    values, sharing the spatial grid's exact sampled weights/locs/scales
    (via `FromDist(("phi1", "phi2"))`) to compute per-star responsibilities
    over all K = K1*K2 nodes -- so two nodes at the same phi1 but different
    phi2 each get their own independent kinematics, rather than being forced
    to share identical pm1/pm2/rv (the original bug this redesign fixes).
    '''
    stream_kinematics = stream_kinematics if stream_kinematics is not None else {}

    offtrack_phi1_dx, offtrack_phi2_dx = offtrack_dx

    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']
    pm1_lim = coord_bounds['pm1']
    pm2_lim = coord_bounds['pm2']
    rv_lim = coord_bounds['radial_velocity']

    if lattice not in ("rectangular", "diamond"):
        raise ValueError(f"lattice must be 'rectangular' or 'diamond', got {lattice!r}")
    if weight_prior not in ("dirichlet", "softmax"):
        raise ValueError(
            f"weight_prior must be 'dirichlet' or 'softmax', got {weight_prior!r}"
        )

    if lattice == "rectangular":
        offtrack_phi1_nodes = jnp.arange(
            phi1_lim[0] + offtrack_phi1_dx,
            phi1_lim[1] - offtrack_phi1_dx / 2 + 1e-3,  # factor of two added by hand to make sure the high phi1 edge is properly included
            offtrack_phi1_dx,
        )
        offtrack_phi2_nodes = jnp.arange(
            phi2_lim[0] + offtrack_phi2_dx,
            phi2_lim[1] - offtrack_phi2_dx / 2 + 1e-3,  # factor of two added by hand to make sure the high phi2 edge is properly included
            offtrack_phi2_dx,
        )
        K1 = offtrack_phi1_nodes.shape[0]
        K2 = offtrack_phi2_nodes.shape[0]
        K = K1 * K2

        # Full K1 x K2 meshgrid of (phi1, phi2) node locations, raveled into the
        # (2, K) locs shape IndependentGMM expects for a D=2 grid.
        phi1_grid, phi2_grid = jnp.meshgrid(
            offtrack_phi1_nodes, offtrack_phi2_nodes, indexing="ij",
        )
        offtrack_phi12_locs = jnp.stack([phi1_grid.ravel(), phi2_grid.ravel()], axis=0)  # (2, K)
        half_dx = jnp.array([offtrack_phi1_dx, offtrack_phi2_dx])  # per-axis unit for fixed scales
    else:
        # Diamond (checkerboard) lattice: rectangular grid of half-spacings
        # (a, b), keeping only nodes with i + j even.
        a = float(offtrack_phi1_dx) / 2
        b = float(offtrack_phi2_dx) / 2
        n1 = int(jnp.floor((float(phi1_lim[1]) - float(phi1_lim[0])) / a + 1e-6)) + 1
        n2 = int(jnp.floor((float(phi2_lim[1]) - float(phi2_lim[0])) / b + 1e-6)) + 1
        ii, jj = jnp.meshgrid(jnp.arange(n1), jnp.arange(n2), indexing="ij")
        keep = ((ii + jj) % 2 == 0).ravel()
        offtrack_phi12_locs = jnp.stack(
            [
                (phi1_lim[0] + a * ii.ravel())[keep],
                (phi2_lim[0] + b * jj.ravel())[keep],
            ],
            axis=0,
        )  # (2, K)
        K = offtrack_phi12_locs.shape[1]
        half_dx = jnp.array([a, b])

    stream_track = stream_track if stream_track is not None else {}
    if stream_track:
        _node_track = evaluate_stream_track_at_nodes(
            stream_track, offtrack_phi12_locs.T, track_min_sigma
        )

    def _loc_vals_prior(coord_name, lim):
        """Per-node stream-track window if `stream_track` has this coord;
        else uninformative Uniform(*lim) by default, or -- if the caller
        supplied a (mean, std) via `stream_kinematics` -- a TruncatedNormal
        recentered on the stream's global value and truncated to
        +/- n_sigma standard deviations."""
        if coord_name in stream_track:
            center, sigma = _node_track[coord_name]
            return dist.TruncatedNormal(
                loc=center, scale=sigma,
                low=center - n_sigma * sigma, high=center + n_sigma * sigma,
            )
        if coord_name in stream_kinematics:
            mean, std = stream_kinematics[coord_name]
            return dist.TruncatedNormal(
                loc=mean, scale=std,
                low=mean - n_sigma * std, high=mean + n_sigma * std,
            ).expand([K])
        return dist.Uniform(low=lim[0], high=lim[1]).expand([K])

    def _scale_vals_prior(coord_name, lim):
        """TruncatedNormal scale prior, per-node from `stream_track`, else
        informed by the stream's global fitted dispersion (`stream_kinematics`),
        else a generic prior sized to ~10% of the coordinate's bounds."""
        if coord_name in stream_track:
            _, sigma = _node_track[coord_name]
            high = None if track_scale_max_factor is None else track_scale_max_factor * sigma
            return dist.TruncatedNormal(
                loc=sigma, scale=sigma, low=0.05 * sigma, high=high,
            )
        if coord_name in stream_kinematics:
            _, std = stream_kinematics[coord_name]
            scale_loc = std
        else:
            scale_loc = 0.1 * (lim[1] - lim[0])
        return dist.TruncatedNormal(
            loc=scale_loc, scale=scale_loc, low=0.05 * scale_loc,
        ).expand([K])

    ## NOTE: This model is tied to the stream model only for the CMD term
    ##       (see below) -- the spatial term is a genuine joint (phi1, phi2)
    ##       grid, and pm1/pm2/radial_velocity are each their own per-node
    ##       TruncatedNormalGMMConditional (see the module NOTE above),
    ##       sharing the grid's sampled weights/locs/scales via FromDist.
    coord_distributions = {
        ("phi1", "phi2"): IndependentGMM,
        "pm1": TruncatedNormalGMMConditional,
        "pm2": TruncatedNormalGMMConditional,
        "radial_velocity": TruncatedNormalGMMConditional,
    }

    dx_per_axis = jnp.array([offtrack_phi1_dx, offtrack_phi2_dx])[:, None]  # (2, 1)
    if weight_prior == "softmax":
        mixing_spec = (
            dist.CategoricalLogits,
            dist.Normal(0.0, logit_scale).expand([K]),
        )
    else:
        mixing_spec = (dist.Categorical, dist.Dirichlet(jnp.full(K, 1)))

    if fixed_scales:
        scales_spec = jnp.broadcast_to((sigma_frac * half_dx)[:, None], (2, K))
    else:
        scales_spec = dist.TruncatedNormal(
            loc=dx_per_axis, scale=dx_per_axis, low=0.1 * dx_per_axis,
        ).expand([2, K])

    coord_parameters = {
        ("phi1", "phi2"): {
            "mixing_distribution": mixing_spec,
            "locs": offtrack_phi12_locs,
            "scales": scales_spec,
            "low": jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
            "high": jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
        },
        "pm1": {
            "conditioning_dist": FromDist(("phi1", "phi2")),
            "loc_vals": _loc_vals_prior("pm1", pm1_lim),
            "scale_vals": _scale_vals_prior("pm1", pm1_lim),
            "x": jnp.stack(
                [_coord_x(data, "pm1"), _coord_x2(data, "pm1")], axis=-1,
            ),
            "low": pm1_lim[0],
            "high": pm1_lim[1],
        },
        "pm2": {
            "conditioning_dist": FromDist(("phi1", "phi2")),
            "loc_vals": _loc_vals_prior("pm2", pm2_lim),
            "scale_vals": _scale_vals_prior("pm2", pm2_lim),
            "x": jnp.stack(
                [_coord_x(data, "pm2"), _coord_x2(data, "pm2")], axis=-1,
            ),
            "low": pm2_lim[0],
            "high": pm2_lim[1],
        },
        "radial_velocity": {
            "conditioning_dist": FromDist(("phi1", "phi2")),
            "loc_vals": _loc_vals_prior("radial_velocity", rv_lim),
            "scale_vals": _scale_vals_prior("radial_velocity", rv_lim),
            "x": jnp.stack(
                [
                    _coord_x(data, "radial_velocity"),
                    _coord_x2(data, "radial_velocity"),
                ],
                axis=-1,
            ),
            "low": rv_lim[0],
            "high": rv_lim[1],
        },
    }
    conditional_data = {
        "pm1": {"x": "phi1"},
        "pm2": {"x": "phi1"},
        "radial_velocity": {"x": "phi1"},
    }

    if CMD_COORD in stream_model.coord_distributions:
        # `ComponentMixtureModel` requires every component in a mixture to
        # share the same `coord_names` (see
        # `stream_membership.model.ComponentMixtureModel.__post_init__`), so
        # once the stream model has a CMD_COORD term (see
        # `make_stream_model_component`'s `stream_isochrone` argument), the
        # offtrack component -- which is combined with the stream model in
        # Stage 4's full mixture -- must have one too, or the mixture
        # construction will fail. Physically, offtrack stars are stream
        # stars that have scattered spatially off the main track, so it's
        # natural to tie the offtrack CMD term directly to the stream's
        # (fixed-track) IsochroneCMD term -- unlike pm1/pm2/rv above, this
        # one stays a direct reuse of the stream's own single term (i.e. it
        # does *not* get its own per-node freedom); revisit this if offtrack
        # stars' colors/magnitudes should also vary node-to-node.
        coord_distributions[CMD_COORD] = stream_model.coord_distributions[CMD_COORD]
        coord_parameters[CMD_COORD] = stream_model.coord_parameters[CMD_COORD]
        conditional_data[CMD_COORD] = {"x": "phi1"}

    offtrack_model = ModelComponent(
        name="offtrack",
        coord_distributions=coord_distributions,
        coord_parameters=coord_parameters,
        conditional_data=conditional_data,
    )

    # (K, 2), not (2, K): matches how downstream code (e.g. the Stage 4
    # notebook's diagnostic scatter plot) indexes this as
    # `offtrack_phi12_locs[:, 0]` / `[:, 1]` for the phi1/phi2 columns of
    # each of the K grid nodes.
    return offtrack_model, offtrack_phi12_locs.T
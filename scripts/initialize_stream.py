import functools
import json
from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
from stream_membership import ComponentMixtureModel, ModelComponent
from stream_membership.distributions import (
    IndependentGMM,
    IsochroneCMD,
    TruncatedNormalSpline,
    DirichletSpline,
    TruncatedNormal1DSplineMixture,
    LinearGradientSplineWithGap
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
    return {
        "track_abs_mag": jnp.array(meta["abs_mag1"]),
        "track_color": jnp.array(meta["abs_mag1"]) - jnp.array(meta["abs_mag2"]),
        "distmod_coeffs": jnp.array(meta["distmod_coeffs"]), # QUITE SPECIFIC TO GD-1
    }

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
    rv_lim = (jnp.min(data['rv']), jnp.max(data['rv']))

    coord_bounds = {"phi1": phi1_lim, "phi2": phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim, "rv": rv_lim}
    
    plot_grids = {
    "phi1": jnp.linspace(*phi1_lim, 512),
    "phi2": jnp.linspace(*phi2_lim, 128),
    "pm1": jnp.linspace(*pm1_lim, 128),
    "pm2": jnp.linspace(*pm2_lim, 128),
    "rv": jnp.linspace(*rv_lim, 128)
}
    return coord_bounds, plot_grids

def interpolate_stream_tracks(stream_data, phi1_lim):
    _phi2_stat = binned_statistic(stream_data["phi1"], stream_data["phi2"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 21))
    _phi2_interp = IUS(
        0.5 * (_phi2_stat.bin_edges[:-1] + _phi2_stat.bin_edges[1:]), _phi2_stat.statistic, ext=0, k=3
    )
    
    _pm1_stat = binned_statistic(stream_data["phi1"], stream_data["pm1"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _pm1_interp = IUS(
        0.5 * (_pm1_stat.bin_edges[:-1] + _pm1_stat.bin_edges[1:]), _pm1_stat.statistic, ext=0, k=3
    )
    
    _pm2_stat = binned_statistic(stream_data["phi1"], stream_data["pm2"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _pm2_interp = IUS(
        0.5 * (_pm2_stat.bin_edges[:-1] + _pm2_stat.bin_edges[1:]), _pm2_stat.statistic, ext=0, k=3
    )

    _rv_stat = binned_statistic(stream_data["phi1"], stream_data["rv"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _rv_interp = IUS(
        0.5 * (_rv_stat.bin_edges[:-1] + _rv_stat.bin_edges[1:]), _rv_stat.statistic, ext=0, k=3
    )

    interp_dict = {'phi2': _phi2_interp, 'pm1': _pm1_interp, 'pm2': _pm2_interp, 'rv': _rv_interp}

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

def make_bkg_model_component(coords, knot_spacings, n_mixture, coord_bounds, data, stream_data=None, cmd_flow=None, cmd_norm_stats=None):
    '''
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
    if 'rv' in coords: rv_lim = coord_bounds['rv']

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
    if 'phi2' in coords: coord_distributions["phi2"] = LinearGradientSplineWithGap
    if 'pm1' in coords: coord_distributions["pm1"] = TruncatedNormal1DSplineMixture
    if 'pm2' in coords: coord_distributions["pm2"] = TruncatedNormal1DSplineMixture
    if 'rv' in coords: coord_distributions["rv"] = TruncatedNormal1DSplineMixture

    conditional_data = {}
    if 'phi2' in coords: conditional_data["phi2"] = {"x": "phi1"}
    if 'pm1' in coords: conditional_data["pm1"] = {"x": "phi1"}
    if 'pm2' in coords: conditional_data["pm2"] = {"x": "phi1"}
    if 'rv' in coords: conditional_data["rv"] = {"x": "phi1"}


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
                "x": data["phi1"],
                "low": phi2_lim[0],
                "high": phi2_lim[1],
                "gap_center_spl": phi2_gap_center_spl,
                "gap_half_width": stream_inputs['GD-1']['width']/2,
                "spline_k": 3,
                "clip_a": (-1+1e-3,1-1e-3),
            }
    if 'pm1' in coords:
        coord_parameters["pm1"] = {
                # "mixing_vals": dist.Dirichlet(jnp.ones((n_mixture, bkg_pm1_knots.shape[0]))),
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(n_mixture)) # two truncated normals
                ),
                "loc_vals": dist.Uniform(low=-20, high=20).expand([n_mixture, bkg_pm1_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=4,scale=3,low=2.78e-3).expand([n_mixture, bkg_pm1_knots.shape[0]]),
                "knots": bkg_pm1_knots,
                "x": data["phi1"],
                "low": pm1_lim[0],
                "high": pm1_lim[1],
                "spline_k": 3,
                "clip_locs": (-20,20),
                "clip_scales": (2.78e-3, None),
        }
    if 'pm2' in coords:
        coord_parameters["pm2"] = {
                # "mixing_vals": dist.Dirichlet(jnp.ones((n_mixture, bkg_pm2_knots.shape[0]))),
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(n_mixture)) # two truncated normals
                ),
                "loc_vals": dist.Uniform(low=-10, high=10).expand([n_mixture, bkg_pm2_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=4,scale=3,low=2.78e-3).expand([n_mixture, bkg_pm2_knots.shape[0]]),
                "knots": bkg_pm2_knots,
                "x": data["phi1"],
                "low": pm2_lim[0],
                "high": pm2_lim[1],
                "spline_k": 3,
                "clip_locs": (-10, 10),
                "clip_scales": (2.78e-3, None),
        }
    if 'rv' in coords: 
        coord_parameters["rv"] = {
                # "mixing_vals": dist.Dirichlet(jnp.ones((n_mixture, bkg_rv_knots.shape[0]))),
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(n_mixture)) # two truncated normals
                ),
                "loc_vals": dist.Uniform(low=-500, high=500).expand([n_mixture, bkg_rv_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=100,scale=100,low=0.1).expand([n_mixture, bkg_rv_knots.shape[0]]),
                "knots": bkg_rv_knots,
                "x": data["phi1"],
                "low": -500, #rv_lim[0],
                "high": 500, #rv_lim[1],
                "spline_k": 3,
                "clip_locs": (-500, 500),
                "clip_scales": (0.1, None),
            }


    if cmd_flow is not None and cmd_norm_stats is not None:
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
            flow=cmd_flow, norm_stats=cmd_norm_stats, x=data["phi1"]
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

def make_stream_model_component(knot_spacings, coord_bounds, data, rv_min, rv_max, stream_isochrone=None):
    '''
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
    '''

    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']
    pm1_lim  = coord_bounds['pm1']
    pm2_lim  = coord_bounds['pm2']
    rv_lim   = coord_bounds['rv']

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
    eval_interp_rv = jnp.array(_interp_dict['rv'](stream_rv_knots))
    
    coord_distributions = {
        "phi1": IndependentGMM,
        "phi2": TruncatedNormalSpline,
        "pm1": TruncatedNormalSpline,
        "pm2": TruncatedNormalSpline,
        "rv": TruncatedNormalSpline,
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
                "x": data["phi1"],
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
                "x": data["phi1"],
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
                "x": data["phi1"],
                "low": pm2_lim[0],
                "high": pm2_lim[1],
                "spline_k": 3,
                "clip_locs": pm2_lim,
                "clip_scales": (2.78e-3, 2),
            },
            "rv": {
                "loc_vals": dist.Uniform(*rv_lim).expand([stream_rv_knots.shape[0]]),
                # "loc_vals": dist.TruncatedNormal(loc=eval_interp_rv, scale=200, low=-500,high=500).expand([stream_rv_knots.shape[0]]),
                "scale_vals": dist.TruncatedNormal(loc=1,scale=5,low=0.1, high=100).expand([stream_rv_knots.shape[0]]),
                "knots": stream_rv_knots,
                "x": data["phi1"],
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
                          "rv": {"x": "phi1"}}

    if stream_isochrone is not None:
        # NOTE: see the matching NOTE in make_bkg_model_component -- fixed
        # (non-numpyro) track/distmod arrays must be bound via
        # functools.partial rather than placed directly in
        # coord_parameters[CMD_COORD], since ModelComponent.make_dists
        # interprets any *dict-valued* entry as kwargs for a nested
        # numpyro.sample() call. Only "x" (a plain array) and the fit
        # nuisance priors (dm_offset, color_offset, color_scale) go in
        # coord_parameters[CMD_COORD].
        coord_distributions[CMD_COORD] = functools.partial(
            IsochroneCMD,
            track_abs_mag=stream_isochrone["track_abs_mag"],
            track_color=stream_isochrone["track_color"],
            distmod_coeffs=stream_isochrone["distmod_coeffs"],
        )
        coord_parameters[CMD_COORD] = {
            "x": data["phi1"],
            # Small, weakly-informative priors letting the joint fit
            # calibrate for any (phi1-independent) mismatch between the
            # fixed Valluri+24 distance track / MIST isochrone and the real
            # GD-1 data, without turning the distance track into a free
            # phi1-dependent spline (per the modeling choice to keep this
            # fixed, given the modest ~1.5k-star stream sample).
            "dm_offset": dist.Normal(loc=0.0, scale=0.3),
            "color_offset": dist.Normal(loc=0.0, scale=0.05),
            "color_scale": dist.TruncatedNormal(loc=0.05, scale=0.05, low=5e-3),
        }
        conditional_data[CMD_COORD] = {"x": "phi1"}

    stream_model = ModelComponent(
        name="stream",
        coord_distributions=coord_distributions,
        coord_parameters=coord_parameters,
        conditional_data=conditional_data,
    )

    return stream_model

def make_offtrack_model_component(offtrack_dx, stream_model, coord_bounds):
    '''
    offtrack_dx: [offtrack_phi1_dx, offtrack_phi2_dx]
    '''

    offtrack_phi1_dx, offtrack_phi2_dx = offtrack_dx

    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']

    offtrack_phi12_locs = jnp.stack(
    jnp.meshgrid(
        jnp.arange(phi1_lim[0]+offtrack_phi1_dx, 
                   phi1_lim[1]-offtrack_phi1_dx/2 + 1e-3, #factor of two added by hand to make sure the high phi1 edge is properly included
                   offtrack_phi1_dx),
        jnp.arange(phi2_lim[0]+offtrack_phi2_dx, 
                   phi2_lim[1]-offtrack_phi2_dx/2 + 1e-3, 
                   offtrack_phi2_dx),
    ),
    axis=-1,
    ).reshape(-1, 2)

    ## NOTE: This model is tied to the stream model for the proper motions
    ##        (and, if present, the CMD term -- see below).
    ##        If they should not be tied, update the parameters either here or after calling it
    coord_distributions = {
        ("phi1", "phi2"): IndependentGMM,
        "pm1": stream_model.coord_distributions["pm1"],
        "pm2": stream_model.coord_distributions["pm2"],
        "rv": stream_model.coord_distributions["rv"],
    }
    coord_parameters = {
        ("phi1", "phi2"): {
            "mixing_distribution": (
                dist.Categorical,
                dist.Dirichlet(jnp.full(offtrack_phi12_locs.shape[0], 1)),
            ),
            "locs": offtrack_phi12_locs.T,
            "scales":
                # dist.HalfNormal(jnp.array([offtrack_phi1_dx, offtrack_phi2_dx])[:, None]).expand(offtrack_phi12_locs.T.shape),
                dist.TruncatedNormal(loc=jnp.array(offtrack_dx)[:, None], scale=jnp.array(offtrack_dx)[:, None],
                                     low=0.1*jnp.array(offtrack_dx)[:, None]).expand(offtrack_phi12_locs.T.shape),
            "low": jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
            "high": jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
        },
        # We don't need to define the parameters here, as they are the same as for the
        # stream model - this will be handled below when we combine the components into
        # a mixture model
        "pm1": stream_model.coord_parameters["pm1"],
        "pm2": stream_model.coord_parameters["pm2"],
        "rv": stream_model.coord_parameters["rv"],
    }
    conditional_data = {"pm1": {"x": "phi1"}, "pm2": {"x": "phi1"}, "rv": {"x": "phi1"}}

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
        # (fixed-track) IsochroneCMD term, exactly as is already done above
        # for pm1/pm2/rv.
        coord_distributions[CMD_COORD] = stream_model.coord_distributions[CMD_COORD]
        coord_parameters[CMD_COORD] = stream_model.coord_parameters[CMD_COORD]
        conditional_data[CMD_COORD] = {"x": "phi1"}

    offtrack_model = ModelComponent(
        name="offtrack",
        coord_distributions=coord_distributions,
        coord_parameters=coord_parameters,
        conditional_data=conditional_data,
    )

    return offtrack_model, offtrack_phi12_locs
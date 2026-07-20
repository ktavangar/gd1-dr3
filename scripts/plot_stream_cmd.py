"""
Diagnostic plots for the stream's fitted color-magnitude (CMD) term -- the
fixed single-stellar-population (SSP) ``IsochroneCMD`` distribution added to
``initialize_stream.make_stream_model_component`` (see that module's
``CMD_COORD = ("bp_rp", "phot_g_mean_mag")`` and the ``stream_membership``
``IsochroneCMD`` distribution), following the stream CMD modeling approach of
Starkman et al. 2025.

Two entry points:

- ``compute_membership_probs(mixture_model, data, data_err, params)``: a thin
  wrapper around ``run_full_gd1_model.make_lp_dict`` that turns its per-
  component total log-posteriors into proper (softmax-normalized) per-star
  mixture membership probabilities. ``make_lp_dict`` already handles the new
  ``IsochroneCMD`` term for free, via its generic
  ``dist_.log_prob(value=_data, x=phi1)`` fallback branch -- no changes to
  that function were needed.

- ``plot_stream_cmd_fit(...)``: a 3-panel figure --
    1. CMD scatter of the input data (optionally colored by stream
       membership probability), with the fitted isochrone ridge line
       overlaid at several representative phi1 values, so you can see the
       phi1-dependent distance-modulus gradient along the stream.
    2. The fitted apparent distance-modulus track vs. phi1 (fixed
       Valluri et al. 2024 polynomial + fit ``dm_offset``), compared to the
       raw (un-offset) literature track, to visualize the small calibration
       offset the joint fit converged to.
    3. Likely-member data (``membership_probs > 0.5``, or all data if no
       membership probabilities are given) vs. ``IsochroneCMD`` posterior
       samples drawn at the same phi1 values -- the same "model samples vs.
       data" sanity check already used for the background CMD term
       elsewhere in the pipeline notebook, applied here to the stream's
       fixed-track term.

Usage
-----
Stage 2 (standalone stream fit, no membership probabilities available yet)::

    from plot_stream_cmd import plot_stream_cmd_fit
    fig, axes = plot_stream_cmd_fit(stream_model, stream_params, stream_data)

Stage 3/4 (joint mixture fit -- color CMD points by fitted stream membership)::

    from plot_stream_cmd import compute_membership_probs, plot_stream_cmd_fit
    membership = compute_membership_probs(mm, run_data, run_data_err, full_params)
    fig, axes = plot_stream_cmd_fit(
        stream_model_mm, full_params["stream"], run_data,
        membership_probs=membership["stream"],
    )
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

import initialize_stream as init_stream
from stream_membership.distributions import (
    IndependentGMM,
    TruncatedNormalSpline,
    TruncatedNormal1DSplineMixture,
)

__all__ = ["compute_membership_probs", "plot_stream_cmd_fit"]


def _make_lp_dict(model, data, data_err, params):
    """
    Per-component, per-star total log-posterior (sum of each coordinate's
    ``log_prob`` plus ``log(mixing_prob)``).

    This is a verbatim copy of ``run_full_gd1_model.make_lp_dict`` -- copied
    rather than imported so that this plotting module doesn't pull in that
    script's module-level ``cats`` import (which isn't available in every
    environment, e.g. this sandbox, and is unrelated to what this function
    actually needs). Keep in sync with the original if it changes.

    Its final ``else`` branch (``dist_.log_prob(value=_data,
    x=jnp.asarray(data['phi1']))``) already handles the new ``IsochroneCMD``
    stream CMD term correctly for free, since ``IsochroneCMD.log_prob``
    follows the same ``(value, x=...)`` calling convention as the other
    phi1-conditioned distributions here.
    """
    lp_dict = {}

    for n_comp, component in enumerate(model.components):
        dists = component.make_dists(params[component.name])
        component_lp_stack = jnp.empty((len(dists), len(data["phi1"])))

        for i, (coord_name, dist_) in enumerate(dists.items()):
            if isinstance(coord_name, tuple):  # e.g. (phi1, phi2) offtrack, or the CMD term
                _data = jnp.stack([data[k] for k in coord_name], axis=-1)
                _errors = jnp.stack([data_err[k] for k in coord_name], axis=-1)
            else:
                _data = jnp.asarray(data[coord_name])
                _errors = jnp.asarray(data_err[coord_name])

            if isinstance(dist_, (IndependentGMM, dist.Uniform)):
                variable_lp = dist_.log_prob(value=_data)

            elif isinstance(dist_, TruncatedNormalSpline):
                spline_mean = dist_._loc_spl(data["phi1"])
                adjusted_spline_scale = jnp.sqrt(dist_._scale_spl(data["phi1"]) ** 2 + _errors**2)
                lower = (dist_.low - spline_mean) / adjusted_spline_scale
                higher = (dist_.high - spline_mean) / adjusted_spline_scale
                variable_lp = jax.scipy.stats.truncnorm.logpdf(
                    x=_data, loc=spline_mean, scale=adjusted_spline_scale, a=lower, b=higher
                )

            elif isinstance(dist_, TruncatedNormal1DSplineMixture):
                spline_mean = jnp.stack(
                    [dist_._component_distributions[i]._loc_spl(data["phi1"]) for i in range(dist_._n_components)]
                )
                spline_scale = jnp.stack(
                    [dist_._component_distributions[i]._scale_spl(data["phi1"]) for i in range(dist_._n_components)]
                )
                adjusted_spline_scale = jnp.sqrt(spline_scale**2 + _errors**2)
                lower = (dist_.low - spline_mean) / adjusted_spline_scale
                higher = (dist_.high - spline_mean) / adjusted_spline_scale
                component_lp = jax.scipy.stats.truncnorm.logpdf(
                    x=_data[None, :], loc=spline_mean, scale=adjusted_spline_scale, a=lower, b=higher
                )
                mixture_weights = params[component.name][coord_name]["mixing_distribution"]
                weighted_comp_lp = component_lp + jnp.log(mixture_weights)[:, None]
                variable_lp = jax.scipy.special.logsumexp(weighted_comp_lp, axis=0)

            else:
                # Handles IsochroneCMD (and anything else phi1-conditioned).
                variable_lp = dist_.log_prob(value=_data, x=jnp.asarray(data["phi1"]))

            component_lp_stack = component_lp_stack.at[i].set(variable_lp)

        comp_lp = jnp.sum(component_lp_stack, axis=0) + jnp.log(params["mixture-probs"][n_comp])
        lp_dict[component.name] = comp_lp

    return lp_dict


def compute_membership_probs(mixture_model, data, data_err, params):
    """
    Per-star mixture membership probabilities for every component in
    ``mixture_model``, i.e. the actual "outcome" of a Stage 3/4
    ``ComponentMixtureModel`` fit (this is a ``stream_membership`` pipeline,
    after all).

    Built on ``_make_lp_dict`` (see above), which computes, for each
    component, the total log-posterior (sum of each coordinate's
    ``log_prob`` plus ``log(mixing_prob)``) for every star. Softmax-
    normalizing those log-posteriors across components gives the standard
    finite-mixture responsibility, i.e. ``p(component | data)``.

    Parameters
    ----------
    mixture_model : ComponentMixtureModel
    data, data_err : dict
        Same ``data``/``err`` dicts the model was fit with (or any other
        aligned dataset you want membership probabilities for).
    params : dict
        The *expanded* fitted parameters for ``mixture_model``, e.g. from
        ``run_full_gd1_model.get_svi_params(model=mixture_model, ...)``.

    Returns
    -------
    dict
        Maps each component name (e.g. "background", "stream", "offtrack")
        to an array of per-star membership probabilities, shape
        ``(len(data["phi1"]),)``. The arrays sum to 1 across components for
        each star.
    """
    lp_dict = _make_lp_dict(mixture_model, data, data_err, params)
    names = mixture_model.component_names
    lps = jnp.stack([lp_dict[name] for name in names], axis=0)  # (n_components, N)
    probs = jax.nn.softmax(lps, axis=0)
    return {name: probs[i] for i, name in enumerate(names)}


def plot_stream_cmd_fit(
    stream_model,
    stream_params,
    data,
    *,
    membership_probs=None,
    phi1_vals=None,
    n_track_phi1=5,
    n_flow_samples=3000,
    flow_sample_key=None,
    cmap="viridis",
    track_cmap="plasma",
    figsize=(15, 4.2),
):
    """
    3-panel diagnostic for the stream's fitted ``IsochroneCMD`` term.

    Parameters
    ----------
    stream_model : ModelComponent
        The stream component *with* a CMD term, i.e. built with
        ``stream_isochrone=...`` (see
        ``initialize_stream.make_stream_model_component``). Works whether
        this is the standalone Stage 2 ``stream_model`` or the
        ``stream_model_mm`` used inside a Stage 3/4 ``ComponentMixtureModel``
        (the CMD term's fitted parameters are what matter, not which mixture
        it's embedded in).
    stream_params : dict
        The stream component's own (expanded) fitted parameters -- e.g.
        Stage 2's ``stream_params``, or Stage 3/4's ``full_params["stream"]``
        (see ``run_full_gd1_model.get_svi_params``).
    data : dict
        Must contain "phi1", "bp_rp", "phot_g_mean_mag", aligned with
        whatever ``membership_probs`` was computed on (e.g. ``stream_data``
        for a Stage 2 fit, or ``run_data`` for a Stage 3/4 joint fit).
    membership_probs : array, optional
        Per-star probability of stream membership, e.g.
        ``compute_membership_probs(mm, run_data, run_data_err, full_params)["stream"]``.
        If given: colors the panel-1 CMD scatter, and restricts panel 3 to
        likely members (``membership_probs > 0.5``). If omitted, all input
        data is treated as the sample to plot/compare (appropriate for a
        Stage 2 standalone fit, where ``data`` is already stream-only).
    phi1_vals : array, optional
        phi1 values (deg) at which to overplot the shifted isochrone ridge
        line. Defaults to ``n_track_phi1`` evenly-spaced percentiles (5th-
        95th) of ``data["phi1"]``.
    n_flow_samples : int
        Max number of ``IsochroneCMD`` samples to draw for panel 3 (capped
        for plotting speed; doesn't affect panels 1-2).

    Returns
    -------
    fig, axes
    """
    if init_stream.CMD_COORD not in stream_model.coord_distributions:
        msg = (
            "stream_model has no CMD term -- it must be built with "
            "stream_isochrone=... (see make_stream_model_component)."
        )
        raise ValueError(msg)

    cmd_dist = stream_model.make_dists(stream_params)[init_stream.CMD_COORD]

    bp_rp = np.asarray(data["bp_rp"])
    g_mag = np.asarray(data["phot_g_mean_mag"])
    phi1 = np.asarray(data["phi1"])
    probs_arr = None if membership_probs is None else np.asarray(membership_probs)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # --- Panel 1: CMD scatter (optionally membership-colored) + isochrone
    # ridge lines at several representative phi1 values ---
    ax = axes[0]
    if probs_arr is not None:
        sc = ax.scatter(
            bp_rp, g_mag, c=probs_arr, cmap=cmap, vmin=0, vmax=1,
            s=3, alpha=0.7, rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("p(stream)")
    else:
        ax.scatter(bp_rp, g_mag, color="0.4", s=3, alpha=0.5, rasterized=True)

    if phi1_vals is None:
        phi1_vals = np.percentile(phi1, np.linspace(5, 95, n_track_phi1))
    else:
        phi1_vals = np.asarray(phi1_vals)

    abs_mag_grid = jnp.linspace(cmd_dist._abs_mag_min, cmd_dist._abs_mag_max, 200)
    ridge_color = np.asarray(cmd_dist._color_spl(abs_mag_grid) + cmd_dist.color_offset)
    abs_mag_grid_np = np.asarray(abs_mag_grid)

    track_line_colors = plt.get_cmap(track_cmap)(np.linspace(0.15, 0.9, len(phi1_vals)))
    for phi1_val, tc in zip(phi1_vals, track_line_colors, strict=True):
        dm = float(cmd_dist._distmod(jnp.asarray(phi1_val)))
        ax.plot(
            ridge_color, abs_mag_grid_np + dm, color=tc, lw=1.5,
            label=rf"$\phi_1={phi1_val:.0f}^\circ$",
        )
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("Gaia G")
    ax.set_title("Stream CMD + fitted isochrone track")

    # --- Panel 2: fitted vs. literature (Valluri+24) distance-modulus track ---
    ax = axes[1]
    phi1_grid = np.linspace(phi1.min(), phi1.max(), 200)
    dm_raw = np.polyval(np.asarray(cmd_dist.distmod_coeffs), phi1_grid)
    dm_fit = np.asarray(cmd_dist._distmod(jnp.asarray(phi1_grid)))
    dm_offset = float(cmd_dist.dm_offset)
    ax.plot(phi1_grid, dm_raw, "k--", lw=1.5, label="Valluri+24 (fixed)")
    ax.plot(phi1_grid, dm_fit, "C1-", lw=1.5, label=f"fitted (dm_offset={dm_offset:+.3f} mag)")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel("apparent distance modulus")
    ax.legend(fontsize=8)
    ax.set_title("Distance-modulus track")

    # --- Panel 3: likely-member data vs. IsochroneCMD posterior samples,
    # drawn at the same phi1 values, as a sanity check on the fit ---
    ax = axes[2]
    member_mask = np.ones(len(phi1), dtype=bool) if probs_arr is None else probs_arr > 0.5
    n_member = int(member_mask.sum())
    ax.scatter(
        bp_rp[member_mask], g_mag[member_mask], s=4, alpha=0.5, color="0.3",
        label=f"data (N={n_member})", rasterized=True,
    )

    key = flow_sample_key if flow_sample_key is not None else jax.random.PRNGKey(0)
    phi1_for_samples = jnp.asarray(phi1[member_mask][:n_flow_samples])
    samples = jax.vmap(lambda k, x: cmd_dist.sample(k, x=x))(
        jax.random.split(key, phi1_for_samples.shape[0]), phi1_for_samples
    )
    ax.scatter(
        np.asarray(samples[:, 0]), np.asarray(samples[:, 1]), s=4, alpha=0.5,
        color="C1", label=f"IsochroneCMD samples (N={phi1_for_samples.shape[0]})",
        rasterized=True,
    )
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("Gaia G")
    ax.set_title("Likely members vs. model samples")

    return fig, axes

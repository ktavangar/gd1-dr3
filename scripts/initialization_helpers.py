import itertools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax
from stream_membership.distributions.gmm import IndependentGMM

import matplotlib.pyplot as plt
from scipy.stats import truncnorm

import initialize_stream as init_stream

def fit_phi1_scales_weights(
    phi1_data, knots, low, high,
    n_steps=3000, lr=1e-1, n_restarts=10, seed=0,
):
    """MLE fit of per-knot scales + mixing weights for phi1's IndependentGMM,
    with component locations FIXED at the knot positions (mirrors the actual
    model -- loc is not a free parameter for phi1). Uses the real IndependentGMM
    log_prob as the objective, so truncation at low/high is handled correctly.

    Initialization is data-informed: each knot's starting weight comes from the
    fraction of `phi1_data` closer to it than to any other knot, and each
    knot's starting scale comes from the local std of the data in a window
    around it (falling back to the prior's default scale if a knot has too
    few nearby points). This matters a lot once `knots` is dense enough to
    resolve real structure in the phi1 density (e.g. tightening phi1's knot
    spacing below other coordinates' spacing to fix a mismatched
    (phi1, y) 2D diagnostic): with many components all competing for
    probability mass through one joint softmax, a flat/uninformed start can
    leave some components stuck at ~zero weight in a real local optimum
    instead of tracking the data.
    """
    x = jnp.asarray(phi1_data).reshape(-1)
    knots = jnp.asarray(knots)
    K = knots.shape[0]
    knot_spacing = jnp.diff(knots).mean()
    min_scale = 0.1 * knot_spacing      # matches your model's TruncatedNormal(low=...) prior bound
    default_scale = 0.5 * knot_spacing  # matches your model's prior default

    locs = knots.reshape(1, -1)
    low_arr = jnp.array([[low]])
    high_arr = jnp.array([[high]])

    # --- Data-informed initialization ----------------------------------
    nearest = jnp.argmin(jnp.abs(x[:, None] - knots[None, :]), axis=1)
    counts = jnp.bincount(nearest, length=K).astype(float)
    init_weights0 = jnp.clip(counts / counts.sum(), 1e-3, None)
    init_logits0 = jnp.log(init_weights0)

    half_window = 0.75 * knot_spacing

    def local_scale(knot_k):
        mask = jnp.abs(x - knot_k) < half_window
        n = jnp.sum(mask)
        std = jnp.sqrt(jnp.sum(jnp.where(mask, (x - knot_k) ** 2, 0.0)) / jnp.maximum(n, 1))
        return jnp.where(n > 10, std, default_scale)

    init_scale0 = jnp.clip(jax.vmap(local_scale)(knots), min_scale + 1e-3, None)
    init_raw_scale0 = jnp.log(jnp.expm1(init_scale0 - min_scale))

    def neg_log_lik(params):
        scales = (jax.nn.softplus(params["raw_scale"]) + min_scale).reshape(1, -1)
        mix = dist.CategoricalLogits(params["logits"])
        gmm = IndependentGMM(mix, locs=locs, scales=scales, low=low_arr, high=high_arr)
        return -gmm.log_prob(x[:, None]).mean()

    def fit_once(key):
        # Jitter the data-informed init per restart (rather than starting
        # from scratch each time), so restarts explore nearby optima instead
        # of re-discovering the same structure from an uninformed start.
        k1, k2 = jax.random.split(key)
        params = {
            "raw_scale": init_raw_scale0 + 0.15 * jax.random.normal(k1, (K,)),
            "logits": init_logits0 + 0.3 * jax.random.normal(k2, (K,)),
        }
        opt = optax.adam(lr)
        opt_state = opt.init(params)

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(neg_log_lik)(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), None, length=n_steps)
        return params, losses

    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    all_params, all_losses = jax.vmap(fit_once)(keys)

    best = jnp.argmin(all_losses[:, -1])
    best_params = jax.tree.map(lambda a: a[best], all_params)
    best_loss_curve = all_losses[best]

    scales = jax.nn.softplus(best_params["raw_scale"]) + min_scale
    weights = jax.nn.softmax(best_params["logits"])
    return scales, weights, best_loss_curve

def fit_truncated_mixture_fixed_loc(x, loc, low, high, n_steps=3000, lr=1e-1, n_restarts=3, seed=0):
    """Same as fit_truncated_mixture, but with component locations fixed
    (e.g., at your phi1 knot positions) instead of learned."""
    x = jnp.asarray(x).reshape(-1)
    loc = jnp.asarray(loc).reshape(-1)
    n_mix = loc.shape[0]
    min_scale = 0.1 * jnp.diff(loc).mean()  # matches your model's own prior lower bound

    def neg_log_lik(params):
        comps = [
            dist.TruncatedNormal(
                loc=loc[k],
                scale=jax.nn.softplus(params["raw_scale"][k]) + min_scale,
                low=low, high=high,
            )
            for k in range(n_mix)
        ]
        mixture = dist.MixtureGeneral(dist.Categorical(logits=params["logits"]), comps)
        return -mixture.log_prob(x).mean()

    def fit_once(key):
        params = {"raw_scale": jnp.zeros(n_mix), "logits": jax.random.normal(key, (n_mix,)) * 0.5}
        opt = optax.adam(lr)
        opt_state = opt.init(params)

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(neg_log_lik)(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), None, length=n_steps)
        return params, losses

    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    all_params, all_losses = jax.vmap(fit_once)(keys)

    best = jnp.argmin(all_losses[:, -1])
    best_params = jax.tree.map(lambda a: a[best], all_params)

    scale = jax.nn.softplus(best_params["raw_scale"]) + min_scale
    weights = jax.nn.softmax(best_params["logits"])
    # no sort needed -- order already matches `loc`'s (knot) order, since loc is fixed
    return scale, weights

def fit_truncated_mixture(x, low, high, n_mix, n_steps=1000, lr=5e-2, n_restarts=20, seed=0):
    x = jnp.asarray(x).reshape(-1)
    min_scale = 0.01 * (x.max() - x.min())  # tie the floor to the data's own scale, not a flat 1e-3

    def neg_log_lik(params):
        comps = [
            dist.TruncatedNormal(
                loc=params["loc"][k],
                scale=jax.nn.softplus(params["raw_scale"][k]) + min_scale,
                low=low, high=high,
            )
            for k in range(n_mix)
        ]
        mixture = dist.MixtureGeneral(dist.Categorical(logits=params["logits"]), comps)
        return -mixture.log_prob(x).mean()

    def fit_once(key):
        k1, k2 = jax.random.split(key)
        params = {
            "loc": jax.random.uniform(k1, (n_mix,), minval=low, maxval=high),
            "raw_scale": jnp.zeros(n_mix),
            "logits": jax.random.normal(k2, (n_mix,)) * 0.1,
        }
        opt = optax.adam(lr)
        opt_state = opt.init(params)

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(neg_log_lik)(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), None, length=n_steps)
        return params, losses

    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    all_params, all_losses = jax.vmap(fit_once)(keys)

    final_losses = all_losses[:, -1]
    final_losses = jnp.where(jnp.isnan(final_losses), jnp.inf, final_losses)  # never let argmin pick a NaN restart
    best = jnp.argmin(final_losses)
    if jnp.isinf(final_losses[best]):
        raise RuntimeError("all restarts produced NaN -- widen min_scale or check the input window's data")

    best_params = jax.tree.map(lambda a: a[best], all_params)
    best_loss_curve = all_losses[best]

    loc = best_params["loc"]
    scale = jax.nn.softplus(best_params["raw_scale"]) + min_scale
    weights = jax.nn.softmax(best_params["logits"])
    # NOTE: intentionally NOT sorted or transformed here. Which component index
    # is "narrowest" can flip from knot to knot (if the two populations' widths
    # genuinely cross somewhere along phi1), so the ordered_scales sort+invert
    # has to happen once *across all knots together*, after this per-knot fit --
    # see `relabel_by_loc_continuity` / `invert_ordered_scales` below.
    return loc, scale, weights, best_loss_curve


def relabel_by_loc_continuity(loc_vals, scale_vals, weights):
    """
    loc_vals, scale_vals, weights: arrays of shape (n_mix, n_knots), each column
    an independent per-knot `fit_truncated_mixture` result in ARBITRARY
    (whatever the optimizer happened to converge to) component order.

    Returns re-labeled (loc_vals, scale_vals, weights) where, going knot to
    knot, components are permuted to keep `loc` as continuous as possible --
    i.e. this answers "which physical population is which" using loc. This
    assumes `ordered_scales=False` (components identified by loc, not a forced
    scale ordering) -- the returned scale_vals is the actual sigma and can be
    assigned to bkg_init_params directly, no further transform needed.
    """
    loc_vals = np.asarray(loc_vals)
    scale_vals = np.asarray(scale_vals)
    weights = np.asarray(weights)
    n_mix, n_knots = loc_vals.shape
    perms = list(itertools.permutations(range(n_mix)))

    loc_out = np.zeros_like(loc_vals)
    scale_out = np.zeros_like(scale_vals)
    w_out = np.zeros_like(weights)

    # First knot: no previous knot to match continuity against, so just sort
    # by loc (an arbitrary but stable starting choice, consistent with using
    # loc as the identifying axis everywhere else).
    order0 = np.argsort(loc_vals[:, 0])
    loc_out[:, 0] = loc_vals[order0, 0]
    scale_out[:, 0] = scale_vals[order0, 0]
    w_out[:, 0] = weights[order0, 0]

    for j in range(1, n_knots):
        best_perm, best_cost = None, np.inf
        for perm in perms:
            perm = list(perm)
            cost = np.sum((loc_vals[perm, j] - loc_out[:, j - 1]) ** 2)
            if cost < best_cost:
                best_cost, best_perm = cost, perm
        loc_out[:, j] = loc_vals[best_perm, j]
        scale_out[:, j] = scale_vals[best_perm, j]
        w_out[:, j] = weights[best_perm, j]

    return loc_out, scale_out, w_out


def invert_ordered_scales(scale_vals):
    """
    NOTE: not needed for the current setup, where `ordered_scales=False` in
    coord_parameters (see initialize_stream.py) and components are identified
    by loc via `relabel_by_loc_continuity` instead. Kept here in case
    ordered_scales=True is ever revisited.

    scale_vals: array of shape (n_mix, n_knots), the ACTUAL (physical) sigma at
    each knot, in whatever component order you've settled on (e.g. the output
    of `relabel_by_loc_continuity`). Returns the raw quadrature increments that
    TruncatedNormal1DSplineMixture's `ordered_scales=True` transform expects,
    so that it reconstructs exactly these actual scales.

    This assumes scale_vals[:, j] is ascending in the component axis for each
    knot j. Where that's not quite true -- e.g. right at a genuine crossing,
    where prioritizing loc continuity means the two components are briefly out
    of scale order -- the `maximum(..., 0)` clip makes the reconstruction
    collapse the two components to (nearly) equal width at that knot instead
    of erroring, which is a reasonable approximation given they really are
    close to equal there.
    """
    scale_vals = np.asarray(scale_vals)
    raw = np.concatenate(
        [
            scale_vals[:1],
            np.sqrt(np.maximum(scale_vals[1:] ** 2 - scale_vals[:-1] ** 2, 0.0)),
        ],
        axis=0,
    )
    return raw


def plot_knot_mixture_fits(coord_name, bkg_model, bkg_data_, N_MIXTURE, bkg_init_params,
                            ncols=5, figsize_per_ax=(3, 4), n_grid=500):
    low = float(bkg_model.coord_parameters[coord_name]["low"])
    high = float(bkg_model.coord_parameters[coord_name]["high"])
    knots = jnp.asarray(bkg_model.coord_parameters[coord_name]["knots"]).flatten()
    n_knots = len(knots)

    loc_vals = jnp.asarray(bkg_init_params[coord_name]["loc_vals"])
    # ordered_scales=False now, so scale_vals IS the actual per-component sigma
    # directly -- no inversion needed (contrast with the ordered_scales=True
    # case, where this used to need sqrt(cumsum(scale_vals**2)) here).
    scale_vals = jnp.asarray(bkg_init_params[coord_name]["scale_vals"])
    # mixing_vals was stored as log(weights) -- softmax it back to get weights per knot
    log_w = jnp.asarray(bkg_init_params[coord_name]["mixing_vals"])
    weights = jnp.exp(log_w) / jnp.exp(log_w).sum(axis=0, keepdims=True)

    ncols = min(ncols, n_knots)
    nrows = int(jnp.ceil(n_knots / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows))

    grid = jnp.linspace(low, high, n_grid)

    for i, knot in enumerate(knots):
        ax = axes.flat[i]

        subset_array_ = bkg_data_[np.abs(bkg_data_["phi1"] - float(knot)) < 2][coord_name]
        subset_array = np.asarray(subset_array_[~np.isnan(subset_array_)])

        ax.hist(subset_array, bins=40, range=(low, high), density=True, color="0.75")

        total = jnp.zeros_like(grid)
        for k in range(N_MIXTURE):
            loc, scale, w = loc_vals[k, i], scale_vals[k, i], weights[k, i]
            a, b = (low - loc) / scale, (high - loc) / scale
            comp_pdf = w * truncnorm.pdf(grid, a, b, loc=loc, scale=scale)
            ax.plot(grid, comp_pdf, lw=1.2, ls="--")
            total += comp_pdf

        ax.plot(grid, total, color="k", lw=1.5)
        ax.set_title(f"phi1 = {knot:.1f} ({len(subset_array)} pts)", fontsize=9)
        ax.set_xlim(low, high)

    for j in range(n_knots, nrows * ncols):
        axes.flat[j].axis("off")

    fig.suptitle(coord_name)
    fig.tight_layout()
    return fig, axes


def plot_phi1_marginal_fit(bkg_model, bkg_data, bkg_init_params, n_grid=1000, ax=None):
    """Quick sanity check for phi1's IndependentGMM fit (from
    fit_phi1_scales_weights): overlays the fitted p(phi1) curve -- built from
    the exact same fixed knot locations, scales, and weights that
    `bkg_model.make_dists(pars=bkg_init_params)["phi1"]` will use -- on top of
    the actual phi1 data histogram, with each fixed component location marked.

    Run this right after fitting phi1's scales/weights (before moving on to
    pm1/pm2/rv). A mismatch here silently corrupts every downstream 2D
    (phi1, y) diagnostic even when y's own per-knot fit is perfect: recall
    `evaluate_on_2d_grids` builds ln_p(phi1, y) = ln_p(phi1) + ln_p(y | phi1),
    so a bad phi1 marginal shows up as a wrong-looking 2D plot at *every*
    phi1 slice, including exactly at y's knots.
    """
    low = float(jnp.asarray(bkg_model.coord_parameters["phi1"]["low"]).flatten()[0])
    high = float(jnp.asarray(bkg_model.coord_parameters["phi1"]["high"]).flatten()[0])
    knots = jnp.asarray(bkg_model.coord_parameters["phi1"]["locs"]).flatten()

    scales = jnp.asarray(bkg_init_params["phi1"]["scales"])
    weights = jnp.asarray(bkg_init_params["phi1"]["mixing_distribution"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    phi1_data = np.asarray(bkg_data["phi1"])
    phi1_data = phi1_data[~np.isnan(phi1_data)]
    ax.hist(phi1_data, bins=100, range=(low, high), density=True, color="0.75", label="data")

    grid = jnp.linspace(low, high, n_grid)
    total = jnp.zeros_like(grid)
    for k in range(len(knots)):
        loc, scale, w = knots[k], scales[k], weights[k]
        a, b = (low - loc) / scale, (high - loc) / scale
        comp_pdf = w * truncnorm.pdf(grid, a, b, loc=loc, scale=scale)
        ax.plot(grid, comp_pdf, lw=1.0, ls="--", color="C0", alpha=0.6)
        total += comp_pdf
    ax.plot(grid, total, color="k", lw=1.5, label="fitted p(phi1)")

    for k in knots:
        ax.axvline(float(k), color="k", lw=0.5, alpha=0.3)

    ax.set_xlim(low, high)
    ax.set_xlabel("phi1 [deg]")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.set_title(f"phi1 marginal (IndependentGMM, {len(knots)} fixed-loc components) vs. data")
    return fig, ax


def check_phi1_selection_effects(bkg_model, bkg_data, coord_names=("pm1", "pm2", "radial_velocity"),
                                  n_bins=60):
    """Check whether phi1's marginal fit (on *all* background stars) is even
    the right thing to compare against a given coordinate's 2D residual plot.

    `fit_phi1_scales_weights` fits p(phi1) to the full background sample,
    including stars with no measured pm1/pm2/radial_velocity -- correct under
    a "missingness doesn't depend on phi1" (MAR) assumption, since that's the
    same population `bkg_model`'s phi1 sample site is scored against during
    SVI. But `plot_residual_projections`'s (phi1, y) panels compare the model
    against a data histogram restricted to *y*-valid stars only (it has no
    choice -- you can't histogram a missing value). If measurability of y
    isn't actually uniform in phi1, those are two different populations, and
    the panel will show a mismatch that has nothing to do with the quality of
    y's own per-knot fit. This overlays the two phi1 distributions (all stars
    vs. y-valid-only) for each coordinate so you can see directly whether/
    where that's happening, before chasing it as a fitting problem.
    """
    low = float(jnp.asarray(bkg_model.coord_parameters["phi1"]["low"]).flatten()[0])
    high = float(jnp.asarray(bkg_model.coord_parameters["phi1"]["high"]).flatten()[0])
    bins = np.linspace(low, high, n_bins + 1)

    full_phi1 = np.asarray(bkg_data["phi1"])
    full_phi1 = full_phi1[~np.isnan(full_phi1)]

    fig, axes = plt.subplots(1, len(coord_names), figsize=(5 * len(coord_names), 4), squeeze=False)
    for ax, coord_name in zip(axes.flat, coord_names):
        # bkg_data[coord_name] is already restricted to that coordinate's
        # valid-only stars (ragged per-coordinate-plate architecture), so its
        # matching phi1 subsample is looked up via `_coord_x` rather than
        # re-masking the full `bkg_data["phi1"]` array (which is a different,
        # generally longer, length).
        valid_phi1 = np.asarray(init_stream._coord_x(bkg_data, coord_name))
        valid_phi1 = valid_phi1[~np.isnan(valid_phi1)]

        ax.hist(full_phi1, bins=bins, density=True, histtype="step", lw=1.5,
                color="0.3", label=f"all stars (n={len(full_phi1)})")
        ax.hist(valid_phi1, bins=bins, density=True, histtype="step", lw=1.5,
                color="crimson", label=f"{coord_name}-valid (n={len(valid_phi1)})")
        ax.set_title(coord_name)
        ax.set_xlabel("phi1 [deg]")
        ax.legend(fontsize=8)
    fig.suptitle("phi1 density: full sample vs. per-coordinate valid subset\n"
                 "(if these differ, that alone explains a 2D residual mismatch at the knots)")
    fig.tight_layout()
    return fig, axes


def plot_shape_only_residual(coord_name, bkg_model, bkg_data, bkg_init_params, plot_grids,
                              ax=None, smooth=1.0, pcolormesh_kwargs=None):
    """A (phi1, coord_name) residual plot that sidesteps the phi1-marginal /
    selection-function issue entirely.

    `bkg_model.plot_residual_projections` compares the *full* model density
    exp(ln_p(phi1) + ln_p(coord_name | phi1)) -- scaled by the total data
    count -- against a plain 2D histogram. If coord_name's measurability
    isn't uniform in phi1 (see `check_phi1_selection_effects`), that
    comparison conflates two separate questions: "is phi1's shared marginal
    right for this subset" and "is p(coord_name | phi1)'s shape right".

    This instead column-normalizes both the model and the data so every phi1
    bin integrates to 1 in coord_name on both sides, which cancels out
    whatever phi1's marginal/selection function looks like and isolates
    exactly the thing `plot_knot_mixture_fits` already checks at the knots --
    now shown continuously across all of phi1, in the same 2D view as the
    standard residual plot. Use this in place of plot_residual_projections
    while sanity-checking pm1/pm2/radial_velocity initialization.
    """
    from scipy.ndimage import gaussian_filter

    dists = bkg_model.make_dists(pars=bkg_init_params)

    phi1_edges = np.asarray(plot_grids["phi1"])
    y_edges = np.asarray(plot_grids[coord_name])
    phi1_c = 0.5 * (phi1_edges[:-1] + phi1_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])

    # bkg_data[coord_name] is already restricted to that coordinate's
    # valid-only stars; look up its matching phi1 subsample via `_coord_x`
    # instead of re-masking the full (differently-sized) `bkg_data["phi1"]`.
    phi1_data = np.asarray(init_stream._coord_x(bkg_data, coord_name))
    y_data = np.asarray(bkg_data[coord_name])
    good = ~np.isnan(phi1_data) & ~np.isnan(y_data)
    phi1_data, y_data = phi1_data[good], y_data[good]

    H_data, _, _ = np.histogram2d(phi1_data, y_data, bins=(phi1_edges, y_edges))
    H_data = H_data.T  # (n_y, n_phi1), matching plot_residual_projections' convention
    data_shape = H_data / np.clip(H_data.sum(axis=0, keepdims=True), 1, None)

    ln_p_y = np.asarray(dists[coord_name].log_prob(
        y_c[:, None], x=jnp.broadcast_to(jnp.asarray(phi1_c)[None, :], (len(y_c), len(phi1_c)))
    ))
    model_shape = np.exp(ln_p_y)
    model_shape /= np.clip(model_shape.sum(axis=0, keepdims=True), 1e-30, None)

    resid = model_shape - data_shape
    if smooth is not None:
        resid = gaussian_filter(resid, smooth)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    pcolormesh_kwargs = dict(pcolormesh_kwargs or {})
    pcolormesh_kwargs.setdefault("cmap", "coolwarm_r")
    v = np.nanpercentile(np.abs(resid), 99)
    pcolormesh_kwargs.setdefault("vmin", -v)
    pcolormesh_kwargs.setdefault("vmax", v)

    im = ax.pcolormesh(phi1_edges, y_edges, resid, **pcolormesh_kwargs)
    for k in np.asarray(bkg_model.coord_parameters[coord_name]["knots"]).flatten():
        ax.axvline(float(k), color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("phi1")
    ax.set_ylabel(coord_name)
    ax.set_title(f"{coord_name}: shape-only residual (phi1-marginal + selection divided out)")
    plt.colorbar(im, ax=ax)
    return fig, ax


def plot_shape_only_residual_mixture(coord_name, mixture_model, data, pars, plot_grids,
                                      ax=None, smooth=1.0, pcolormesh_kwargs=None):
    """`ComponentMixtureModel`-aware version of `plot_shape_only_residual`, for
    checking a joint (e.g. background+stream, `stream_bkg_mm`) fit by eye
    without the phi1-marginal/selection-function confound described in
    `plot_shape_only_residual`'s docstring -- most relevant for pm1/pm2/
    radial_velocity, which are ragged the same way in `run_data` as they were
    in `bkg_data`, but works for any coordinate (e.g. phi2).

    Unlike the single-component version, this doesn't need to separately
    combine each component's conditional shape by hand: `evaluate_on_2d_grids`
    already returns the properly mixture-weighted *joint* ln p(phi1,
    coord_name) (each component's own phi1 density and its `mixture-probs`
    weight are both already folded in -- see `ComponentMixtureModel.
    evaluate_on_2d_grids`). Column-normalizing that joint density (so each
    phi1 bin integrates to 1 over coord_name) gives exactly the mixture's
    conditional p(coord_name | phi1) shape, which is the same quantity
    `plot_shape_only_residual` isolates for a single component -- so the
    comparison to the (also column-normalized) data histogram cancels out
    phi1's marginal/selection function here too.

    `pars` can be a packed (flat) params dict (e.g. `packed_params`, used to
    seed SVI) or an already-expanded/unpacked one (e.g. `no_off_params`,
    `full_params`) -- `evaluate_on_2d_grids` auto-detects and expands packed
    params for you.
    """
    from scipy.ndimage import gaussian_filter

    phi1_edges = np.asarray(plot_grids["phi1"])
    y_edges = np.asarray(plot_grids[coord_name])

    grids, ln_ps = mixture_model.evaluate_on_2d_grids(
        pars=pars, grids=plot_grids,
        grid_coord_names=[("phi1", coord_name)], x_coord_name="phi1",
    )
    (ln_p,) = ln_ps.values()  # joint ln p(phi1, coord_name), mixture-weighted

    model_shape = np.exp(np.asarray(ln_p))
    model_shape /= np.clip(model_shape.sum(axis=0, keepdims=True), 1e-30, None)

    # coord_name may be ragged (pm1/pm2/radial_velocity) -- use its own
    # valid-star subsample and the matching phi1 subsample, same convention
    # as plot_shape_only_residual.
    phi1_data = np.asarray(init_stream._coord_x(data, coord_name))
    y_data = np.asarray(data[coord_name])
    good = ~np.isnan(phi1_data) & ~np.isnan(y_data)
    phi1_data, y_data = phi1_data[good], y_data[good]

    H_data, _, _ = np.histogram2d(phi1_data, y_data, bins=(phi1_edges, y_edges))
    H_data = H_data.T  # (n_y, n_phi1), matching plot_residual_projections' convention
    data_shape = H_data / np.clip(H_data.sum(axis=0, keepdims=True), 1, None)

    resid = model_shape - data_shape
    if smooth is not None:
        resid = gaussian_filter(resid, smooth)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    pcolormesh_kwargs = dict(pcolormesh_kwargs or {})
    pcolormesh_kwargs.setdefault("cmap", "coolwarm_r")
    v = np.nanpercentile(np.abs(resid), 99)
    pcolormesh_kwargs.setdefault("vmin", -v)
    pcolormesh_kwargs.setdefault("vmax", v)

    im = ax.pcolormesh(phi1_edges, y_edges, resid, **pcolormesh_kwargs)
    ax.set_xlabel("phi1")
    ax.set_ylabel(coord_name)
    ax.set_title(f"{coord_name}: shape-only residual (mixture, phi1-marginal + selection divided out)")
    plt.colorbar(im, ax=ax)
    return fig, ax
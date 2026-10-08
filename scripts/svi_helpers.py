import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import optax

import gala.coordinates as gc
import pyia
import astropy.coordinates as coord
import astropy.units as u
import astropy.table as at

from pathlib import Path
import pickle
import cloudpickle

import initialize_stream as init_stream
import pretrain_bkg_cmd_flow as cmd_flow_pretrain


_CANDIDATE_ROOTS = [
    Path("/Users/Tavangar/Work/gd1-dr3"),
    Path.cwd().parent,
    Path.cwd(),
]
ROOT_DIR = next((p for p in _CANDIDATE_ROOTS if (p / "data").exists() and (p / "scripts").exists()), _CANDIDATE_ROOTS[0])
DATA_DIR = ROOT_DIR / "data"

def load_real_data(filename):
    ## ADD STREAM_DATA LATER
    '''Load (or compute) the real, CATS-processed run/background/stream tables.'''

    with open(filename, "rb") as f:
        post_cats = pickle.load(f)

    run_data_ = post_cats["run_data"]
    run_data_["pm1"] = run_data_["pm_phi1_cosphi2_unrefl"]
    run_data_["pm2"] = run_data_["pm_phi2_unrefl"]

    bkg_data_ = post_cats["bkg_data"]
    bkg_data_["pm1"] = bkg_data_["pm_phi1_cosphi2_unrefl"]
    bkg_data_["pm2"] = bkg_data_["pm_phi2_unrefl"]

    stream_data_ = post_cats["stream_data"]
    stream_data_["pm1"] = stream_data_["pm_phi1_cosphi2_unrefl"]
    stream_data_["pm2"] = stream_data_["pm_phi2_unrefl"]

    pawprint = post_cats["pawprint"]
    print(f"Loaded cached CATS-processed data from {filename}")

    return run_data_, bkg_data_, stream_data_, pawprint


def table_to_arrays(tbl):
    '''Convert an astropy Table (masked columns for pm/rv/errors) into the
    ragged, per-coordinate-plate dict-of-jnp-array format consumed by the
    new `ComponentMixtureModel.__call__` (see `stream_membership.model`) and
    `initialize_stream._coord_x`/`_phi1_key`.

    Unlike the old convention (fill missing pm1/pm2/radial_velocity with
    value=0, error=1e4 to keep every coordinate's arrays the same length as
    `phi1`), this DROPS invalid/masked rows independently per coordinate, so
    each coordinate gets its own (generally smaller, differently-sized)
    array. Each ragged coordinate's own valid-star `phi1` subsample is
    stored alongside it under the `initialize_stream._phi1_key(coord)`
    convention (e.g. "pm1_phi1") so it can be used as the conditioning
    array for that coordinate's spline/mixture terms at model-construction
    time, independently of the other coordinates' (differently-sized)
    subsamples.
    '''
    tbl = tbl.copy()

    # Drop rows with invalid/sentinel PS1 photometry (e.g. missing-photometry
    # placeholders around -999) BEFORE building the data dict, so the CMD
    # term never sees them. phi1/phi2/PS_g/PS_r are treated as always-present
    # for every remaining star (unlike pm1/pm2/radial_velocity, which are
    # ragged below).
    good_phot = (tbl["g_0"] > 0) & (tbl["g_0"] < 30) & (tbl["r_0"] > 0) & (tbl["r_0"] < 30)
    tbl = tbl[good_phot]

    data = {
        "phi1": jnp.array(tbl["phi1"], dtype="f8"),
        "phi2": jnp.array(tbl["phi2"], dtype="f8"),
    }
    err = {}

    # CMD columns, needed for the background model's flow term. Every star
    # surviving the photometry cut above has valid (PS_g, PS_r), so this
    # coordinate's own phi1 subsample is just the full `data["phi1"]`.
    # FIX TO MAKE MORE GENERAL LATER
    data["PS_g"] = jnp.array(tbl["g_0"], dtype="f8")
    data["PS_r"] = jnp.array(tbl["r_0"], dtype="f8")
    data[init_stream._phi1_key(init_stream.CMD_COORD)] = data["phi1"]

    # pm1/pm2/radial_velocity are ragged: not every star has a good
    # measurement, so each is masked and subset independently (rather than
    # sentinel-filled to a shared length), and each coordinate's own valid
    # phi1 subsample is tagged alongside it via `_phi1_key`.
    for coord, err_col in [
        ("pm1", "pm1_error"),
        ("pm2", "pm2_error"),
        ("radial_velocity", "radial_velocity_error"),
    ]:
        col = tbl[coord]
        err_col_data = tbl[err_col]
        valid = jnp.ones(len(tbl), dtype=bool)
        if hasattr(col, "mask"):
            valid = valid & jnp.array(~col.mask)
        if hasattr(err_col_data, "mask"):
            valid = valid & jnp.array(~err_col_data.mask)

        data[coord] = jnp.array(col, dtype="f8")[valid]
        err[coord] = jnp.array(err_col_data, dtype="f8")[valid]
        data[init_stream._phi1_key(coord)] = data["phi1"][valid]
        data[init_stream._phi2_key(coord)] = data["phi2"][valid]
        # Full-length boolean mask (aligned with data["phi1"]) marking which
        # stars survived into this coordinate's ragged subsample -- needed
        # to scatter this coordinate's per-star values (e.g. log-probs) back
        # into a full-length array later, e.g. in `make_lp_dict`/
        # `_make_lp_dict`. See `init_stream._valid_mask_key`/`_scatter_ragged_lp`.
        data[init_stream._valid_mask_key(coord)] = valid

    return data, err


def get_coord_bounds(run_data, pawprint):
    '''Coordinate bounds for phi1/phi2/pm1/pm2/rv. Uses the real
    initialize_stream.get_bounds_and_grids (pawprint-based pm1/pm2 bounds)
    when a pawprint is available, otherwise falls back to a percentile-based
    proper-motion box (proxy data path).

    NOTE: `run_data["pm1"]`/`"pm2"`/`"radial_velocity"` are now the ragged,
    already-valid-only arrays produced by `table_to_arrays` (invalid/missing
    rows dropped per-coordinate, not sentinel-filled with 0/1e4), so no
    `!= 0`-style sentinel filtering is needed here anymore -- doing so would
    now incorrectly drop genuine near-zero measurements.'''
    # if pawprint is not None:
    #     return init_stream.get_bounds_and_grids(run_data, pawprint)

    phi1_lim = (jnp.min(run_data["phi1"]), jnp.max(run_data["phi1"]))
    phi2_lim = (jnp.min(run_data["phi2"]), jnp.max(run_data["phi2"]))
    pm1_lim = tuple(jnp.percentile(run_data["pm1"], jnp.array([0, 100.0])))
    pm2_lim = tuple(jnp.percentile(run_data["pm2"], jnp.array([0, 100.0])))
    rv_lim = tuple(jnp.percentile(run_data["radial_velocity"], jnp.array([0, 100.0])))

    coord_bounds = {"phi1": phi1_lim, "phi2": phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim, "radial_velocity": rv_lim}
    plot_grids = {
        "phi1": jnp.linspace(*phi1_lim, 512),
        "phi2": jnp.linspace(*phi2_lim, 128),
        "pm1": jnp.linspace(*pm1_lim, 128),
        "pm2": jnp.linspace(*pm2_lim, 128),
        "radial_velocity": jnp.linspace(*rv_lim, 128),
    }
    return coord_bounds, plot_grids


def run_SVI(model, init_params, data, err=None,
            num_steps=10_000, GuideFunction=AutoNormal,
            adaptive_lr=True, init_lr=5e-3, num_particles=3,
            keys=None):
    '''Run SVI to fit `model` starting from `init_params`. Mirrors
    run_full_gd1_model.run_SVI exactly.'''
    if keys is None:
        keys = jax.random.split(jax.random.PRNGKey(42), num=2)

    if adaptive_lr:
        eighth_steps = num_steps // 8
        quarter_steps = num_steps // 4
        half_steps = num_steps // 2

        lr_schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(init_lr),
                optax.cosine_decay_schedule(init_value=init_lr, decay_steps=quarter_steps + eighth_steps, alpha=1e-1),
                optax.cosine_decay_schedule(init_value=init_lr / 10, decay_steps=quarter_steps + eighth_steps, alpha=1e-1),
            ],
            boundaries=[eighth_steps, half_steps],
        )
        optimizer = optax.chain(optax.clip(10.0), optax.adam(learning_rate=lr_schedule))
    else:
        optimizer = numpyro.optim.ClippedAdam(init_lr)

    guide = GuideFunction(model, init_loc_fn=numpyro.infer.init_to_value(values=init_params))

    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles))
    with numpyro.validation_enabled(), jax.debug_nans():
        init_state = svi.init(keys[0], init_params=init_params, data=data, err=err)
        svi_results = svi.run(rng_key=keys[1], num_steps=num_steps, init_state=init_state, data=data, err=err)

    return svi_results, guide


def get_svi_params(model, data, svi_results, guide, num_samples=1, key=None):
    if key is None:
        key = jax.random.PRNGKey(12345)
    pred_dist = Predictive(guide, params=svi_results.params, num_samples=num_samples)
    pars_ = pred_dist(key, data=data)
    # There are two different kinds of ":modeldata" sites, and only one of
    # them should be dropped here:
    #
    # - ComponentMixtureModel.__call__ names its per-coordinate latent site
    #   bare, e.g. "pm1:modeldata" or "PS_g-PS_r:modeldata" (one colon) --
    #   this is the per-star mixture-blended value shared across every
    #   component, and it doesn't nest meaningfully under any single
    #   component's `expand_numpyro_params` (it would just land as a stray
    #   top-level key), so it's dropped as bookkeeping/nuisance -- this was
    #   the original single joint "mixture:modeldata" site pre-ragged-plate,
    #   just now split per-coordinate.
    # - ModelComponent.__call__ (used for standalone fits, e.g. the
    #   background/stream models in the notebook, and for each component
    #   inside a mixture) names its site "{component}:{coord}:modeldata"
    #   (two colons) -- this one DOES nest correctly, into
    #   `expanded[coord]["modeldata"]`, and it's a genuinely useful
    #   diagnostic (the per-star denoised/latent coordinate value inferred
    #   by SVI, e.g. used by notebook cells that plot
    #   `bkg_params[coord_name]["modeldata"]`), so it must be kept.
    #
    # Filtering by suffix alone (dropping every ":modeldata"-suffixed key)
    # was too broad -- it silently stripped the component-scoped sites too,
    # which is why `bkg_params[coord_name]["modeldata"]` was raising
    # `KeyError: 'modeldata'`.
    pars = {
        k: jnp.median(v, axis=0)
        for k, v in pars_.items()
        if k != "_auto_latent"
        and not (k.endswith(":modeldata") and k.count(":") == 1)
    }
    return model.expand_numpyro_params(pars)


def save_svi_result(filename, svi_results, guide, params=None):
    # NOTE: the CMD flow term embeds a flowjax/equinox flow object, which in
    # turn holds references to JAX internals (e.g. jax.nn.softplus) that the
    # stdlib `pickle` module cannot serialize reliably ("Can't pickle
    # <function softplus ...>: it's not the same object as
    # jax._src.nn.functions.softplus"). `cloudpickle` serializes by value
    # rather than by reference and handles this correctly, so it's used here
    # instead of `pickle` for any SVI result/guide that might reference the
    # CMD flow (i.e. anything from the background model onward).
    d = {"svi_results": svi_results, "guide": guide}
    if params is not None:
        d["params"] = params
    with open(filename, "wb") as f:
        cloudpickle.dump(d, f)


def load_svi_result(filename):
    with open(filename, "rb") as f:
        return cloudpickle.load(f)


def try_load_cached(filename):
    '''Load a cached SVI result if `filename` exists and is readable;
    otherwise (missing, or corrupt e.g. from an interrupted previous run)
    return None so the caller can refit instead of crashing.'''
    if not filename.exists():
        return None
    try:
        return load_svi_result(filename)
    except Exception as exc:
        print(f"Cached file at {filename} could not be loaded ({exc!r}); refitting instead.")
        return None


def create_svi_datasets(filename=None):
    if filename is None:
        print("No filename provided for cached data; computing from scratch.")

    with open(filename, "rb") as f:
        post_cats = pickle.load(f)
    iso_obj = post_cats['iso_obj']


    run_data_, bkg_data_, stream_data_, pawprint = load_real_data(filename)
    run_data_ = run_data_[run_data_['phi1'] > -80]
    bkg_data_ = bkg_data_[bkg_data_['phi1'] > -80]
    stream_data_ = stream_data_[stream_data_['phi1'] > -80]

    print(f"run_data: {len(run_data_)} rows, bkg_data: {len(bkg_data_)} rows, stream_data: {len(stream_data_)} rows")

    run_data, run_data_err = table_to_arrays(run_data_)
    bkg_data, bkg_data_err = table_to_arrays(bkg_data_)
    stream_data, stream_data_err = table_to_arrays(stream_data_)

    coord_bounds, plot_grids = get_coord_bounds(run_data, pawprint)
    phi1_lim = coord_bounds["phi1"]

    return run_data,run_data_err,bkg_data,bkg_data_err,stream_data,stream_data_err,iso_obj,pawprint

def load_CMD_flow(cmd_flow_prefix, flow_key=42):
    flow_key = jax.random.key(flow_key)
    if (cmd_flow_prefix.with_suffix(".eqx")).exists() and Path(f"{cmd_flow_prefix}_meta.json").exists():
        cmd_flow, cmd_norm_stats = cmd_flow_pretrain.load_flow(cmd_flow_prefix, flow_key)
        print(f"Loaded pretrained CMD flow from {cmd_flow_prefix}.eqx")
    else:
        cmd_flow, cmd_norm_stats = None, None
        print(f"No pretrained CMD flow found at {cmd_flow_prefix}.eqx -- please train first")

    return cmd_flow, cmd_norm_stats

def load_stream_isochrone(iso_path):
    if iso_path.exists():
        stream_isochrone = init_stream.load_stream_isochrone(iso_path)
        print(f"Loaded stream isochrone CMD track from {iso_path}")
    else:
        stream_isochrone = None
        print(
            f"No stream isochrone found at {iso_path} -- proceeding "
            "WITHOUT a stream CMD term (stream_isochrone=None). Run "
            "`python scripts/generate_stream_isochrone.py` on a machine with "
            "network access and copy the resulting .json file here to enable it."
    )
    return stream_isochrone


def make_stream_frac_prior(f_target, order_of_magnitude=1.0):
    """
    A 2-component ``mixture-probs`` prior (drop-in replacement for
    ``dist.Dirichlet(jnp.array([1.0, 1.0]))`` in ``stream_bkg_mm =
    ComponentMixtureModel(<this>, components=[bkg_model_mm, stream_model_mm])``)
    that softly favors the stream fraction being within roughly
    ``order_of_magnitude`` orders of magnitude of ``f_target`` (1-sigma),
    instead of the flat/uninformative Dirichlet(1, 1).

    Implemented as a Normal in logit ("log-odds") space, pushed through
    numpyro's ``StickBreakingTransform`` -- the *same* transform
    ``AutoNormal`` already uses by default for any Dirichlet-supported site
    -- so this is a SOFT prior (smooth restoring gradient everywhere, no
    hard wall/cutoff) rather than a hard truncation. This matters for two
    reasons:

    1. A hard cutoff (e.g. a truncated/boxcar prior in logit space) would
       contribute exactly zero gradient inside its allowed region -- same
       problem as the flat Dirichlet(1, 1) -- so it wouldn't actually help
       pull SVI out of the near-zero, vanishing-gradient corner of
       AutoNormal's unconstrained parameterization that's been the
       recurring cause of the mixture-probs collapsing to ~0. A smooth
       Normal-shaped prior in logit space, by contrast, supplies a
       nonzero, non-vanishing gradient pulling the fit back toward
       ``f_target`` from anywhere (see the `TransformedDistribution` /
       `StickBreakingTransform` Jacobian-cancellation argument that makes
       this exactly a clean Gaussian pull in the guide's own unconstrained
       space -- verified numerically).
    2. ``len(stream_data)/len(run_data)`` is actually a fairly trustworthy
       estimate of the true stream fraction -- if anything a slight
       OVER-estimate rather than an under-estimate: ``stream_data``'s PM +
       CMD cut is strict/tight but deliberately built to not exclude any
       expected stream stars, so its main failure mode is letting in a bit
       of background contamination near the box edges, not missing real
       members. That means the direct ELBO sweep's f_stream ~ 0.024 optimum
       (~24x the naive ratio) should NOT be read as "the naive count
       undershooting the truth" -- it's a real discrepancy, more likely a
       sign of a genuine mismatch between the current background/stream
       model SHAPES in the region of PM/CMD/spatial space where they
       overlap (an open question, separate from this prior, worth
       investigating on its own). Given that, a SOFT prior anchored on the
       naive count is the right call either way: it doesn't hard-commit to
       a number that may itself be a modeling artifact, while still
       supplying the gradient needed to escape the near-zero corner.

    Contributes only O(1) to the total loss (a single log-density term, not
    summed over stars), so -- like the earlier concentration-sweep tests --
    it's far too weak to override an ~N_star-scale likelihood pull; its job
    is to nudge SVI away from the pathological near-zero corner, not to pin
    the answer to ``f_target``.

    Parameters
    ----------
    f_target
        Central stream-fraction value to anchor the prior on, e.g.
        ``len(stream_data['phi1']) / len(run_data['phi1'])``.
    order_of_magnitude
        Roughly how many orders of magnitude around ``f_target`` should be
        covered at 1-sigma (e.g. 1.0 -- the default -- means the 16th/84th
        percentiles of the induced f_stream marginal sit close to
        ``f_target / 10`` and ``f_target * 10``).

    Returns
    -------
    A numpyro ``Distribution`` with ``event_shape == (2,)`` and
    ``support == constraints.simplex``, giving ``[1 - f_stream, f_stream]``
    -- matching the ``[bkg, stream]`` ordering used elsewhere in this
    notebook/pipeline (e.g. ``packed_params["mixture-probs"] = jnp.array([1
    - f_stream, f_stream])``).
    """
    f_target = jnp.asarray(f_target)
    # NOTE: StickBreakingTransform's FIRST simplex output component is the
    # first stick-breaking fraction, sigmoid(y) itself (not its complement)
    # -- so to get index 0 (bkg) large and index 1 (stream) small, the base
    # Normal must be centered on logit(1 - f_target), not logit(f_target).
    # Getting this backwards silently swaps which component is "bkg" and
    # which is "stream" -- verified numerically before writing this.
    loc = jnp.log((1 - f_target) / f_target)
    scale = order_of_magnitude * jnp.log(10.0)

    base = dist.Normal(loc=jnp.array([loc]), scale=jnp.array([scale])).to_event(1)
    return dist.TransformedDistribution(base, dist.transforms.StickBreakingTransform())
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import optax

import gala.coordinates as gc
import pyia
import astropy.coordinates as coord
import astropy.units as u

from pathlib import Path
import pickle
import cloudpickle

import initialize_stream as init_stream


_CANDIDATE_ROOTS = [
    Path("/Users/Tavangar/Work/gd1-dr3"),
    Path.cwd().parent,
    Path.cwd(),
]
ROOT_DIR = next((p for p in _CANDIDATE_ROOTS if (p / "data").exists() and (p / "scripts").exists()), _CANDIDATE_ROOTS[0])
DATA_DIR = ROOT_DIR / "data"

def load_real_data(stream_frame = gc.GD1Koposov10()):
    ## ADD STREAM_DATA LATER
    '''Load (or compute) the real, CATS-processed run/background/stream tables.'''
    post_cats_path = DATA_DIR / "post_cats_data_flow_version.pkl"

    with open(post_cats_path, "rb") as f:
        post_cats = pickle.load(f)

    run_data_ = post_cats["run_data"]
    run_data_["pm1"] = run_data_["pm_phi1_cosphi2_unrefl"]
    run_data_["pm2"] = run_data_["pm_phi2_unrefl"]
    tbl = pyia.GaiaData(run_data_)
    pm_cov, _ = tbl.get_cov(coords=['pmra', 'pmdec'])
    c = coord.SkyCoord(ra = tbl['ra'], dec=tbl['dec'])
    gd1_coord_pm_cov = gc.transform_pm_cov(c=c, cov=pm_cov, to_frame=stream_frame)
    run_data_['pm1_error'] = jnp.sqrt(gd1_coord_pm_cov[:,0,0]) * u.mas/u.yr
    run_data_['pm2_error'] = jnp.sqrt(gd1_coord_pm_cov[:,1,1]) * u.mas/u.yr

    bkg_data_ = post_cats["bkg_data"]
    bkg_data_["pm1"] = bkg_data_["pm_phi1_cosphi2_unrefl"]
    bkg_data_["pm2"] = bkg_data_["pm_phi2_unrefl"]
    tbl = pyia.GaiaData(bkg_data_)
    pm_cov, _ = tbl.get_cov(coords=['pmra', 'pmdec'])
    c = coord.SkyCoord(ra = tbl['ra'], dec=tbl['dec'])
    gd1_coord_pm_cov = gc.transform_pm_cov(c=c, cov=pm_cov, to_frame=stream_frame)
    bkg_data_['pm1_error'] = jnp.sqrt(gd1_coord_pm_cov[:,0,0]) * u.mas/u.yr
    bkg_data_['pm2_error'] = jnp.sqrt(gd1_coord_pm_cov[:,1,1]) * u.mas/u.yr

    stream_data_ = post_cats["stream_data"]
    stream_data_["pm1"] = stream_data_["pm_phi1_cosphi2_unrefl"]
    stream_data_["pm2"] = stream_data_["pm_phi2_unrefl"]
    tbl = pyia.GaiaData(stream_data_)
    pm_cov, _ = tbl.get_cov(coords=['pmra', 'pmdec'])
    c = coord.SkyCoord(ra = tbl['ra'], dec=tbl['dec'])
    gd1_coord_pm_cov = gc.transform_pm_cov(c=c, cov=pm_cov, to_frame=stream_frame)
    stream_data_['pm1_error'] = jnp.sqrt(gd1_coord_pm_cov[:,0,0]) * u.mas/u.yr
    stream_data_['pm2_error'] = jnp.sqrt(gd1_coord_pm_cov[:,1,1]) * u.mas/u.yr

    pawprint = post_cats["pawprint"]
    print(f"Loaded cached CATS-processed data from {post_cats_path}")

    return run_data_, bkg_data_, stream_data_, pawprint


def table_to_arrays(tbl):
    '''Convert an astropy Table (masked columns for rv/rv_err) into the
    plain dict-of-jnp-array format used throughout the pipeline, filling
    missing radial velocities with rv=0, rv_err=1e4 (matches
    run_full_gd1_model.main's convention -- these rows are effectively
    uninformative but keep array shapes consistent).'''
    tbl = tbl.copy()
    for col in ["rv", "rv_err"]:
        if hasattr(tbl[col], "mask"):
            fill = 0 if col == "rv" else 1e4
            tbl[col][tbl[col].mask] = fill

    # Drop rows with invalid/sentinel PS1 photometry (e.g. missing-photometry
    # placeholders around -999) BEFORE building the data dict, so the CMD
    # term never sees them.
    good_phot = (tbl["g0"] > 0) & (tbl["g0"] < 30) & (tbl["r0"] > 0) & (tbl["r0"] < 30)
    tbl = tbl[good_phot]

    data = {k: jnp.array(tbl[k], dtype="f8") for k in ["phi1", "phi2", "pm1", "pm2", "rv"]}
    err = {
        "pm1": jnp.array(tbl["pm1_error"], dtype="f8"),
        "pm2": jnp.array(tbl["pm2_error"], dtype="f8"),
        "rv": jnp.array(tbl["rv_err"], dtype="f8"),
    }
    # CMD columns, needed for the background model's flow term.
    # FIX TO MAKE MORE GENERAL LATER
    data["PS_g"] = jnp.array(tbl["g0"], dtype="f8")
    data["PS_r"] = jnp.array(tbl["r0"], dtype="f8")

    return data, err


def get_coord_bounds(run_data, pawprint):
    '''Coordinate bounds for phi1/phi2/pm1/pm2/rv. Uses the real
    initialize_stream.get_bounds_and_grids (pawprint-based pm1/pm2 bounds)
    when a pawprint is available, otherwise falls back to a percentile-based
    proper-motion box (proxy data path).'''
    if pawprint is not None:
        return init_stream.get_bounds_and_grids(run_data, pawprint)

    phi1_lim = (jnp.min(run_data["phi1"]), jnp.max(run_data["phi1"]))
    phi2_lim = (jnp.min(run_data["phi2"]), jnp.max(run_data["phi2"]))
    pm1_lim = tuple(jnp.percentile(run_data["pm1"], jnp.array([1.0, 99.0])))
    pm2_lim = tuple(jnp.percentile(run_data["pm2"], jnp.array([1.0, 99.0])))
    rv_lim = (jnp.min(run_data["rv"]), jnp.max(run_data["rv"]))

    coord_bounds = {"phi1": phi1_lim, "phi2": phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim, "rv": rv_lim}
    plot_grids = {
        "phi1": jnp.linspace(*phi1_lim, 512),
        "phi2": jnp.linspace(*phi2_lim, 128),
        "pm1": jnp.linspace(*pm1_lim, 128),
        "pm2": jnp.linspace(*pm2_lim, 128),
        "rv": jnp.linspace(*rv_lim, 128),
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
                optax.cosine_decay_schedule(init_value=init_lr, decay_steps=quarter_steps + eighth_steps, alpha=1e-2),
                optax.cosine_decay_schedule(init_value=init_lr / 100, decay_steps=quarter_steps + eighth_steps, alpha=1e-2),
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
    pars = {k: jnp.median(v, axis=0) for k, v in pars_.items() if (k != "_auto_latent" and k != "mixture:modeldata")}
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

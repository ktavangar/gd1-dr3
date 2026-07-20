"""
Debug helper for the "FloatingPointError: invalid value (nan) encountered in
mul" that happens at step 1 of 2000 when fitting the background model with
production settings (N_MIXTURE=2, BKG_KNOT_SPACINGS=[10,10,10,10],
num_particles=3).

Why this exists
----------------
`run_SVI()` in run_full_gd1_model.py wraps `svi.init()` / `svi.run()` in
`jax.debug_nans()`, which raises `FloatingPointError` the INSTANT any NaN is
created anywhere in the traced computation. That's great for catching the
bug, but it kills the process before you can inspect what the guide's
parameters or sampled latents actually looked like at the failing step.

This script re-runs the exact same model/guide/optimizer setup, but WITHOUT
jax.debug_nans(), stepping through SVI manually so you can catch the NaN
loss, then inspect exactly which named site is responsible.

Important finding from sandbox testing: AutoNormal's guide has its own
per-datapoint ":modeldata" latent sites (one auto_loc/auto_scale pair per
row of bkg_data, for pm1/pm2/rv) in ADDITION to the "real" model parameters
(mixing_distribution, loc_vals, scale_vals, etc). A `log_density`+`jax.grad`
check that only substitutes the named model parameters (and lets
":modeldata" get freshly sampled from the model's PRIOR via a `seed`
handler) is NOT the same computation Trace_ELBO does at each SVI step --
Trace_ELBO samples EVERY latent site, including ":modeldata", from the
guide's current (learned) approximate posterior. That distinction is exactly
why earlier gradient-at-init checks (using only the named parameters) came
back completely clean on real data, while the actual SVI step can still go
NaN: the per-datapoint ":modeldata" guide samples are the untested piece.
This script's `_find_nan_site` function fixes that by sampling ALL latent
sites straight from the guide (via substitute+trace+seed), not just the
named model parameters.

How to use
----------
In run_full_gd1_model.py, temporarily replace the call that fits the
background model:

    bkg_svi_results, bkg_guide = run_SVI(bkg_model, bkg_init_params, bkg_data, bkg_data_err,
                                         num_steps=2_000, GuideFunction=guide_function, keys=keys,
                                         init_lr=1e-2, adaptive_lr=False, num_particles=3)

with:

    from debug_svi_step1_nan import debug_svi_step1_nan
    debug_svi_step1_nan(bkg_model, bkg_init_params, bkg_data, bkg_data_err,
                         num_particles=3, init_lr=1e-2)

and run that cell/script. It will print, step by step, the loss and (once it
goes NaN) exactly which guide parameter or sampled latent site has a NaN
gradient, plus the offending values themselves.
"""
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density


def debug_svi_step1_nan(model, init_params, data, err=None, num_particles=3,
                         init_lr=1e-2, max_steps=5, n_probe_draws=20, seed=42):
    """Drop-in debugging replacement for run_SVI() -- see module docstring."""

    guide = AutoNormal(model, init_loc_fn=numpyro.infer.init_to_value(values=init_params))
    optimizer = numpyro.optim.ClippedAdam(init_lr)
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles))

    key_init, key_run = jax.random.split(jax.random.PRNGKey(seed))

    # Deliberately NOT wrapped in jax.debug_nans() -- we want a normal
    # (possibly NaN) return value here, not an immediate crash.
    print("=== svi.init() ===")
    svi_state = svi.init(key_init, init_params=init_params, data=data, err=err)
    _print_guide_param_summary(svi.get_params(svi_state), label="guide params right after svi.init()")

    print(f"\n=== stepping through svi.update() one step at a time (up to {max_steps}) ===")
    prev_state = svi_state
    nan_step = None
    for step in range(max_steps):
        new_state, loss = svi.update(prev_state, data=data, err=err)
        loss = float(loss)
        is_nan = loss != loss  # NaN check without needing jnp here
        print(f"step {step}: loss={loss}" + ("  <-- NaN!" if is_nan else ""))
        if is_nan:
            nan_step = step
            break
        prev_state = new_state

    if nan_step is None:
        print(f"\nNo NaN loss in the first {max_steps} steps -- increase max_steps, "
              "or the NaN may be data/seed-dependent (try a different `seed=`).")
        return svi_state, guide

    print(f"\n>>> loss went NaN at step {nan_step}. Using the guide's params from "
          "BEFORE that step to figure out why. <<<")
    bad_params = svi.get_params(prev_state)
    _print_guide_param_summary(bad_params, label="guide params going INTO the failing step",
                                flag_bad=True)

    print(f"\n--- sampling ALL latent sites from the guide (incl. per-datapoint "
          f"':modeldata' sites) using these exact params, {n_probe_draws} times, "
          "and checking log_density/grad of the MODEL at each draw ---")
    _find_nan_site(model, guide, bad_params, data, err, n_draws=n_probe_draws)

    return svi_state, guide


def _print_guide_param_summary(params, label, flag_bad=False):
    print(f"  {label}:")
    for k, v in params.items():
        v = jnp.asarray(v)
        any_nan = bool(jnp.any(jnp.isnan(v)))
        any_inf = bool(jnp.any(jnp.isinf(v)))
        flag = "  <-- NaN/Inf!" if (flag_bad and (any_nan or any_inf)) else ""
        print(f"    {k}: shape={v.shape} nan={any_nan} inf={any_inf} "
              f"min={float(jnp.min(v)):.4g} max={float(jnp.max(v)):.4g}{flag}")


def _find_nan_site(model, guide, guide_params, data, err, n_draws=20, seed=999):
    """
    Sample from the guide `n_draws` times using its CURRENT params (loc/scale
    for every latent site -- including per-datapoint ':modeldata' sites, not
    just the named model parameters), then run log_density/grad of the model
    at each full sampled dict. This mirrors exactly what Trace_ELBO does
    internally when drawing each of its `num_particles` samples, so it
    should reproduce the NaN and tell you which named site is responsible.
    """
    seeded_model = numpyro.handlers.seed(model, jax.random.PRNGKey(0))

    def loss_fn(sampled, data, err):
        logdensity, _ = log_density(seeded_model, (), {"data": data, "err": err}, sampled)
        return logdensity

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    key = jax.random.PRNGKey(seed)
    any_nan_found = False
    for i in range(n_draws):
        key, subkey = jax.random.split(key)
        guide_trace = numpyro.handlers.trace(
            numpyro.handlers.seed(numpyro.handlers.substitute(guide, guide_params), subkey)
        ).get_trace(data=data, err=err)
        sampled = {
            name: site["value"] for name, site in guide_trace.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        val, grads = grad_fn(sampled, data, err)
        is_nan_val = bool(jnp.isnan(val))
        nan_sites = [n for n, g in grads.items() if bool(jnp.any(jnp.isnan(jnp.asarray(g))))]
        if is_nan_val or nan_sites:
            any_nan_found = True
            print(f"  draw {i}: loss={float(val)}  <-- NaN! sites with NaN grad: {nan_sites}")
            for n in nan_sites:
                v = sampled[n]
                print(f"      {n}: shape={jnp.shape(v)} "
                      f"min={float(jnp.min(v)):.4g} max={float(jnp.max(v)):.4g} "
                      f"any_nan_in_value={bool(jnp.any(jnp.isnan(jnp.asarray(v))))}")
        else:
            print(f"  draw {i}: loss={float(val):.3f}  OK")

    if not any_nan_found:
        print(f"\n  None of the {n_draws} guide draws reproduced a NaN. If svi.update() "
              "is still going NaN at this step, try: (a) more draws, (b) a couple different "
              "`seed=` values in debug_svi_step1_nan(), or (c) it may be an interaction "
              "between multiple particles' gradients being averaged together inside "
              "Trace_ELBO (num_particles=3) rather than any single particle being bad on "
              "its own -- in which case, try re-running with num_particles=1 here to see if "
              "single-particle draws alone can go NaN.")

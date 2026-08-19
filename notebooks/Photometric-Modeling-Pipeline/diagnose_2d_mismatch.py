"""
Diagnostic: what parameters is `plot_model_projections`/`evaluate_on_2d_grids`
*actually* using at plot time for the ("phi1","pm1") panel, and is the "wrong
at the knots" 2D mismatch coming from pm1's own conditional fit or from the
separately-fit phi1 marginal (`dists["phi1"]`, an IndependentGMM with its own,
generally different, knot grid)?

Paste into the notebook after cell 20 (needs bkg_model, bkg_init_params,
bkg_data_, plot_grids, N_MIXTURE, phi1_lim already defined).
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

coord_name = "pm1"

# --- 1. Rebuild the *actual* distribution objects used at plot time ---------
# This is exactly what evaluate_on_2d_grids does internally: pars=bkg_init_params
# fills in whatever's present in bkg_init_params, and falls back to the fixed
# values baked into bkg_model.coord_parameters for anything not in pars
# (e.g. phi1's "locs", pm1's "knots"/"x"/"low"/"high").
dists = bkg_model.make_dists(pars=bkg_init_params)

phi1_knots = np.asarray(bkg_model.coord_parameters[coord_name]["knots"])
phi1_marg_locs = np.asarray(bkg_model.coord_parameters["phi1"]["locs"]).flatten()
print(f"pm1 knots (n={len(phi1_knots)}):        {phi1_knots}")
print(f"phi1-marginal locs (n={len(phi1_marg_locs)}): {phi1_marg_locs}")
print("-> these are generally DIFFERENT grids if phi1_knot_spacing != "
      f"{coord_name}_knot_spacing in BKG_KNOT_SPACINGS\n")

# --- 2. Exactly replicate evaluate_on_2d_grids's grid_cs / x used for pm1 ---
grid = np.asarray(plot_grids["phi1"])
x_grid = 0.5 * (grid[:-1] + grid[1:])  # bin centers -- what's actually fed as `x`

ln_p_x = np.asarray(dists["phi1"].log_prob(x_grid))          # phi1 marginal, p(phi1)
data_mask = ~bkg_data_[coord_name].mask
phi1_data = np.asarray(bkg_data_["phi1"][data_mask])

# Data's actual phi1 density on the same bin edges used for plotting
H_phi1, edges = np.histogram(phi1_data, bins=grid, density=True)

# --- 3. Plot: model p(phi1) vs. data phi1 density, with pm1 knots marked ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.stairs(H_phi1, edges, label="data phi1 density (same subset/bins as pm1 panel)",
          color="0.6", fill=True, alpha=0.6)
ax.plot(x_grid, np.exp(ln_p_x), color="crimson", lw=1.5,
        label="model p(phi1) = dists['phi1'].log_prob(x_grid)  [IndependentGMM]")
for k in phi1_knots:
    ax.axvline(k, color="k", lw=0.6, alpha=0.5)
ax.set_xlabel("phi1 [deg]")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_title("phi1 marginal: model vs. data (black lines = pm1 knots)")
plt.show()

# --- 4. Numeric comparison AT each pm1 knot: is p(phi1) over/under-shooting?
print(f"{'knot':>8} | {'data p(phi1)':>14} | {'model p(phi1)':>14} | {'ratio (model/data)':>18}")
for k in phi1_knots:
    i = np.searchsorted(edges, k) - 1
    i = np.clip(i, 0, len(H_phi1) - 1)
    d = H_phi1[i]
    m = float(np.exp(dists["phi1"].log_prob(jnp.array([k])))[0])
    ratio = m / d if d > 0 else np.nan
    print(f"{k:8.1f} | {d:14.5f} | {m:14.5f} | {ratio:18.3f}")

# --- 5. Isolate pm1's OWN conditional shape at each knot (no phi1 factor) --
# This reproduces what plot_knot_mixture_fits/cell 26 already show you look
# good, but for *every* knot, via the same object plot_model_projections uses
# (dists["pm1"]), to confirm the spline reconstruction itself is fine at each
# knot -- isolating ln_p_y from the ln_p_x term above.
low = float(bkg_model.coord_parameters[coord_name]["low"])
high = float(bkg_model.coord_parameters[coord_name]["high"])
pm1_grid = jnp.linspace(low, high, 400)

fig, axes = plt.subplots(1, len(phi1_knots), figsize=(3.2 * len(phi1_knots), 3),
                          sharey=False, constrained_layout=True)
for ax, k in zip(np.atleast_1d(axes), phi1_knots):
    subset = bkg_data_[np.abs(bkg_data_["phi1"] - float(k)) < 2][coord_name]
    subset = np.asarray(subset[~np.isnan(subset)])
    ax.hist(subset, bins=25, range=(low, high), density=True, color="0.75")

    ln_p_y = dists[coord_name].log_prob(pm1_grid, x=jnp.full_like(pm1_grid, k))
    ax.plot(pm1_grid, np.exp(ln_p_y), color="k", lw=1.3)
    ax.set_title(f"phi1={k:.0f}", fontsize=9)
fig.suptitle(f"{coord_name} | dists['{coord_name}'] conditional shape at each knot "
             "(the exact object plot_model_projections uses)")
plt.show()

# --- 6. "Shape-only" 2D residual: divide out the phi1-marginal term --------
# Column-normalize both the data histogram and the model image so each phi1
# column integrates to 1 in pm1 -- this isolates p(pm1|phi1) and should look
# clean at the knots if the phi1-marginal term (step 3/4) is the culprit.
pm1_grid_edges = np.asarray(plot_grids[coord_name])
pm1_c = 0.5 * (pm1_grid_edges[:-1] + pm1_grid_edges[1:])

H_data2d, _, _ = np.histogram2d(phi1_data, np.asarray(bkg_data_[coord_name][data_mask]),
                                 bins=(grid, pm1_grid_edges))
H_data2d = H_data2d.T  # (n_pm1, n_phi1), matching plot_residual_projections convention

ln_p_y_grid = np.asarray(dists[coord_name].log_prob(
    pm1_c[:, None], x=jnp.broadcast_to(x_grid[None, :], (len(pm1_c), len(x_grid)))
))
model_shape = np.exp(ln_p_y_grid)
model_shape /= model_shape.sum(axis=0, keepdims=True)
data_shape = H_data2d / H_data2d.sum(axis=0, keepdims=True)
shape_resid = model_shape - data_shape

fig, ax = plt.subplots(figsize=(10, 4))
v = np.nanpercentile(np.abs(shape_resid), 99)
im = ax.pcolormesh(grid, pm1_grid_edges, shape_resid, cmap="coolwarm_r", vmin=-v, vmax=v)
for k in phi1_knots:
    ax.axvline(k, color="k", lw=0.6, alpha=0.4)
plt.colorbar(im, ax=ax, label="model_shape - data_shape (phi1-marginal divided out)")
ax.set_xlabel("phi1"); ax.set_ylabel(coord_name)
ax.set_title("Shape-only residual (should look clean at knots if step 3/4 is the culprit)")
plt.show()

"""
Pretrain a phi1-conditioned normalizing flow for the background color-magnitude
(CMD) density, for use as a ``stream_membership.distributions.FlowDensity`` term
in the background ``ModelComponent`` built by ``initialize_stream.make_bkg_model_component``.

This mirrors how a background CMD flow is used in the PyTorch/zuko-based
``stream_mapper`` models: the flow is trained once, ahead of time, by maximum
likelihood on off-stream ("background") data, then loaded as a frozen density
and *not* updated during the main SVI/MCMC fit of the rest of the mixture model.

Data selection
--------------
The real background training sample should be the same ``bkg_data_`` selection
used elsewhere in the pipeline (see ``run_full_gd1_model.run_CATS`` /
``main``): sources that pass the rough proper-motion cut and the hard
isochrone/horizontal-branch CMD cut, but fall outside the on-sky stream
footprint (``pm_mask & (iso_mask | hb_mask) & ~on_skymask``). That selection
depends on the ``cats`` package (which in turn depends on ``ugali`` and
``healpy``), which is not always importable in every environment (e.g. it
requires compiled extensions that are platform-specific).

To keep this script runnable everywhere, ``load_background_table`` builds a
*proxy* background sample directly from the raw data file, using only
quantities already present in ``GD1-region-alldata.fits``: a magnitude cut and
an on-sky cut requiring stars to sit well off the stream track in ``phi2``
(the stream is aligned to run along ``phi2 ~ 0`` by construction of the
stream-aligned coordinate frame). If you have access to the real
``bkg_data_`` table (e.g. via ``run_CATS`` or a cached ``post_cats_data.pkl``),
pass it directly to ``prepare_training_arrays`` / ``main`` instead -- see the
``--bkg-table-pickle`` CLI option.

Usage
-----
    python scripts/pretrain_bkg_cmd_flow.py \\
        --data data/GD1-region-alldata.fits \\
        --out data/bkg_cmd_flow

This writes ``data/bkg_cmd_flow.eqx`` (flow weights), ``data/bkg_cmd_flow_meta.json``
(architecture hyperparameters + normalization stats needed to reload the flow),
and ``data/bkg_cmd_flow_diagnostic.png`` (a quick visual convergence/sanity check).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import equinox as eqx
import jax

# Match the x64 setting used by the rest of the pipeline (run_full_gd1_model.py
# enables this at import time), so that a flow trained here can be
# deserialised/used downstream without a dtype (float32 vs float64) mismatch.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data

# Columns modeled by the flow, in order: (color, magnitude).
EVENT_COLUMNS = ("bp_rp", "phot_g_mean_mag")
COND_COLUMNS = ("phi1",)


def load_background_table(
    data_path: str | Path,
    phi1_lim: tuple[float, float] = (-100.0, 20.0),
    g_max: float = 20.5,
    phi2_track_buffer: float = 1.5,
    max_n: int | None = 100_000,
    seed: int = 0,
) -> Table:
    """
    Build a *proxy* background CMD training sample directly from the raw data
    file, without depending on the ``cats`` package.

    This selects sources within ``phi1_lim``, brighter than ``g_max`` in Gaia
    G (matching the magnitude cut already applied in
    ``run_full_gd1_model.run_CATS``), and offset from the stream track by more
    than ``phi2_track_buffer`` degrees in ``phi2`` (the stream-aligned frame
    puts the stream at ``phi2 ~ 0``). It does *not* reproduce the hard
    isochrone/horizontal-branch cut from ``cats.CMD.Isochrone`` -- so this
    proxy background sample is somewhat broader (more contaminated by e.g.
    main-sequence turnoff stars unrelated to GD-1's population) than the real
    ``bkg_data_`` used in the rest of the pipeline. It is intended for
    developing/testing this script; prefer the real ``bkg_data_`` table when
    available.

    Parameters
    ----------
    data_path
        Path to ``GD1-region-alldata.fits`` (or an equivalent table with the
        same column names).
    phi1_lim
        (min, max) phi1 range to include, in degrees.
    g_max
        Faint-end magnitude cut in Gaia G.
    phi2_track_buffer
        Minimum |phi2| (degrees) from the nominal stream track (phi2=0) for a
        source to be considered "background".
    max_n
        If the selected sample has more than this many rows, randomly
        subsample down to ``max_n`` (with a fixed seed for reproducibility).
        The raw data file covers ~5.5M sources and the vast majority pass
        this loose proxy cut, which is far more than needed to pretrain a
        low-dimensional (2D + 1 conditioning variable) density and would
        make training unnecessarily slow. Set to ``None`` to disable.
    seed
        RNG seed used for the subsampling above.
    """
    tbl = Table.read(data_path)

    bp_rp = tbl["phot_bp_mean_mag"] - tbl["phot_rp_mean_mag"]
    finite = (
        np.isfinite(tbl["phi1"])
        & np.isfinite(tbl["phot_g_mean_mag"])
        & np.isfinite(bp_rp)
    )
    mask = (
        finite
        & (tbl["phi1"] > phi1_lim[0])
        & (tbl["phi1"] < phi1_lim[1])
        & (tbl["phot_g_mean_mag"] < g_max)
        & (np.abs(tbl["phi2"]) > phi2_track_buffer)
    )

    bkg_tbl = tbl[mask]
    bkg_tbl["bp_rp"] = bp_rp[mask]

    if max_n is not None and len(bkg_tbl) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(bkg_tbl), size=max_n, replace=False)
        idx.sort()
        bkg_tbl = bkg_tbl[idx]

    return bkg_tbl


def load_real_background_table(
    pickle_path: str | Path,
    key: str = "bkg_data",
    max_n: int | None = 100_000,
    seed: int = 0,
) -> Table:
    """
    Load the *real*, CATS-selected background table from a cached pickle,
    instead of the raw-FITS proxy selection in ``load_background_table``.

    This is for the ``--bkg-table-pickle`` CLI option referenced in this
    module's docstring: the pickle is expected to be a ``dict`` with a
    ``bkg_data`` (or ``key``) entry holding an astropy ``Table`` -- the same
    real ``bkg_data_`` selection used elsewhere in the pipeline (see
    ``run_full_gd1_model.run_CATS`` / the notebook's ``load_real_data``),
    typically produced on a machine where the ``cats`` package (and its
    ``ugali``/``healpy`` dependencies) is actually importable, then copied
    here so this script can train on it without needing ``cats`` itself.

    Parameters
    ----------
    pickle_path
        Path to a pickle file containing a dict with a ``bkg_data`` table
        (e.g. ``real_bkg_only.pkl``, written by extracting just the
        background table out of a full ``post_cats_data.pkl``-style cache).
    key
        Dict key holding the background table, in case a different pickle
        layout is used. Defaults to ``"bkg_data"``.
    max_n
        Same as in ``load_background_table``: randomly subsample down to
        this many rows if the table is larger. Set to ``None`` to disable.
    seed
        RNG seed used for the subsampling above.
    """
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    bkg_tbl = obj[key] if isinstance(obj, dict) else obj

    if max_n is not None and len(bkg_tbl) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(bkg_tbl), size=max_n, replace=False)
        idx.sort()
        bkg_tbl = bkg_tbl[idx]

    return bkg_tbl


def prepare_training_arrays(
    bkg_tbl: Table,
) -> tuple[jax.Array, jax.Array, dict[str, dict[str, float]]]:
    """
    Extract and standardize the (color, magnitude) target array and the phi1
    conditioning array from a background table.

    Returns
    -------
    target
        Array of shape ``(N, 2)`` with standardized ``(bp_rp, phot_g_mean_mag)``.
    condition
        Array of shape ``(N, 1)`` with standardized ``phi1``.
    norm_stats
        Dictionary of ``{column_name: {"mean": ..., "std": ...}}`` used to
        standardize each column, needed to undo the standardization when
        sampling from / evaluating the trained flow on raw data later.
    """
    if "bp_rp" not in bkg_tbl.colnames:
        bkg_tbl = bkg_tbl.copy()
        bkg_tbl["bp_rp"] = bkg_tbl["phot_bp_mean_mag"] - bkg_tbl["phot_rp_mean_mag"]

    columns = {**{c: bkg_tbl[c] for c in EVENT_COLUMNS}, **{c: bkg_tbl[c] for c in COND_COLUMNS}}

    norm_stats = {}
    standardized = {}
    for name, col in columns.items():
        arr = np.asarray(col, dtype="f8")
        mean, std = float(np.mean(arr)), float(np.std(arr))
        norm_stats[name] = {"mean": mean, "std": std}
        standardized[name] = (arr - mean) / std

    target = jnp.stack([jnp.asarray(standardized[c]) for c in EVENT_COLUMNS], axis=-1)
    condition = jnp.stack([jnp.asarray(standardized[c]) for c in COND_COLUMNS], axis=-1)
    return target, condition, norm_stats


def build_flow(
    key: jax.Array,
    *,
    cond_dim: int = 1,
    event_dim: int = 2,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    rqs_knots: int = 8,
    rqs_interval: float = 5.0,
):
    """
    Construct an (untrained) conditional masked-autoregressive flow.

    Uses a rational-quadratic-spline (neural spline flow) transformer rather
    than flowjax's default affine transformer. The default affine transformer
    has an unconstrained (only lower-bounded) scale, and empirically produced
    wildly exploding samples (e.g. |value| ~ 1e6-1e7 in standardized units)
    for this data, likely from under-trained regions of conditioning space
    driving the exponentiated scale to extreme values. The spline transformer
    is identity outside of ``[-rqs_interval, rqs_interval]``, which keeps
    samples/log_probs numerically stable, and is standard practice for
    conditional density estimation with normalizing flows. Since inputs are
    standardized to zero mean/unit variance (see ``prepare_training_arrays``),
    an interval of a few sigma comfortably covers the bulk of the data.
    """
    transformer = RationalQuadraticSpline(knots=rqs_knots, interval=rqs_interval)
    return masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(event_dim)),
        transformer=transformer,
        cond_dim=cond_dim,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )


def train_flow(
    key: jax.Array,
    flow,
    target: jax.Array,
    condition: jax.Array,
    *,
    learning_rate: float = 5e-4,
    max_epochs: int = 200,
    max_patience: int = 10,
    batch_size: int = 256,
    val_prop: float = 0.1,
    show_progress: bool = True,
):
    """Fit the flow by maximum likelihood to (target, condition) pairs."""
    trained_flow, losses = fit_to_data(
        key,
        flow,
        (target, condition),
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
        val_prop=val_prop,
        show_progress=show_progress,
    )
    return trained_flow, losses


def save_flow(
    out_prefix: str | Path,
    flow,
    norm_stats: dict[str, dict[str, float]],
    flow_kwargs: dict,
) -> None:
    """Save flow weights (``.eqx``) plus a metadata sidecar (``_meta.json``)."""
    out_prefix = Path(out_prefix)
    eqx.tree_serialise_leaves(out_prefix.with_suffix(".eqx"), flow)
    meta = {
        "flow_kwargs": flow_kwargs,
        "norm_stats": norm_stats,
        "event_columns": list(EVENT_COLUMNS),
        "cond_columns": list(COND_COLUMNS),
    }
    with open(f"{out_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_flow(out_prefix: str | Path, key: jax.Array):
    """Reconstruct a trained flow + its normalization stats from disk."""
    out_prefix = Path(out_prefix)
    with open(f"{out_prefix}_meta.json") as f:
        meta = json.load(f)

    skeleton = build_flow(key, **meta["flow_kwargs"])
    flow = eqx.tree_deserialise_leaves(out_prefix.with_suffix(".eqx"), skeleton)
    return flow, meta["norm_stats"]


def make_diagnostic_plot(
    out_path: str | Path,
    flow,
    target: jax.Array,
    condition: jax.Array,
    norm_stats: dict[str, dict[str, float]],
    losses: dict,
    key: jax.Array,
) -> None:
    """Save a quick side-by-side comparison of real vs. flow-sampled CMD data,
    plus the training/validation loss curves, for a visual convergence check."""
    import matplotlib.pyplot as plt

    n_draw = target.shape[0]
    # Sample the flow at the same conditioning (phi1) values as the real data,
    # so the two panels are directly comparable.
    samples = jax.vmap(lambda k, c: flow.sample(k, condition=c))(
        jax.random.split(key, n_draw), condition
    )

    def unstd(name, arr):
        s = norm_stats[name]
        return arr * s["std"] + s["mean"]

    real_color = unstd("bp_rp", np.asarray(target[:, 0]))
    real_mag = unstd("phot_g_mean_mag", np.asarray(target[:, 1]))
    flow_color = unstd("bp_rp", np.asarray(samples[:, 0]))
    flow_mag = unstd("phot_g_mean_mag", np.asarray(samples[:, 1]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(real_color, real_mag, s=2, alpha=0.3, rasterized=True)
    axes[0].set_title(f"Real background (N={n_draw})")

    axes[1].scatter(flow_color, flow_mag, s=2, alpha=0.3, color="C1", rasterized=True)
    axes[1].set_title("Flow samples (same phi1 values)")

    for ax in axes[:2]:
        ax.set_xlabel("BP - RP")
        ax.set_ylabel("Gaia G")
        ax.invert_yaxis()

    axes[2].plot(losses["train"], label="train")
    axes[2].plot(losses["val"], label="val")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("-log likelihood")
    axes[2].set_title("Training curve")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/GD1-region-alldata.fits")
    parser.add_argument(
        "--bkg-table-pickle",
        default=None,
        help=(
            "Path to a pickle containing the real, CATS-selected bkg_data "
            "table (dict with a 'bkg_data' key, or a bare Table). If given, "
            "this is used instead of the raw-FITS proxy selection in "
            "--data/load_background_table -- see load_real_background_table."
        ),
    )
    parser.add_argument("--out", default="data/bkg_cmd_flow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-n",
        type=int,
        default=100_000,
        help="Max number of background sources to train on (random subsample).",
    )
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--nn-width", type=int, default=50)
    parser.add_argument("--nn-depth", type=int, default=1)
    parser.add_argument("--rqs-knots", type=int, default=8)
    parser.add_argument("--rqs-interval", type=float, default=5.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--max-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-diagnostic-plot", action="store_true")
    args = parser.parse_args()

    key = jax.random.key(args.seed)
    key, subkey_flow, subkey_train, subkey_diag = jax.random.split(key, 4)

    if args.bkg_table_pickle is not None:
        print(f"Loading REAL (CATS-selected) background sample from {args.bkg_table_pickle} ...")
        bkg_tbl = load_real_background_table(
            args.bkg_table_pickle, max_n=args.max_n, seed=args.seed
        )
    else:
        print(f"Loading proxy background sample from {args.data} ...")
        bkg_tbl = load_background_table(args.data, max_n=args.max_n, seed=args.seed)
    print(f"  -> {len(bkg_tbl)} background sources")

    target, condition, norm_stats = prepare_training_arrays(bkg_tbl)

    flow_kwargs = dict(
        cond_dim=1,
        event_dim=2,
        flow_layers=args.flow_layers,
        nn_width=args.nn_width,
        nn_depth=args.nn_depth,
        rqs_knots=args.rqs_knots,
        rqs_interval=args.rqs_interval,
    )
    flow = build_flow(subkey_flow, **flow_kwargs)

    print("Training flow ...")
    trained_flow, losses = train_flow(
        subkey_train,
        flow,
        target,
        condition,
        max_epochs=args.max_epochs,
        max_patience=args.max_patience,
        batch_size=args.batch_size,
    )
    print(f"  final train loss: {losses['train'][-1]:.4f}")
    print(f"  final val loss:   {losses['val'][-1]:.4f}")

    save_flow(args.out, trained_flow, norm_stats, flow_kwargs)
    print(f"Saved flow weights to {args.out}.eqx and metadata to {args.out}_meta.json")

    if not args.no_diagnostic_plot:
        diag_path = f"{args.out}_diagnostic.png"
        make_diagnostic_plot(
            diag_path, trained_flow, target, condition, norm_stats, losses, subkey_diag
        )
        print(f"Saved diagnostic plot to {diag_path}")


if __name__ == "__main__":
    main()

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




# Match the x64 setting used by the rest of the pipeline (run_full_gd1_model.py
# enables this at import time), so that a flow trained here can be
# deserialised/used downstream without a dtype (float32 vs float64) mismatch.
import jax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
import numpy as np
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from cats.pawprint.pawprint import Pawprint
from cats.proper_motions import rough_pm_poly
from astropy.table import Table
import astropy.units as u

from pathlib import Path
import equinox as eqx
import json
import argparse

COND_COLUMNS = ("phi1",) # conditioning variable(s) of the normalizing flow
CMAP = cmr.sunburst_r
RESID_CMAP = cmr.fusion


def load_background_table(
    pickle_path: str | Path,
    key: str = "bkg_data",
) -> Table:
    """
    Call when the bkg table has already been created (usually by ``run_CATS``).

    The pickle is expected to be a ``dict`` with a ``bkg_data`` (or ``key``) 
    entry holding an astropy ``Table``

    Parameters
    ----------
    pickle_path
        Path to a pickle file containing a dict with a ``bkg_data`` table
    key
        Dict key holding the background table, in case a different pickle
        layout is used. Defaults to ``"bkg_data"``.
    """
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    bkg_tbl = obj[key] if isinstance(obj, dict) else obj
    return bkg_tbl

def create_background_table(
    data: Table,
    stream_name: str = 'GD-1',
    galstreams_track_id: str = 'pricewhelan2018',
    stream_width: float = 2.0,
    phi1_min: float = None,
    phi1_max: float = None,
) -> Table:
    '''
    Call to create the background table if one does not exist.
    
    Uses galstreams and CATS code to create a polygon for the stream and 
    then selects all stars not in the polygon.

    Parameters
    ----------
    data
        astropy Table containing the data. Must at least have columns for 
        phi1 and phi2 along with whatever will be used for the normalizing flow
    stream_name
        Name of the stream to use for creating the background table. Must be 
        a valid stream name in the galstreams database. Defaults to ``"GD-1"``.
    galstreams_track_id
        ID of the galstreams track to use for creating the background table. 
        Must be a valid track ID in the galstreams database. 
        Defaults to ``"pricewhelan2018"``.
    stream_width
        Full width of the stream to use for creating the background table. 
        Defaults to 2.0 degrees. NOTE: Some streams will be wider than this default!
    phi1_min, phi1_max
        Limit on the phi1 values to make the polygon. If ``None``, the limits 
        will be set to the minimum and maximum of the phi1 column in the data.
    '''

    ## TODO: FIX THESE DIAGNOSTICS

    # Check that inputs are appropriate
    # if not isinstance(data, Table):
    #     raise ValueError("Input 'data' must be an astropy Table.")
    # mws = galstreams.MWStreams(verbose=False, implement_Off=False, print_topcat_friendly_files=False)
    # if stream_name not in mws.all_unique_stream_names():
    #     print('Valid stream names are: ', mws.all_unique_stream_names())
    #     raise ValueError(f"""Stream '{stream_name}' not found in galstreams database. \n
    #                      Valid stream names are: {mws.all_unique_stream_names()}""")

    if phi1_min is None:
        phi1_min = np.min(data['phi1'])
    if phi1_max is None:
        phi1_max = np.max(data['phi1'])

    phi1_lim = (phi1_min, phi1_max)

    on_stream_outline = Pawprint.pawprint_from_galstreams(
            stream_name = stream_name,
            pawprint_ID = galstreams_track_id,
            width=stream_width * u.deg,
            phi1_lim=phi1_lim)
    
    on_stream_patch = mpl.patches.Polygon(
        on_stream_outline.skyprint["stream"].vertices[::50],
    )
    
    on_points = np.vstack((data['phi1'], data['phi2'])).T
    on_mask = on_stream_patch.get_path().contains_points(on_points)
    off_mask = ~on_mask

    _, pm_mask = rough_pm_poly(on_stream_outline, data, buffer=2)
    bkg_tbl = data[pm_mask & off_mask]
    return bkg_tbl


def clean_bkg_data(
    bkg_tbl: Table,
    mag_param_names: tuple[str, str], 
    maxmag: float = 22, 
    max_n: int = None, 
    seed: int = 42) -> Table:
    """
    Clean the background data by removing any rows with missing values.

    Parameters
    ----------
    bkg_tbl : Table
        Background table to clean
    mag_param_names : tuple[str, str]
        Names of the magnitude parameters to use for the normalizing flow.
    maxmag : float
        Maximum magnitude value to consider.
    max_n : int
        Maximum number of background points to use for training. If None, all points are used.
    seed : int
        Random seed for sampling the background points.

    Returns
    -------
    Table
        Cleaned background table.
    """
    
    mag1 = mag_param_names[0]
    mag2 = mag_param_names[1]

    # check that the magnitude parameters exist
    if mag1 not in bkg_tbl.colnames or mag2 not in bkg_tbl.colnames:
        raise ValueError("Specified magnitude parameters not found in background table.")
    
    # quality cut to keep only the rows with reasonable magnitude values
    clean_bkg_tbl = bkg_tbl[(bkg_tbl[mag1] > 0) & (bkg_tbl[mag2] > 0) & \
                       (bkg_tbl[mag1] < maxmag) & (bkg_tbl[mag2] < maxmag)]

    # sample from the bkg_data if there are too many points for the model to handle
    if max_n is not None and len(clean_bkg_tbl) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(clean_bkg_tbl), size=max_n, replace=False)
        idx.sort()
        clean_bkg_tbl = clean_bkg_tbl[idx]

    return clean_bkg_tbl

def prepare_training_arrays(
    bkg_tbl: Table,
    mag_param_names: tuple[str, str],
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

    columns = {**{c: bkg_tbl[c] for c in mag_param_names}, **{c: bkg_tbl[c] for c in COND_COLUMNS}}

    norm_stats = {}
    standardized = {}
    for name, col in columns.items():
        arr = np.asarray(col, dtype="f8")
        mean, std = float(np.mean(arr)), float(np.std(arr))
        norm_stats[name] = {"mean": mean, "std": std}
        standardized[name] = (arr - mean) / std

    target = jnp.stack([jnp.asarray(standardized[c]) for c in mag_param_names], axis=-1)
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
    mag_param_names: tuple[str, str],
) -> None:
    """Save flow weights (``.eqx``) plus a metadata sidecar (``_meta.json``)."""
    out_prefix = Path(out_prefix)
    eqx.tree_serialise_leaves(out_prefix.with_suffix(".eqx"), flow)
    meta = {
        "flow_kwargs": flow_kwargs,
        "norm_stats": norm_stats,
        "event_columns": list(mag_param_names),
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
    mag_param_names: tuple[str, str],
    target: jax.Array,
    condition: jax.Array,
    norm_stats: dict[str, dict[str, float]],
    losses: dict,
    key: jax.Array,
) -> None:
    """Save a quick side-by-side comparison of real vs. flow-sampled CMD data,
    plus the training/validation loss curves, for a visual convergence check."""

    n_draw = target.shape[0]
    # Sample the flow at the same conditioning (phi1) values as the real data,
    # so the two panels are directly comparable.
    samples = jax.vmap(lambda k, c: flow.sample(k, condition=c))(
        jax.random.split(key, n_draw), condition
    )

    def unstd(name, arr):
        s = norm_stats[name]
        return arr * s["std"] + s["mean"]

    real_mag1 = unstd(mag_param_names[0], np.asarray(target[:, 0]))
    real_mag2 = unstd(mag_param_names[1], np.asarray(target[:, 1]))
    flow_mag1 = unstd(mag_param_names[0], np.asarray(samples[:, 0]))
    flow_mag2 = unstd(mag_param_names[1], np.asarray(samples[:, 1]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0][0].hist2d(real_mag1, real_mag2, bins=200, range=[[10,22], [10,22]], norm=mpl.colors.LogNorm(), cmap=CMAP, rasterized=True)
    axes[0][0].set_title(f"Real background (N={n_draw})")

    axes[0][1].hist2d(flow_mag1, flow_mag2, bins=200, range=[[10,22], [10,22]], norm=mpl.colors.LogNorm(), cmap=CMAP, rasterized=True)
    axes[0][1].set_title("Flow samples (same phi1 values)")

    for ax in axes[0,:2]:
        ax.set_xlabel(f"{mag_param_names[0]}")
        ax.set_ylabel(f"{mag_param_names[1]}")
    
    axes[1][0].hist2d(real_mag1-real_mag2, real_mag1, bins=200, range=[[-2,3], [10, 22]], norm=mpl.colors.LogNorm(), cmap=CMAP, rasterized=True)
    axes[1][1].hist2d(flow_mag1 - flow_mag2, flow_mag1, bins=200, range=[[-2,3], [10, 22]], norm=mpl.colors.LogNorm(), cmap=CMAP, rasterized=True)
    H1, xe, ye, = np.histogram2d(real_mag1-real_mag2, real_mag1, bins=100, range=[[-0.5,1.5], [12, 22]])
    H2, _, _    = np.histogram2d(flow_mag1 - flow_mag2, flow_mag1, bins=100, range=[[-0.5,1.5], [12, 22]])
    resid = H1-H2
    vmax = np.nanmax(np.abs(resid))
    axes[1][0].pcolormesh(xe, ye, H1.T, norm=mpl.colors.LogNorm(), cmap=CMAP, rasterized=True)
    axes[1][1].pcolormesh(xe, ye, H2.T, norm=mpl.colors.LogNorm(), cmap=CMAP, rasterized=True)
    axes[1][2].pcolormesh(xe, ye, resid.T, cmap=RESID_CMAP, vmin=-vmax, vmax=vmax, rasterized=True)

    for ax in axes[1,:]:
        ax.set_xlabel(f"{mag_param_names[0]}  - {mag_param_names[1]}")
        ax.set_ylabel(f"{mag_param_names[1]}")
        ax.invert_yaxis()


    axes[0,2].plot(losses["train"], label="train")
    axes[0,2].plot(losses["val"], label="val")
    axes[0,2].set_xlabel("epoch")
    axes[0,2].set_ylabel("-log likelihood")
    axes[0,2].set_title("Training curve")
    axes[0,2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="/Users/Tavangar/Work/gd1-dr3/data/GD1-region-alldata.fits")
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
    parser.add_argument("--stream-name", default="GD-1")
    parser.add_argument("--galstreams-track-id", type=str, default='pricewhelan2018')
    parser.add_argument("--stream-width", type=float, default=2.0)
    parser.add_argument("--phi1-min", type=float, default=-100)
    parser.add_argument("--phi1-max", type=float, default=20)
    parser.add_argument("--mag-param_name1", type=str, default='g0')
    parser.add_argument("--mag-param_name2", type=str, default='r0')

    parser.add_argument("--maxmag", type=float, default=22.0)
    parser.add_argument("--out", default="data/bkg_cmd_flow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n", type=int, default=None,
        help="Max number of background sources to train on (random subsample).",
    )
    parser.add_argument("--flow-layers", type=int, default=32)
    parser.add_argument("--nn-width", type=int, default=100)
    parser.add_argument("--nn-depth", type=int, default=2)
    parser.add_argument("--rqs-knots", type=int, default=32)
    parser.add_argument("--rqs-interval", type=float, default=5.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--max-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-diagnostic-plot", action="store_true")
    args = parser.parse_args()

    MAG_PARAM_NAMES = [args.mag_param_name1, args.mag_param_name2]

    key = jax.random.key(args.seed)
    key, subkey_flow, subkey_train, subkey_diag = jax.random.split(key, 4)

    # if args.bkg_table_pickle does not exist, load background table
    if args.bkg_table_pickle is not None and Path(args.bkg_table_pickle).exists():
        print(f"Pickle file {args.bkg_table_pickle} found. Loading background table...")
        bkg_tbl = load_background_table(args.bkg_table_pickle, key='bkg_data')
    else:
        print(f"Creating background table...")
        data_tbl = Table.read(args.data_file)
        bkg_tbl = create_background_table(data=data_tbl, 
                                          stream_name=args.stream_name, 
                                          galstreams_track_id=args.galstreams_track_id, 
                                          stream_width=args.stream_width, 
                                          phi1_min=args.phi1_min,
                                          phi1_max=args.phi1_max)
    
    clean_bkg_tbl = clean_bkg_data(bkg_tbl,
                                   MAG_PARAM_NAMES,
                                   maxmag=args.maxmag, 
                                   max_n=args.max_n, 
                                   seed=args.seed)

    print(f"  -> {len(clean_bkg_tbl)} background sources")

    target, condition, norm_stats = prepare_training_arrays(clean_bkg_tbl, mag_param_names=MAG_PARAM_NAMES)

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

    save_flow(args.out, mag_param_names=MAG_PARAM_NAMES, flow=trained_flow, norm_stats=norm_stats, flow_kwargs=flow_kwargs)
    print(f"Saved flow weights to {args.out}.eqx and metadata to {args.out}_meta.json")

    if not args.no_diagnostic_plot:
        diag_path = f"{args.out}_diagnostic.png"
        make_diagnostic_plot(
            diag_path, mag_param_names=MAG_PARAM_NAMES, flow=trained_flow, 
            target=target, condition=condition, norm_stats=norm_stats, 
            losses=losses, key=subkey_diag
        )
        print(f"Saved diagnostic plot to {diag_path}")


if __name__ == "__main__":
    main()

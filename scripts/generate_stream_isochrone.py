"""
Generate a single-stellar-population (SSP) isochrone track for the
GD-1 stream's color-magnitude diagram, following the SSP-isochrone approach
of Starkman et al. 2025 ("Stream Members Only") for modeling the stream's (as
opposed to the background's) CMD structure.

Unlike the background CMD term (``pretrain_bkg_cmd_flow.py``), which fits a
flexible, phi1-conditioned normalizing flow to off-stream sources, the
stream's own CMD is modeled here as a *fixed*, physically-motivated isochrone
track (old age, metal-poor, matching the population parameters already used
for GD-1's existing PS1-band isochrone pre-selection cut in
``cats.inputs.stream_inputs["GD-1"]``: age=12 Gyr, [Fe/H]=-2.5). See
``IsochroneCMD`` in ``stream_membership.distributions.isochrone_cmd`` for how
this track is turned into a probability density, conditioned on phi1 through
the (fixed) Valluri et al. 2024 GD-1 apparent-distance-modulus polynomial.

Data/network dependency
------------------------
This script uses the ``isochrones`` package
(https://isochrones.readthedocs.io/), which wraps the MIST
bolometric-correction/model grids. **The first time it's run (per machine),
``isochrones`` downloads several hundred MB of MIST grid data from MIST's
servers**, caching it under ``~/.isochrones/mist/``. That download requires
outbound network access to a host that is *not* on this sandbox's network
allowlist (the same restriction that blocks fetching arXiv/ADS pages
elsewhere in this project) -- confirmed by testing here: the download
silently produces a truncated placeholder file, which then fails when
``isochrones`` tries to ``tarfile.open()`` it. Concretely, in this sandbox:

    >>> from isochrones.mist import MIST_Isochrone
    >>> MIST_Isochrone(bands=["G", "BP", "RP"]).isochrone(10.08, -2.5)
    ...
    tarfile.ReadError: file could not be opened successfully

**Run this script on a machine with normal internet access** (e.g. your own
laptop: ``pip install isochrones numba``), then copy the resulting
``data/stream_isochrone_ssp.json`` file back into this repo's
``gd1-dr3/data/`` directory. ``IsochroneCMD`` /
``initialize_stream.make_stream_model_component`` only need that small
cached JSON file, not the ``isochrones`` package itself or network access --
so nothing else in the pipeline needs to run outside this sandbox.

Usage
-----
    python scripts/generate_stream_isochrone.py --out data/stream_isochrone_ssp

This writes ``data/stream_isochrone_ssp.json`` (the restricted, monotonic
main-sequence-to-lower-RGB absolute-magnitude and color arrays used by
``IsochroneCMD``) and ``data/stream_isochrone_ssp_diagnostic.png`` (a
quick plot of the generated track).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Same population parameters already used for GD-1's PS1-band isochrone
# pre-selection cut (cats.inputs.stream_inputs["GD-1"]), from Valluri et al.
# 2024. Reusing these keeps the new CMD track consistent with the
# same underlying stellar population already assumed elsewhere in the
# pipeline.
from cats.inputs import stream_inputs as inputs
stream_name = "GD-1"
AGE_GYR = inputs[stream_name]['age']
FEH = inputs[stream_name]['feh']
PHOT_SURVEY = inputs[stream_name]['phot_survey'][:-1]
# `cats`/ugali survey names differ from the photometric-system prefixes that
# `isochrones`' MIST bolometric-correction grids recognize (e.g. MIST calls
# Pan-STARRS "PanSTARRS", with bands stored as PS_g, PS_r, ...).
_MIST_PHOT_SYSTEM = {"PS1": "PanSTARRS", "ps1": "PanSTARRS"}
_MIST_SURVEY = _MIST_PHOT_SYSTEM.get(PHOT_SURVEY, PHOT_SURVEY)
MAG1 = f'{_MIST_SURVEY}_{inputs[stream_name]["band1"]}'
MAG2 = f'{_MIST_SURVEY}_{inputs[stream_name]["band2"]}'

# Valluri et al. 2024 GD-1 apparent distance-modulus track, as a function of
# phi1 (degrees) -- see cats.CMD.Isochrone.__init__:
#   distmod_spl = np.poly1d([1/64**2, 100/64**2, (50/64)**2 + 18.82 - 4.45])
# Stored here as plain poly1d coefficients (highest power first) so this
# script and IsochroneCMD don't need to import `cats` (which depends on
# `ugali`/`healpy` and isn't importable in every environment).
DISTMOD_COEFFS = (1 / 64**2, 100 / 64**2, (50 / 64) ** 2 + 18.82 - 4.45)


def generate_ssp_isochrone(
    age_gyr: float = AGE_GYR, feh: float = FEH,
    mag_param_name1: str = MAG1,
    mag_param_name2: str = MAG2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single-age, single-metallicity MIST isochrone.

    Requires network access (the first time it's run on a given machine) to
    download/cache the MIST grid data used by the ``isochrones`` package --
    see this module's docstring.

    Returns
    -------
    initial_mass, abs_mag1, abs_mag2
        1D arrays, sorted by increasing initial mass, of initial stellar mass
        (solar masses), and two magnitude bands
    """
    from isochrones.mist import MIST_Isochrone

    mist = MIST_Isochrone(bands=[mag_param_name1, mag_param_name2])
    log_age = np.log10(age_gyr * 1e9)

    # No `distance`/`AV` kwargs -> absolute magnitudes (distance=10pc, no
    # extinction). The phi1-dependent apparent shift is applied later, in
    # IsochroneCMD, via the Valluri+24 distance-modulus track.
    df = mist.isochrone(log_age, feh)

    # Column-naming has varied across `isochrones` versions
    # ("mass"/"initial_mass", "G"/"G_mag"); check both.
    mass_col = "initial_mass" if "initial_mass" in df.columns else "mass"
    order = np.argsort(df[mass_col].to_numpy())

    def _col(band: str) -> np.ndarray:
        name = band if band in df.columns else f"{band}_mag"
        return df[name].to_numpy()[order]

    initial_mass = df[mass_col].to_numpy()[order]
    abs_mag1 = _col(mag_param_name1)
    abs_mag2 = _col(mag_param_name2)

    return initial_mass, abs_mag1, abs_mag2


def restrict_to_monotonic_branch(
    initial_mass: np.ndarray, abs_mag1: np.ndarray, abs_mag2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict an isochrone track (ordered by increasing initial mass, i.e.
    increasingly luminous / decreasing absolute mag along the main sequence) to
    its leading monotonic-in-absolute-mag segment: from the lowest-mass point
    included, through the main sequence, turnoff, and subgiant branch,
    stopping at the base of the red giant branch where ``abs_g`` first stops
    monotonically decreasing. (The "turnover" is where the absolute magnitude 
    stops decreasing and starts increasing again.)

    This restriction is necessary because the RGB bends back in
    color-magnitude space (multiple masses share the same absolute
    magnitude), so a single-valued spline ``color = f(abs_g)``, as used by
    ``IsochroneCMD``, is only well-defined on this leading branch. GD-1
    member photometry is dominated by the main-sequence/turnoff region
    anyway (``stream_inputs["GD-1"]["turnoff"] = 17.8`` mag, well below the
    RGB tip), so this restriction should not discard the astrophysically
    relevant part of the track for membership modeling.
    """
    d = np.diff(abs_mag1)
    turnover = np.flatnonzero(d >= 0)
    stop = (turnover[0] + 1) if turnover.size else len(abs_mag1)
    return initial_mass[:stop], abs_mag1[:stop], abs_mag2[:stop]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        default="/Users/Tavangar/Work/gd1-dr3/data/stream_isochrone_ssp",
        help="Output path prefix (writes <out>.json and <out>_diagnostic.png)",
    )
    parser.add_argument("--age-gyr", type=float, default=AGE_GYR)
    parser.add_argument("--feh", type=float, default=FEH)
    parser.add_argument("--mag-param_name1", type=str, default=MAG1)
    parser.add_argument("--mag-param_name2", type=str, default=MAG2)
    args = parser.parse_args()
    mag_name1 = args.mag_param_name1
    mag_name2 = args.mag_param_name2

    print(f"Generating MIST isochrone: age={args.age_gyr} Gyr, [Fe/H]={args.feh} ...")
    initial_mass, abs_mag1, abs_mag2 = generate_ssp_isochrone(
        args.age_gyr, args.feh,
        mag_param_name1=mag_name1, 
        mag_param_name2=mag_name2)
    print(f"  -> {len(initial_mass)} raw track points")

    mass_r, abs_mag1_r, abs_mag2_r = restrict_to_monotonic_branch(initial_mass, abs_mag1, abs_mag2)
    print(
        f"  -> restricted to {len(mass_r)} monotonic (MS/turnoff/subgiant) points"
        f"  -> abs {mag_name1} in [{abs_mag1_r.min():.2f}, {abs_mag1_r.max():.2f}]"
    )

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "age_gyr": args.age_gyr,
        "feh": args.feh,
        "bands": [mag_name1, mag_name2],
        "distmod_coeffs": list(DISTMOD_COEFFS),
        "distmod_source": "Valluri et al. 2024 (see cats.CMD.Isochrone)",
        "initial_mass": mass_r.tolist(),
        "abs_mag1": abs_mag1_r.tolist(),
        "abs_mag2": abs_mag2_r.tolist(),
    }
    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {json_path}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(abs_mag1-abs_mag2, abs_mag1, color="0.7", lw=1, label="full track")
        ax.plot(abs_mag1_r-abs_mag2_r, abs_mag1_r, color="tab:red", lw=2, label="restricted (used)")
        ax.invert_yaxis()
        ax.set_xlabel(f"{mag_name1} - {mag_name2}")
        ax.set_ylabel(f"absolute {mag_name1}")
        ax.set_title(f"MIST isochrone: age={args.age_gyr} Gyr, [Fe/H]={args.feh}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        png_path = out_prefix.parent / f"{out_prefix.name}_diagnostic.png"
        fig.savefig(png_path, dpi=150)
        print(f"Wrote {png_path}")
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"(skipped diagnostic plot: {e})")


if __name__ == "__main__":
    main()


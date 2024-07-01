import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from stream_membership import StreamModel, StreamMixtureModel
from stream_membership.utils import get_grid
from stream_membership.variables import (
    GridGMMVariable,
    Normal1DSplineMixtureVariable,
    Normal1DSplineVariable,
    UniformVariable,
    Normal1DVariable
)

#phi1_lim = (-100, 20)

class Base:
    phi1_lim = (-20, 20)
    coord_bounds = {"phi1": phi1_lim, "phi2": (-7,7), "pm1": (-20,20), "pm2": (-20,20)}

class BackgroundModel(Base, StreamModel):
    name = "background"

    ln_N_dist = dist.Uniform(-10, 15)

    phi1_locs = get_grid(*Base.phi1_lim, 10.0, pad_num=1).reshape(-1, 1)  # every 10º
    pm1_knots = get_grid(*Base.coord_bounds["phi1"], 10.0) # changed from 10 to 15 because of small scale features in pm2
    pm2_knots = get_grid(*Base.coord_bounds["phi1"], 15.0)

    variables = {"phi1": None,
                 "phi2": None,
                 "pm1": None,
                 "pm2": None,
                }
    data_required = {
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }


class StreamDensModel(Base, StreamModel):
    name = "stream"

    ln_N_dist = dist.Uniform(5, 15)

    phi1_dens_step = 4.0  # knots every 4º
    phi1_locs = get_grid(*Base.phi1_lim, phi1_dens_step, pad_num=1).reshape(-1, 1)

    phi2_knots = get_grid(*Base.phi1_lim, 10.0)  # knots every 10º

    pm1_knots = get_grid(*Base.phi1_lim, 10.0)  # knots every 10º
    pm2_knots = get_grid(*Base.phi1_lim, 10.0)  # knots every 10º

    variables = {"phi1": None,
                 "phi2": None,
                 "pm1": None,
                 "pm2": None,
                }
    data_required = {
        "phi2": {"x": "phi1", "y": "phi2"},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    def extra_ln_prior(self, params):
        lp = 0.0

        std_map = {"mean": 0.5, "ln_std": 0.25}
        for var_name, var in self.variables.items():
            if hasattr(var, "splines"):
                for par_name, spl_y in params[var_name].items():
                    if par_name in std_map:
                        lp += (
                            dist.Normal(0, std_map[par_name])
                            .log_prob(spl_y[1:] - spl_y[:-1])
                            .sum()
                        )

        return lp


class OffTrackModel(Base, StreamModel):
    name = "offtrack"

    ln_N_dist = dist.Uniform(-5, 10)

    dens_phi1_lim = (-100, 20)
    dens_phi2_lim = (-8, 3.5)

    dens_steps = np.array([4.0, 0.4]) # should find some optimal spacing here
    # spar_steps = 5*dens_steps
            
    dens_locs = np.stack(
        np.meshgrid(
            np.arange(dens_phi1_lim[0], dens_phi1_lim[1] + 1e-3, dens_steps[0]),
            np.arange(dens_phi2_lim[0], dens_phi2_lim[1] + 1e-3, dens_steps[1]),
        )
    ).T.reshape(-1, 2)

    # spar_locs = np.stack(
    #     np.meshgrid(
    #         get_grid(*Base.coord_bounds["phi1"], spar_steps[0], pad_num=1),
    #         get_grid(*Base.coord_bounds["phi2"], spar_steps[1], pad_num=1),
    #     )
    # ).T.reshape(-1, 2)
    # _mask = (
    #     (spar_locs[:, 0] >= dens_phi1_lim[0])
    #     & (spar_locs[:, 0] <= dens_phi1_lim[1])
    #     & (spar_locs[:, 1] >= dens_phi2_lim[0])
    #     & (spar_locs[:, 1] <= dens_phi2_lim[1])
    # )
    # spar_locs = spar_locs[~_mask]

    # phi12_locs = np.concatenate((dens_locs, spar_locs))
    # phi12_scales = np.concatenate(
    #     (np.full_like(dens_locs, dens_steps[0]), np.full_like(spar_locs, spar_steps[0]))
    # )
    # phi12_scales[: dens_locs.shape[0], 1] = dens_steps[1]
    # phi12_scales[dens_locs.shape[0] :, 1] = spar_steps[1]

    phi12_locs = dens_locs
    phi12_scales = np.full_like(dens_locs, dens_steps[0])
    phi12_scales[:, 1] = dens_steps[1]

    variables = {("phi1", "phi2"): None,
                 "pm1": None,
                 "pm2": None}

    data_required = {
        ("phi1", "phi2"): {"y": ("phi1", "phi2")},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

class MixtureModel(Base, StreamMixtureModel):
    components = [BackgroundModel, StreamDensModel, OffTrackModel]
    
    
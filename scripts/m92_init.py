import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from stream_membership.utils import get_grid
from stream_membership.variables import (
    GridGMMVariable,
    Normal1DSplineMixtureVariable,
    Normal1DSplineVariable,
    UniformVariable,
    Normal1DVariable
)

from gd1_helpers.membership.gd1_model import StreamDensModel


def setup_models(cls, pawprint, data):
    cls.phi1_lim = (np.min(data['phi1']), np.max(data['phi1']))
    cls.phi2_lim = (np.min(data['phi2']), np.max(data['phi2']))
    pm1_lim = (np.min(pawprint.pmprint.vertices[:,0]), np.max(pawprint.pmprint.vertices[:,0]))
    pm2_lim = (np.min(pawprint.pmprint.vertices[:,1]), np.max(pawprint.pmprint.vertices[:,1]))

    cls.coord_bounds = {"phi1": cls.phi1_lim, "phi2": cls.phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim}
    
    cls.default_grids = {
        "phi1": np.arange(*cls.coord_bounds["phi1"], 0.2),
        "phi2": np.arange(*cls.coord_bounds["phi2"], 0.1),
        "pm1": np.arange(*cls.coord_bounds["pm1"], 0.025),
        "pm2": np.arange(*cls.coord_bounds["pm2"], 0.025),
    }
    return cls

def make_bkg_model(cls, pawprint, data, knot_sep, phi2_bkg=False):

    cls = setup_models(cls, pawprint, data)

    cls.phi1_locs = get_grid(*cls.phi1_lim, knot_sep, pad_num=1).reshape(-1, 1)
    cls.phi2_locs = get_grid(*cls.phi2_lim, knot_sep, pad_num=1).reshape(-1, 1) # works for m92 but knot_sep might not be good in general
    cls.phi2_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)
    cls.pm1_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)
    cls.pm2_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)
    
    cls.variables = {
        "phi1": GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(-8.0, 8.0).expand((cls.phi1_locs.shape[0] - 1,)),
            },
            locs=cls.phi1_locs,
            scales=np.full_like(cls.phi1_locs, 10.0),
            coord_bounds=cls.phi1_lim,
        ),
        "phi2": (GridGMMVariable(
                    param_priors={
                        "zs": dist.Uniform(-8.0, 8.0).expand((cls.phi2_locs.shape[0] - 1,)),
                    },
                    locs=cls.phi2_locs,
                    scales=np.full_like(cls.phi2_locs, 10.0),
                    coord_bounds=cls.phi2_lim,
                )
            )
            # (Normal1DSplineVariable(
            #     param_priors={
            #         "mean": dist.Uniform(-20,20).expand(cls.phi2_knots.shape),
            #         "ln_std": dist.Uniform(0.5,5).expand(cls.phi2_knots.shape),
            #     },
            #     knots=cls.phi2_knots,
            #     # spline_ks = {"w":1},
            #     coord_bounds=cls.coord_bounds["phi2"],
            #     )
            # )
            if phi2_bkg else 
                (UniformVariable(
                    param_priors={}, coord_bounds=cls.coord_bounds["phi2"]
                )),
        "pm1": Normal1DSplineMixtureVariable(
            param_priors={
                "w": dist.Uniform(0, 1).expand((cls.pm1_knots.size,)),
                "mean1": dist.Uniform(-20, 20).expand((cls.pm1_knots.size,)),
                "mean2": dist.Uniform(-20, 20).expand((cls.pm1_knots.size,)),
                "ln_std1": dist.Uniform(-2, 3).expand((cls.pm1_knots.size,)),
                "ln_std2": dist.Uniform(-2, 3).expand((cls.pm1_knots.size,)),
            },
            knots=cls.pm1_knots,
            spline_ks={"w": 1},
            coord_bounds=cls.coord_bounds.get("pm1"),
        ),
        "pm2": Normal1DSplineMixtureVariable(
            param_priors={
                "w": dist.Uniform(0, 1).expand((cls.pm2_knots.size,)),
                "mean1": dist.Uniform(-20, 20).expand((cls.pm2_knots.size,)),
                "mean2": dist.Uniform(-20, 20).expand((cls.pm2_knots.size,)),
                "ln_std1": dist.Uniform(-2, 3).expand((cls.pm2_knots.size,)),
                "ln_std2": dist.Uniform(-2, 3).expand((cls.pm2_knots.size,)),
            },
            knots=cls.pm2_knots,
            spline_ks={"w": 1},
            coord_bounds=cls.coord_bounds.get("pm2"),
        ),
    }
    
    if phi2_bkg: 
        cls.data_required = {
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
        # "phi2": {"x": "phi1", "y": "phi2"},
    }
        # cls._data_required['phi2'] = {"x": "phi1", "y": "phi2"}
        cls._data_required['phi2'] = {"y": "phi2", "y_err": "phi2_err"}
    else:
        cls.data_required = {
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    return cls


def make_stream_model(cls, pawprint, data, knot_sep):
    cls = setup_models(cls, pawprint, data)
    
    # cls.phi1_lim =(np.min(pawprint.skyprint['stream'].vertices[:,0]), np.max(pawprint.skyprint['stream'].vertices[:,0]))
    cls.phi1_locs = get_grid(*cls.phi1_lim, cls.phi1_dens_step, pad_num=1).reshape(-1, 1)

    cls.phi2_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º

    cls.pm1_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º
    cls.pm2_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º

    cls.variables = {
        "phi1": GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(
                    jnp.full(cls.phi1_locs.shape[0] - 1, -8),
                    jnp.full(cls.phi1_locs.shape[0] - 1, 8),
                )
            },
            locs=cls.phi1_locs,
            scales=np.full_like(cls.phi1_locs, cls.phi1_dens_step),
            coord_bounds=cls.phi1_lim,
        ),
        "phi2": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(
                    *cls.coord_bounds.get("phi2")).expand(cls.phi2_knots.shape
                ),
                "ln_std": dist.Uniform(
                    jnp.full_like(cls.phi2_knots, -2.0), jnp.full_like(cls.phi2_knots, 2)
                ),
            },
            knots=cls.phi2_knots,
            coord_bounds=cls.coord_bounds["phi2"],
        ),
        "pm1": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(*cls.coord_bounds.get("pm1")).expand(
                    cls.pm1_knots.shape
                ),
                "ln_std": dist.Uniform(-5, 0).expand(cls.pm1_knots.shape),  # ~20 km/s
            },
            knots=cls.pm1_knots,
            coord_bounds=cls.coord_bounds.get("pm1"),
        ),
        "pm2": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(*cls.coord_bounds.get("pm2")).expand(cls.pm2_knots.shape),
                "ln_std": dist.Uniform(-5, 0).expand(cls.pm2_knots.shape),  # ~20 km/s
            },
            knots=cls.pm2_knots,
            coord_bounds=cls.coord_bounds.get("pm2"),
        ),
    }
    cls.data_required = {
        "phi2": {"x": "phi1", "y": "phi2"},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    return cls

def make_offtrack_model(cls, pawprint, data, dens_steps, StrModel):
    '''
    dens_steps: array - [phi1_dens_steps, phi2_dens_steps]
    '''
    
    cls = setup_models(cls, pawprint, data)
    cls.dens_phi1_lim = (np.min(data['phi1']), np.max(data['phi1']))
    cls.dens_phi2_lim = (np.min(data['phi2']), np.max(data['phi2']))
    cls.dens_steps = dens_steps
    
    cls.dens_locs = np.stack(
        np.meshgrid(
            np.arange(cls.dens_phi1_lim[0], cls.dens_phi1_lim[1] + 1e-3, dens_steps[0]),
            np.arange(cls.dens_phi2_lim[0], cls.dens_phi2_lim[1] + 1e-3, dens_steps[1]),
        )
    ).T.reshape(-1, 2)

    # spar_steps = 5*dens_steps

    # cls.spar_locs = np.stack(
    #     np.meshgrid(
    #         get_grid(*cls.coord_bounds["phi1"], spar_steps[0], pad_num=1),
    #         get_grid(*cls.coord_bounds["phi2"], spar_steps[1], pad_num=1),
    #     )
    # ).T.reshape(-1, 2)
    # _mask = (
    #     (cls.spar_locs[:, 0] >= cls.dens_phi1_lim[0])
    #     & (cls.spar_locs[:, 0] <= cls.dens_phi1_lim[1])
    #     & (cls.spar_locs[:, 1] >= cls.dens_phi2_lim[0])
    #     & (cls.spar_locs[:, 1] <= cls.dens_phi2_lim[1])
    # )
    # cls.spar_locs = cls.spar_locs[~_mask]

    # cls.phi12_locs = np.concatenate((cls.dens_locs, cls.spar_locs))
    # cls.phi12_scales = np.concatenate(
    #     (np.full_like(cls.dens_locs, cls.dens_steps[0]), np.full_like(cls.spar_locs, cls.spar_steps[0]))
    # )
    # cls.phi12_scales[: cls.dens_locs.shape[0], 1] = cls.dens_steps[1]
    # cls.phi12_scales[cls.dens_locs.shape[0] :, 1] = cls.spar_steps[1]
    
    cls.phi12_locs = cls.dens_locs
    cls.phi12_scales = (np.full_like(cls.phi12_locs, cls.dens_steps[0]))
    cls.phi12_scales[:, 1] = cls.dens_steps[1]

    cls.variables = {
        ("phi1", "phi2"): GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(-8.0, 8.0).expand((cls.phi12_locs.shape[0] - 1,))
                #                 "zs": dist.TruncatedNormal(
                #                     loc=-8, scale=4.0, low=-8.0, high=8.0
                #                 ).expand((phi12_locs.shape[0] - 1,))
            },
            locs=cls.phi12_locs,
            scales=cls.phi12_scales,
            coord_bounds=(
                np.array(
                    [cls.coord_bounds["phi1"][0], cls.coord_bounds["phi2"][0]]
                ),
                np.array(
                    [cls.coord_bounds["phi1"][1], cls.coord_bounds["phi2"][1]]
                ),
            ),
        ),
        "pm1": StrModel.variables["pm1"],
        "pm2": StrModel.variables["pm2"],
    }

    cls.data_required = {
        ("phi1", "phi2"): {"y": ("phi1", "phi2")},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    return cls

def make_mixture_model(cls, Components):
    
    cls.components = [C for C in Components] # without calling on parameters yet
    
    return cls
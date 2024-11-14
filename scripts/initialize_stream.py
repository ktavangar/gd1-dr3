import jax.numpy as jnp
import numpyro.distributions as dist
from stream_membership import ComponentMixtureModel, ModelComponent
from stream_membership.distributions import (
    IndependentGMM, 
    TruncatedNormalSpline, 
    DirichletSpline, 
    TruncatedNormal1DSplineMixture
)
from scipy.stats import binned_statistic
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

def get_bounds_and_grids(data, pawprint):
    
    phi1_lim = (jnp.min(data['phi1']), jnp.max(data['phi1']))
    phi2_lim = (jnp.min(data['phi2']), jnp.max(data['phi2']))
    pm1_lim = (jnp.min(pawprint.pmprint.vertices[:,0]), jnp.max(pawprint.pmprint.vertices[:,0]))
    pm2_lim = (jnp.min(pawprint.pmprint.vertices[:,1]), jnp.max(pawprint.pmprint.vertices[:,1]))

    coord_bounds = {"phi1": phi1_lim, "phi2": phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim}
    
    plot_grids = {
    "phi1": jnp.linspace(*phi1_lim, 512),
    "phi2": jnp.linspace(*phi2_lim, 128),
    "pm1": jnp.linspace(*pm1_lim, 128),
    "pm2": jnp.linspace(*pm2_lim, 128)
}
    return coord_bounds, plot_grids

def interpolate_stream_tracks(stream_data, phi1_lim):
    _phi2_stat = binned_statistic(stream_data["phi1"], stream_data["phi2"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 21))
    _phi2_interp = IUS(
        0.5 * (_phi2_stat.bin_edges[:-1] + _phi2_stat.bin_edges[1:]), _phi2_stat.statistic, ext=0, k=1
    )
    
    _pm1_stat = binned_statistic(stream_data["phi1"], stream_data["pm1"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _pm1_interp = IUS(
        0.5 * (_pm1_stat.bin_edges[:-1] + _pm1_stat.bin_edges[1:]), _pm1_stat.statistic, ext=0, k=1
    )
    
    _pm2_stat = binned_statistic(stream_data["phi1"], stream_data["pm2"], bins=jnp.linspace(phi1_lim[0], phi1_lim[1], 32))
    _pm2_interp = IUS(
        0.5 * (_pm2_stat.bin_edges[:-1] + _pm2_stat.bin_edges[1:]), _pm2_stat.statistic, ext=0, k=1
    )

    interp_dict = {'phi2': _phi2_interp, 'pm1': _pm1_interp, 'pm2': _pm2_interp}

    return interp_dict

def make_bkg_model_component(knot_spacings, n_pm_mixture, coord_bounds, data):
    
    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']
    pm1_lim  = coord_bounds['pm1']
    pm2_lim  = coord_bounds['pm2']

    phi1_knot_spacing, pm1_knot_spacing, pm2_knot_spacing = knot_spacings

    bkg_phi1_knots = jnp.arange(jnp.around(phi1_lim[0]),# + phi1_knot_spacing/2, 
                                jnp.around(phi1_lim[1])+1e-3,# - phi1_knot_spacing/2 + 1e-3, 
                                phi1_knot_spacing)
    bkg_pm1_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm1_knot_spacing)
    bkg_pm2_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm2_knot_spacing)
    
    bkg_model = ModelComponent(
        name="background",
        coord_distributions={
            "phi1": IndependentGMM,
            "phi2": dist.Uniform,
            "pm1": TruncatedNormal1DSplineMixture,
            "pm2": TruncatedNormal1DSplineMixture,
        },
        coord_parameters={
            "phi1": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(len(bkg_phi1_knots))),
                ),
                "locs": bkg_phi1_knots.reshape(1, -1),
                "scales": dist.HalfNormal(phi1_knot_spacing/2),
                "low":  jnp.array([phi1_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1]])[:, None],                  
            },
            "phi2": {
                "low": phi2_lim[0],
                "high": phi2_lim[1],
            },
            "pm1": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(n_pm_mixture)) # two truncated normals
                ),
                "loc_vals": dist.Uniform(low=-10, high=10).expand([n_pm_mixture, bkg_pm1_knots.shape[0]]),
                "scale_vals": dist.HalfNormal(5).expand([n_pm_mixture, bkg_pm1_knots.shape[0]]),
                "knots": bkg_pm1_knots,
                "x": data["phi1"],
                "low": pm1_lim[0],
                "high": pm1_lim[1],
                "spline_k": 3,
                "clip_locs": (-10,10),
                "clip_scales": (1, None),
            },
            "pm2": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(n_pm_mixture))
                ),
                "loc_vals": dist.Uniform(low=-10, high=10).expand([n_pm_mixture, bkg_pm2_knots.shape[0]]),
                "scale_vals": dist.HalfNormal(5).expand([n_pm_mixture, bkg_pm2_knots.shape[0]]),
                "knots": bkg_pm2_knots,
                "x": data["phi1"],
                "low": pm2_lim[0],
                "high": pm2_lim[1],
                "spline_k": 3,
                "clip_locs": (-10, 10),
                "clip_scales": (1, None),
            }
        },
        conditional_data={"pm1": {"x": "phi1"}, 
                          "pm2": {"x": "phi1"}},
    )

    return bkg_model

def make_stream_model_component(knot_spacings, coord_bounds, data):
    
    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']
    pm1_lim  = coord_bounds['pm1']
    pm2_lim  = coord_bounds['pm2']

    phi1_knot_spacing, phi2_knot_spacing, pm1_knot_spacing, pm2_knot_spacing = knot_spacings

    stream_phi1_knots = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, phi1_knot_spacing)
    stream_phi2_knots = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, phi1_knot_spacing)
    stream_pm1_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm1_knot_spacing)
    stream_pm2_knots  = jnp.arange(jnp.around(phi1_lim[0]), jnp.around(phi1_lim[1]) + 1e-3, pm2_knot_spacing)

    _interp_dict = interpolate_stream_tracks(data, phi1_lim)
    eval_interp_phi2 = jnp.array(_interp_dict['phi2'](stream_phi2_knots))
    eval_interp_pm1 = jnp.array(_interp_dict['pm1'](stream_pm1_knots))
    eval_interp_pm2 = jnp.array(_interp_dict['pm2'](stream_pm2_knots))
    
    stream_model = ModelComponent(
        name="stream",
        coord_distributions={
            "phi1": IndependentGMM,
            "phi2": TruncatedNormalSpline,
            "pm1": TruncatedNormalSpline,
            "pm2": TruncatedNormalSpline,
        },
        coord_parameters={
            "phi1": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(len(stream_phi1_knots))),
                ),
                "locs": stream_phi1_knots.reshape(1, -1),
                "scales": dist.HalfNormal(5.0),
                "low": jnp.array([phi1_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1]])[:, None],
            },
            "phi2": {
                # "loc_vals": dist.Uniform(*phi2_lim).expand([stream_phi2_knots.shape[0]]),
                "loc_vals": dist.Normal(loc=eval_interp_phi2, scale=1),
                "scale_vals": dist.HalfNormal(0.5).expand([stream_phi2_knots.shape[0]]),
                "knots": stream_phi2_knots,
                "x": data["phi1"],
                "low": phi2_lim[0],
                "high": phi2_lim[1],
                "spline_k": 3,
                "clip_scales": (1e-1, None),
            },
            "pm1": {
                # "loc_vals": dist.Uniform(*pm1_lim).expand([stream_pm1_knots.shape[0]]),
                "loc_vals": dist.Normal(loc=eval_interp_pm1, scale=2),
                "scale_vals": dist.HalfNormal(0.5).expand([stream_pm1_knots.shape[0]]),
                "knots": stream_pm1_knots,
                "x": data["phi1"],
                "low": pm1_lim[0],
                "high": pm1_lim[1],
                "spline_k": 3,
                "clip_locs": pm1_lim,
                "clip_scales": (1e-1, None),
            },
            "pm2": {
                # "loc_vals": dist.Uniform(*pm2_lim).expand([stream_pm2_knots.shape[0]]),
                "loc_vals": dist.Normal(loc=eval_interp_pm2, scale=2),
                "scale_vals": dist.HalfNormal(0.5).expand([stream_pm2_knots.shape[0]]),
                "knots": stream_pm2_knots,
                "x": data["phi1"],
                "low": pm2_lim[0],
                "high": pm2_lim[1],
                "spline_k": 3,
                "clip_locs": pm2_lim,
                "clip_scales": (1e-1, None),
            },
        },
        conditional_data={"phi2": {"x": "phi1"}, "pm1": {"x": "phi1"}, "pm2": {"x": "phi1"}},
    )

    return stream_model

def make_offtrack_model_component(offtrack_dx, stream_model, coord_bounds):
    '''
    offtrack_dx: [offtrack_phi1_dx, offtrack_phi2_dx]
    '''

    offtrack_phi1_dx, offtrack_phi2_dx = offtrack_dx

    phi1_lim = coord_bounds['phi1']
    phi2_lim = coord_bounds['phi2']

    offtrack_phi12_locs = jnp.stack(
    jnp.meshgrid(
        jnp.arange(phi1_lim[0]+offtrack_phi1_dx, 
                   phi1_lim[1]-offtrack_phi1_dx/2 + 1e-3, #factor of two added by hand to make sure the high phi1 edge is properly included
                   offtrack_phi1_dx),
        jnp.arange(phi2_lim[0]+offtrack_phi2_dx, 
                   phi2_lim[1]-offtrack_phi2_dx/2 + 1e-3, 
                   offtrack_phi2_dx),
    ),
    axis=-1,
    ).reshape(-1, 2)

    # # Remove nodes in areas we don't have data:
    # offtrack_node_mask = np.ones(offtrack_phi12_locs.shape[0], dtype=bool)
    # pt1 = (-55, 0)
    # pt2 = (-40, 2)
    # m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    # offtrack_node_mask &= offtrack_phi12_locs[:, 1] < (
    #     m * (offtrack_phi12_locs[:, 0] - pt1[0]) + pt1[1]
    # )
    # offtrack_node_mask &= offtrack_phi12_locs[:, 1] > (
    #     m * (offtrack_phi12_locs[:, 0] - pt1[0]) + pt1[1] - 4.0
    # )

    offtrack_model = ModelComponent(
        name="offtrack",
        coord_distributions={
            ("phi1", "phi2"): IndependentGMM,
            "pm1": stream_model.coord_distributions["pm1"],
            "pm2": stream_model.coord_distributions["pm2"],
        },
        coord_parameters={
            ("phi1", "phi2"): {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.full(offtrack_phi12_locs.shape[0], 0.5)),
                ),
                "locs": offtrack_phi12_locs.T,
                "scales":
                    # dist.HalfNormal(jnp.array([offtrack_phi1_dx, offtrack_phi2_dx])[:, None]).expand(offtrack_phi12_locs.T.shape), 
                    dist.TruncatedNormal(loc=0.5*jnp.array(offtrack_dx)[:, None], scale=0.5, 
                                         low=0.1*jnp.array(offtrack_dx)[:, None]).expand(offtrack_phi12_locs.T.shape),
                "low": jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
            },
            # We don't need to define the parameters here, as they are the same as for the
            # stream model - this will be handled below when we combine the components into
            # a mixture model
            "pm1": stream_model.coord_parameters["pm1"],
            "pm2": stream_model.coord_parameters["pm2"],
        },
        conditional_data={"pm1": {"x": "phi1"}, "pm2": {"x": "phi1"}},
    )

    return offtrack_model, offtrack_phi12_locs
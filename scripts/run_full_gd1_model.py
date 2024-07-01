import astropy.table as at
import astropy.units as u

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal

numpyro.enable_x64()
numpyro.set_host_device_count(2)
#from numpyro_ext.optim import optimize

import pickle

from scipy.stats import binned_statistic
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

import sys
import importlib
sys.path.append('/Users/Tavangar/Work/stream-membership/')
sys.path.append('../../')
sys.path.append('/Users/Tavangar/Work/CATS_workshop/cats/')
from stream_membership import StreamMixtureModel
from stream_membership.plot import plot_data_projections
from gd1_helpers.membership.gd1_model import (
    Base,
    BackgroundModel,
    StreamDensModel,
    OffTrackModel,
    MixtureModel
)

from scripts.gd1_init import *

from cats.pawprint.pawprint import Pawprint, Footprint2D
from cats.CMD import Isochrone
from cats.inputs import stream_inputs as inputs
from cats.proper_motions import ProperMotionSelection, rough_pm_poly

if __name__ == '__main__':

    #####################
    ## Setup from CATS ##
    #####################
    print('Setting up CATS')
    cat = at.Table.read("/Users/Tavangar/Work/CATS_Workshop/cats/data/joined-GD-1.fits")
    
    cat['pm1'] = cat['pm_phi1_cosphi2_unrefl']
    cat['pm2'] = cat['pm_phi2_unrefl']
    
    stream='GD-1'
    phi1_lim = (np.min(cat['phi1']), np.max(cat['phi1']))
    phi1_lim = [-100,20]
    
    cat = cat[(cat['phi1'] < phi1_lim[1]) & \
              (cat['phi1'] > phi1_lim[0]) & \
              (cat['phot_g_mean_mag'] < 20.5)] # clunky to hard code this maybe

    p = Pawprint.pawprint_from_galstreams(inputs[stream]['short_name'],
                                          inputs[stream]['pawprint_id'],
                                          width=inputs[stream]['width'] * u.deg,
                                          phi1_lim=phi1_lim)
    
    # rough pm cut to start with (this comes only from the galstreams proper motion tracks)
    p.pmprint, pm_mask = rough_pm_poly(p, cat, buffer=2)
    
    # Create the CMD cuts
    o = Isochrone(stream, cat, pawprint=p)
    _, iso_mask, _, hb_mask, pprint = o.simpleSln(maxmag=22, mass_thresh=0.83)
    
    pmsel = ProperMotionSelection(stream, cat, pprint,
                                  n_dispersion_phi1=3, n_dispersion_phi2=3, cutoff=0.1)
    
    ##### need this a second time
    p.pmprint, pm_mask = rough_pm_poly(p, cat, buffer=2)

    #####################
    ## Setup the Model ##
    #####################
    print('Setting up the model')
    
    run_data_ = o.cat[pm_mask & (iso_mask | hb_mask)]
    run_data = {k: np.array(run_data_[k], dtype="f8") for k in run_data_.colnames}
    
    bkg_data_ = o.cat[pm_mask & (iso_mask | hb_mask) & ~o.on_skymask]
    bkg_data = {k: np.array(bkg_data_[k], dtype="f8") for k in bkg_data_.colnames}
    
    stream_data_ = o.cat[pmsel.pm12_mask & (iso_mask | hb_mask) & o.on_skymask]
    stream_data = {k: np.array(stream_data_[k], dtype="f8") for k in stream_data_.colnames}


    #################################
    ## Setup Variational Inference ##
    #################################
    
    sep = 'xsmall'

    if sep == 'xsmall':
        knot_sep = 5
        dens_steps = np.array([2, 0.2]) #small, goes with knot_step=5
    elif sep == 'small':
        knot_sep = 10
        dens_steps = np.array([4, 0.4]) #medium, goes with knot_step=10
    elif sep == 'medium':
        knot_sep = 15
        dens_steps = np.array([6, 0.6]) #large, goes with knot_step=15
    elif sep == 'large':
        knot_sep = 20
        dens_steps = np.array([8, 0.8])

    

    print('Loading Parameters from Previous Optimization...')
    with open('/Users/Tavangar/Work/gd1-dr3/data/full_model_opt_params_tied_params_no_bounds.pkl', 'rb') as input_file:
        params = pickle.load(input_file)
    dens_steps = np.array([5,0.5])
    
    BkgModel = make_bkg_model(BackgroundModel, p, cat, knot_sep=knot_sep, phi2_bkg=False)
    StrModel = make_stream_model(StreamDensModel, p, cat, knot_sep=knot_sep)
    OffModel = make_offtrack_model(OffTrackModel, p, cat, dens_steps=dens_steps)
    
    FullComponents = [StrModel, BkgModel, OffModel]
    MixModel = make_mixture_model(StreamMixtureModel, FullComponents)

    tied_params = [
        (("offtrack", "pm1"), ("stream", "pm1")),
        (("offtrack", "pm2"), ("stream", "pm2")),
    ]

    print('Starting Variational Inference...')
    rng_key = jax.random.PRNGKey(8675309)
    
    guide = AutoNormal(model=MixModel.setup_numpyro)
    optimizer = numpyro.optim.Adam(step_size=1e-1) #play with this
    svi = SVI(MixModel.setup_numpyro, guide, optimizer, loss=Trace_ELBO())
    rng_key, rng_subkey = jax.random.split(key=rng_key)
    svi_result = svi.run(
        rng_subkey,
        2_000,
        data=run_data,
        init_params=params
    )

    var_inf_params = svi_result.params
    posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), var_inf_params, sample_shape=(10000,))

    with open('/Users/Tavangar/Work/gd1-dr3/data/var_inf_results/params.pkl', 'wb') as param_file:
        pickle.dump(var_inf_params, param_file)
    with open('/Users/Tavangar/Work/gd1-dr3/data/var_inf_results/posterior_samples.pkl', 'wb') as posterior_file:
        pickle.dump(posterior_samples, posterior_file)

    

    # ######################
    # ## Background Model ##
    # ######################
    # print('Background Optimization')
    
    # BkgModel = make_bkg_model(BackgroundModel, p, cat, knot_sep=knot_sep, phi2_bkg=False)

    # bkg_init_p = {
    #     "ln_N": np.log(len(bkg_data['phi1'])),
    #     "phi1": {'zs': np.zeros(BackgroundModel.phi1_locs.shape[0]-1)},
    #     "phi2": {},
    #     "pm1": {
    #         "w": np.full_like(BackgroundModel.pm1_knots, 0.5),
    #         "mean1": np.full_like(BackgroundModel.pm1_knots, 0),
    #         "ln_std1": np.full_like(BackgroundModel.pm1_knots, 1),
    #         "mean2": np.full_like(BackgroundModel.pm1_knots, 5),
    #         "ln_std2": np.full_like(BackgroundModel.pm1_knots, 2)
    #     },
    #     "pm2": {
    #         "w": np.full_like(BackgroundModel.pm2_knots, 0.5),
    #         "mean1": np.full_like(BackgroundModel.pm2_knots, -2.),
    #         "ln_std1": np.full_like(BackgroundModel.pm2_knots, 1),
    #         "mean2": np.full_like(BackgroundModel.pm2_knots, -3),
    #         "ln_std2": np.full_like(BackgroundModel.pm2_knots, 2)
    #     },
    # }
    
    # background_init = BkgModel(bkg_init_p)

    # bkg_opt_pars, bkg_info = background_init.optimize(
    #     data=bkg_data,
    #     init_params=bkg_init_p,
    #     use_bounds=True,
    #     jaxopt_kwargs=dict(maxiter=4096),
    # )
    # print(bkg_info)

    # ##################
    # ## Stream Model ##
    # ##################
    # print('Stream Optimization')
    # StrModel = make_stream_model(StreamDensModel, p, cat, knot_sep=knot_sep)

    # # TODO: replace this with galstreams initialization
    # _phi2_stat = binned_statistic(stream_data["phi1"], stream_data["phi2"], bins=np.linspace(phi1_lim[0], phi1_lim[1], 21))
    # _phi2_interp = IUS(
    #     0.5 * (_phi2_stat.bin_edges[:-1] + _phi2_stat.bin_edges[1:]), _phi2_stat.statistic, ext=0, k=1
    # )
    
    # _pm1_stat = binned_statistic(stream_data["phi1"], stream_data["pm1"], bins=np.linspace(phi1_lim[0], phi1_lim[1], 32))
    # _pm1_interp = IUS(
    #     0.5 * (_pm1_stat.bin_edges[:-1] + _pm1_stat.bin_edges[1:]), _pm1_stat.statistic, ext=0, k=1
    # )
    
    # _pm2_stat = binned_statistic(stream_data["phi1"], stream_data["pm2"], bins=np.linspace(phi1_lim[0], phi1_lim[1], 32))
    # _pm2_interp = IUS(
    #     0.5 * (_pm2_stat.bin_edges[:-1] + _pm2_stat.bin_edges[1:]), _pm2_stat.statistic, ext=0, k=1
    # )

    # stream_init_p = {
    #     "ln_N": np.log(len(stream_data['phi1'])),
    #     "phi1": {
    #         "zs": np.zeros(StreamDensModel.phi1_locs.shape[0]-1)
    #     },
    #     "phi2": {
    #         "mean": _phi2_interp(StreamDensModel.phi2_knots),
    #         "ln_std": np.full_like(StreamDensModel.phi2_knots, -0.5)
    #     },
    #     "pm1": {
    #         "mean": _pm1_interp(StreamDensModel.pm1_knots),
    #         "ln_std": np.full_like(StreamDensModel.pm1_knots, -0.5)
    #     },
    #     "pm2": {
    #         "mean": _pm2_interp(StreamDensModel.pm2_knots),
    #         "ln_std": np.full_like(StreamDensModel.pm2_knots, -0.5)
    #     }
    # }

    # stream_init = StrModel(stream_init_p)

    # stream_opt_pars, stream_info = stream_init.optimize(
    #     data=stream_data, init_params=stream_init_p, use_bounds=True
    # )
    # print(stream_info)

    # ###############################
    # ## Stream + Background Model ##
    # ###############################
    # ## This is just as an intermediate step, it is not used again

    # # Components = [StreamDensModel, BackgroundModel]
    # # mix_params0 = {"stream": stream_opt_pars, "background": bkg_opt_pars}

    # # mix_opt_pars, mix_info = StreamMixtureModel.optimize(
    # #     data=run_data, Components=Components, init_params=mix_params0, use_bounds=True
    # # )
    # # print(mix_info)

    # ####################
    # ## Offtrack Model ##
    # ####################
    # print('Offtrack Initialization')

    # OffModel = make_offtrack_model(OffTrackModel, p, cat, dens_steps=dens_steps)
    
    # offtrack_init_p = {
    #     "ln_N": np.log(500),
    #     ("phi1", "phi2"): {
    #         "zs": np.zeros(OffModel.phi12_locs.shape[0] - 1)
    #     },
    #     "pm1": stream_opt_pars["pm1"].copy(),
    #     "pm2": stream_opt_pars["pm2"].copy()
    # }

    # ########################
    # ## Full Mixture Model ##
    # ########################
    # print('Full Model Optimization')

    # full_Components =  [StrModel, BkgModel, OffModel]
    # full_mix_params0 = {
    #     "stream": stream_opt_pars,
    #     "background": bkg_opt_pars,
    #     "offtrack": offtrack_init_p,
    # }

    # tied_params = [
    #     (("offtrack", "pm1"), ("stream", "pm1")),
    #     (("offtrack", "pm2"), ("stream", "pm2")),
    # ]
    # full_mix_init = StreamMixtureModel(
    #     full_mix_params0, full_Components, tied_params=tied_params
    # )

    # full_mix_opt_pars, full_mix_info = StreamMixtureModel.optimize(
    #     data=run_data,
    #     Components=full_Components,
    #     #tied_params=tied_params,
    #     init_params=full_mix_params0,
    #     use_bounds=True,
    # )
    # print(full_mix_info)
    
    # with open('/Users/Tavangar/Work/gd1-dr3/data/full_model_opt_params_{}_sep.pkl'.format(sep), 'wb') as output_file:
    #     pickle.dump(full_mix_opt_pars, output_file)

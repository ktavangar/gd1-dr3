import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoDiagonalNormal

from stream_membership import ComponentMixtureModel

import astropy.table as at
import astropy.units as u

import pickle
import json
from pathlib import Path

import sys
sys.path.append('/Users/Tavangar/Work/gd1-dr3/scripts/')

import initialize_stream as init_stream

from cats.pawprint.pawprint import Pawprint
from cats.CMD import Isochrone
from cats.inputs import stream_inputs as inputs
from cats.proper_motions import ProperMotionSelection, rough_pm_poly

def run_CATS(data, stream_name, phi1_lim):
    
    cat = data[data['phot_g_mean_mag'] <20.5]
    cat['pm1'] = cat['pm_phi1_cosphi2_unrefl']
    cat['pm2'] = cat['pm_phi2_unrefl']
    
    cat = cat[(cat['phi1'] < phi1_lim[1]) & (cat['phi1'] > phi1_lim[0])] # clunky to hard code this

    pawprint = Pawprint.pawprint_from_galstreams(inputs[stream_name]['short_name'],
                                                 inputs[stream_name]['pawprint_id'],
                                                 width=inputs[stream_name]['width'] * u.deg,
                                                 phi1_lim=phi1_lim)
    
    # rough pm cut to start with (this comes only from the galstreams proper motion tracks)
    pawprint.pmprint, pm_mask = rough_pm_poly(pawprint, cat, buffer=2)
    
    # Create the CMD cuts
    iso_obj_ = Isochrone(cat, inputs[stream_name], pawprint=pawprint)
    _, _, _, _, pprint = iso_obj_.simpleSln(maxmag=22, mass_thresh=0.83)
    
    pmsel = ProperMotionSelection(cat, inputs[stream_name], pprint,
                                  n_dispersion_phi1=3, n_dispersion_phi2=3, cutoff=0.1)

    iso_obj = Isochrone(cat, inputs[stream_name], pawprint=pmsel.pawprint,)
    _, iso_mask, _, hb_mask, pprint = iso_obj.simpleSln(22, mass_thresh=0.83)

    
    ##### rough pm cut to start with (this comes only from the galstreams proper motion tracks)
    pawprint.pmprint, pm_mask = rough_pm_poly(pawprint, cat, buffer=2)

    return pawprint, iso_obj, iso_mask, hb_mask, pmsel, pm_mask

def run_SVI(model, init_params, data, err=None, num_steps=10_000, GuideFunction=AutoNormal, keys=jax.random.split(jax.random.PRNGKey(42), num=2), 
            learning_rate=1e-2, num_particles=1):
    optimizer = numpyro.optim.ClippedAdam(learning_rate, eps=1e-8) # consider using optax for flexible learning rate scheduling
    
    guide = GuideFunction(
        model, init_loc_fn=numpyro.infer.init_to_value(values=init_params)
    )
    MAP_svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles))

    with numpyro.validation_enabled(), jax.debug_nans():
        init_state = MAP_svi.init(keys[0], init_params=init_params, data=data, err=err)
        MAP_svi_results = MAP_svi.run(
        rng_key=keys[1], num_steps=num_steps, init_state=init_state, data=data, err=err)
        
    return MAP_svi_results, guide

def get_svi_params(model, data, svi_results, guide, num_samples=1, key=jax.random.PRNGKey(12345)):

    pred_dist = Predictive(guide, params=svi_results.params, num_samples=num_samples)
    pars_ = pred_dist(key, data=data)
    pars = {k: jnp.median(v, axis=0) for k, v in pars_.items() if (k != '_auto_latent' and k != 'mixture:modeldata')}
    pars_expanded = model.expand_numpyro_params(pars)

    return pars_expanded

def main(bkg_knot_spacings, stream_knot_spacings, offtrack_dx, 
         bkg_filename, stream_filename, stream_bkg_mm_filename, full_filename):
    #####################
    ## Setup from CATS ##
    #####################
    print('Setting up the model')

    try:
        with open('/Users/Tavangar/Work/gd1-dr3/data/post_cats_data.pkl', 'rb') as input_file_:
            post_cats = pickle.load(input_file_)
        run_data_ = post_cats['run_data']
        bkg_data_ = post_cats['bkg_data']
        stream_data_ = post_cats['stream_data']
        pawprint = post_cats['pawprint']
    except:
        data = at.Table.read("/Users/Tavangar/Work/CATS_Workshop/cats/data/joined-GD-1.fits")
    
        pawprint,iso_obj,iso_mask,hb_mask,pmsel,pm_mask = run_CATS(data, stream_name='GD-1', phi1_lim=[-100,20])
        run_data_ = iso_obj.cat[pm_mask & (iso_mask | hb_mask)]
        bkg_data_ = iso_obj.cat[pm_mask & (iso_mask | hb_mask) & ~iso_obj.on_skymask]
        stream_data_ = iso_obj.cat[pmsel.pm12_mask & (iso_mask | hb_mask) & iso_obj.on_skymask]

    #####################
    ## Setup the Model ##
    #####################
    
    run_data = {k: jnp.array(run_data_[k], dtype="f8") for k in ['phi1', 'phi2', 'pm1', 'pm2']}#run_data_.colnames}
    run_data_err = {'pm1': jnp.array(run_data_['pm1_error'], dtype="f8"),
                    'pm2': jnp.array(run_data_['pm2_error'], dtype="f8")}
    
    bkg_data = {k: jnp.array(bkg_data_[k], dtype="f8") for k in ['phi1', 'phi2', 'pm1', 'pm2']}#bkg_data_.colnames}
    bkg_data_err = {'pm1': jnp.array(bkg_data_['pm1_error'], dtype="f8"),
                    'pm2': jnp.array(bkg_data_['pm2_error'], dtype="f8")}
    
    stream_data = {k: jnp.array(stream_data_[k], dtype="f8") for k in ['phi1', 'phi2', 'pm1', 'pm2']}#stream_data_.colnames}
    stream_data_err = {'pm1': jnp.array(stream_data_['pm1_error'], dtype="f8"),
                       'pm2': jnp.array(stream_data_['pm2_error'], dtype="f8")}

    coord_bounds, _ = init_stream.get_bounds_and_grids(run_data, pawprint)
    
    phi1_lim = coord_bounds['phi1']

    guide_function = AutoNormal

    ##########################################
    ## Create and Optimize Background Model ##
    ##########################################
    
    n_pm_mixture = 2 # number of mixture model components in the background proper motion models
        
    bkg_model = init_stream.make_bkg_model_component(knot_spacings=bkg_knot_spacings, n_pm_mixture=n_pm_mixture, 
                                                     coord_bounds=coord_bounds, data=bkg_data)
    try:
        with open(bkg_filename, 'rb') as input_file_:
            bkg_dict = pickle.load(input_file_)
        print('Background model with this node spacing already exists! Moving on to stream model')
        bkg_svi_results = bkg_dict['svi_results']
        bkg_guide = bkg_dict['guide']
    except:
        print('Creating and optimizing background model with node spacings: {}'.format(bkg_knot_spacings))
    
        n_phi1_knots = bkg_model.coord_parameters['phi1']['locs'].shape[-1]
        n_pm1_knots = bkg_model.coord_parameters['pm1']['knots'].shape[-1]
        n_pm2_knots = bkg_model.coord_parameters['pm2']['knots'].shape[-1]
        
        bkg_init_params = {
            "phi1": {
                "mixing_distribution": jnp.ones(n_phi1_knots) / n_phi1_knots,
                "scales": jnp.full(n_phi1_knots, bkg_knot_spacings[0])
            },
            "phi2": {},
            "pm1": {
                "mixing_distribution": jnp.ones(n_pm_mixture) / n_pm_mixture,
                "loc_vals": jnp.full((n_pm_mixture, n_pm1_knots), 0),
                "scale_vals": jnp.full((n_pm_mixture, n_pm1_knots), 5.0),
            },
            "pm2": {
                "mixing_distribution": jnp.ones(n_pm_mixture) / n_pm_mixture,
                "loc_vals": jnp.full((n_pm_mixture, n_pm2_knots), -3.0),
                "scale_vals": jnp.full((n_pm_mixture, n_pm2_knots), 3.0),
            }
        }
    
        keys = jax.random.split(jax.random.PRNGKey(42), num=2)
        bkg_svi_results, bkg_guide = run_SVI(bkg_model, bkg_init_params, bkg_data, bkg_data_err,
                                             num_steps=20_000, GuideFunction=guide_function, keys=keys,
                                             learning_rate=1e-3, num_particles=5)
        bkg_dict = {'svi_results': bkg_svi_results,
                    'guide': bkg_guide,
                    'bkg_knot_spacings': bkg_knot_spacings
                   }
        with open(bkg_filename, 'wb') as param_file:
            pickle.dump(bkg_dict, param_file)
        

    bkg_params = get_svi_params(model=bkg_model, data=bkg_data, svi_results=bkg_svi_results, 
                                guide=bkg_guide, num_samples=1, key=jax.random.PRNGKey(12345))

    ######################################
    ## Create and Optimize Stream Model ##
    ######################################
    
    stream_model = init_stream.make_stream_model_component(knot_spacings=stream_knot_spacings, 
                                                           coord_bounds=coord_bounds, data=stream_data)
    try:
        with open(stream_filename, 'rb') as input_file_:
            stream_dict = pickle.load(input_file_)
        print('Stream model with this node spacing already exists! Moving on to mixture models')
        stream_svi_results = stream_dict['svi_results']
        stream_guide = stream_dict['guide']
    except:
        print('Creating and Optimizing Stream Model with node spacings: {}'.format(stream_knot_spacings))
        stream_phi1_knots = stream_model.coord_parameters['phi1']['locs'][0]
        stream_phi2_knots = stream_model.coord_parameters['phi2']['knots']
        stream_pm1_knots = stream_model.coord_parameters['pm1']['knots']
        stream_pm2_knots = stream_model.coord_parameters['pm2']['knots']
        
        _interp_dict = init_stream.interpolate_stream_tracks(stream_data, phi1_lim)
        eval_interp_phi2 = jnp.array(_interp_dict['phi2'](stream_phi2_knots))
        eval_interp_pm1 = jnp.array(_interp_dict['pm1'](stream_pm1_knots))
        eval_interp_pm2 = jnp.array(_interp_dict['pm2'](stream_pm2_knots))
        
        stream_init_params = {
            "phi1": {
                "mixing_distribution": jnp.ones(len(stream_phi1_knots))
                / len(stream_phi1_knots),
                "scales": jnp.full(stream_phi1_knots.shape[0], stream_knot_spacings[0]),
            },
            "phi2": {
                "loc_vals": eval_interp_phi2,
                "scale_vals": jnp.full(stream_phi2_knots.shape[0], 0.5),
            },
            "pm1": {
                "loc_vals": eval_interp_pm1,
                "scale_vals": jnp.full(stream_pm1_knots.shape[0], 0.35),
            },
            "pm2": {
                "loc_vals": eval_interp_pm2,
                "scale_vals": jnp.full(stream_pm2_knots.shape[0], 0.35),
            },
        }
    
        keys = jax.random.split(jax.random.PRNGKey(42), num=2)
        stream_svi_results, stream_guide = run_SVI(stream_model, stream_init_params, stream_data, stream_data_err,
                                                   num_steps=50_000, GuideFunction=guide_function, keys=keys,
                                                   learning_rate=1e-3, num_particles=10)
        
        stream_dict = {'svi_results': stream_svi_results,
                       'guide': stream_guide,
                       'stream_knot_spacings': stream_knot_spacings
                      }
        with open(stream_filename, 'wb') as param_file:
            pickle.dump(stream_dict, param_file)

    stream_params = get_svi_params(model=stream_model, data=stream_data, svi_results=stream_svi_results, 
                                   guide=stream_guide, num_samples=1, key=jax.random.PRNGKey(12345))

    #########################################################
    ## Create and Optimize Background+Stream Mixture Model ##
    #########################################################

    bkg_model_mm = init_stream.make_bkg_model_component(knot_spacings=bkg_knot_spacings, n_pm_mixture=n_pm_mixture, 
                                                        coord_bounds=coord_bounds, data=run_data)
    stream_model_mm = init_stream.make_stream_model_component(knot_spacings=stream_knot_spacings, 
                                                              coord_bounds=coord_bounds, data=run_data)
    stream_bkg_mm = ComponentMixtureModel(dist.Dirichlet(jnp.array([1.0, 1.0])), 
                                          components=[bkg_model_mm, stream_model_mm])

    try:
        with open(stream_bkg_mm_filename, 'rb') as input_file_:
            stream_bkg_mm_dict = pickle.load(input_file_)
        print('Background+Stream mixture model with this node spacing already exists! Moving on to full mixture model')
        no_off_svi_results = stream_bkg_mm_dict['svi_results']
        no_off_guide = stream_bkg_mm_dict['guide']

    except:
        print('Creating and Optimizing Background+Stream Mixture Model...')
        f_stream = jnp.around(len(stream_data_)/len(run_data_), 3)
    
        init_params = {"background": bkg_params, "stream": stream_params}
        packed_params = stream_bkg_mm.pack_params(init_params)
        packed_params["mixture-probs"] = jnp.array([1-f_stream, f_stream])
        packed_params["mixture"] = jnp.stack([v for v in run_data.values()], axis=-1)
    
        keys = jax.random.split(jax.random.PRNGKey(42), num=2)
        no_off_svi_results, no_off_guide = run_SVI(stream_bkg_mm, packed_params, run_data, run_data_err,
                                                   num_steps=15_000, GuideFunction=guide_function, keys=keys,
                                                   learning_rate=1e-2, num_particles=3)
        
        no_off_dict = {'svi_results': no_off_svi_results,
                       'guide': no_off_guide,
                       'bkg_knot_spacings': bkg_knot_spacings,
                       'stream_knot_spacings': stream_knot_spacings
                      }
        with open(stream_bkg_mm_filename, 'wb') as param_file:
            pickle.dump(no_off_dict, param_file)

    no_off_params = get_svi_params(model=stream_bkg_mm, data=run_data, svi_results=no_off_svi_results, 
                                   guide=no_off_guide, num_samples=1, key=jax.random.PRNGKey(12345))

    ##########################################################
    ## Create and Run Full Mixture Model Including Offtrack ##
    ##########################################################
    print('Creating and Running Full Mixture Model Including Offtrack...')
    
    
    print('Offtrack spacings: {}'.format(offtrack_dx))
    
    offtrack_model, offtrack_phi12_locs = init_stream.make_offtrack_model_component(offtrack_dx, stream_model, coord_bounds)

    ## Untie the offtrack model and create tight priors around stream+background results instead
    ##  This can't be splines since I wouldn't expect this to vary continuously
    offtrack_model.coord_parameters['pm1']['loc_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['pm1']['loc_vals'],
        scale=no_off_params['stream']['pm1']['scale_vals'],
        low=no_off_params['stream']['pm1']['loc_vals']-3*no_off_params['stream']['pm1']['scale_vals'],
        high=no_off_params['stream']['pm1']['loc_vals']+3*no_off_params['stream']['pm1']['scale_vals']
        )
    offtrack_model.coord_parameters['pm2']['loc_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['pm2']['loc_vals'],
        scale=no_off_params['stream']['pm2']['scale_vals'],
        low=no_off_params['stream']['pm2']['loc_vals']-3*no_off_params['stream']['pm2']['scale_vals'],
        high=no_off_params['stream']['pm2']['loc_vals']+3*no_off_params['stream']['pm2']['scale_vals'])
    
    offtrack_model.coord_parameters['pm1']['scale_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['pm1']['scale_vals'],
        scale=no_off_params['stream']['pm1']['scale_vals'],
        low=0, high=3*no_off_params['stream']['pm1']['scale_vals'])
    offtrack_model.coord_parameters['pm2']['scale_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['pm2']['scale_vals'],
        scale=no_off_params['stream']['pm2']['scale_vals'],
        low=0, high=3*no_off_params['stream']['pm2']['scale_vals'])
    offtrack_model.coord_parameters['pm1']['x'] = run_data['phi1']
    offtrack_model.coord_parameters['pm2']['x'] = run_data['phi1']


    _init_prob = jnp.ones(len(offtrack_phi12_locs))
    _init_prob /= _init_prob.sum()
    
    offtrack_init_params = {
        ("phi1", "phi2"): {
            "mixing_distribution": _init_prob,
            "scales": jnp.array([offtrack_dx] * len(offtrack_phi12_locs)).T,
        },
        "pm1": no_off_params['stream']['pm1'],
        "pm2": no_off_params['stream']['pm2'],
    }

    stream_probs = no_off_params['mixture-probs'][-1]
    off_probs = 0.25*stream_probs
    bkg_probs = 1-stream_probs-off_probs
    mm = ComponentMixtureModel(dist.Dirichlet(jnp.array([1.0, 1.0, 1.0])),
                               components=[bkg_model_mm, stream_model_mm, offtrack_model],
                            #    tied_coordinates={"offtrack": {"pm1": "stream", "pm2": "stream"}},
                              )

    mm_init_params = {
        "background": no_off_params["background"],
        "stream": no_off_params['stream'],
        "offtrack": offtrack_init_params,
    }
    mm_packed_params = mm.pack_params(mm_init_params)
    mm_packed_params["mixture-probs"] = jnp.array([bkg_probs, stream_probs, off_probs])
    mm_packed_params["mixture"] = jnp.stack([v for v in run_data.values()], axis=-1)

    keys = jax.random.split(jax.random.PRNGKey(42), num=2)
    full_svi_results, full_guide = run_SVI(mm, mm_packed_params, run_data, run_data_err,
                                           num_steps=20_000, GuideFunction=guide_function, keys=keys,
                                           learning_rate=1e-2, num_particles=3)

    results = {'svi_results': full_svi_results,
               'guide': full_guide,
               'bkg_knot_spacings': bkg_knot_spacings,
               'stream_knot_spacings': stream_knot_spacings,
               'offtrack_dx': offtrack_dx
              }
    
    with open(full_filename, 'wb') as param_file:
        pickle.dump(results, param_file)

if __name__ == '__main__':
    '''
    sys.argv[1] : background knot spacings for phi1, pm1, pm2 (MUST be integers as currently constructed)
    sys.argv[2] : stream knot spacings for phi1, phi2, pm1, pm2 (MUST be integers as currently constructed)
    sys.argv[3] : list of offtrack separations between the nodes in [phi1, phi2]
    '''
    bkg_knot_spacings = jnp.array(json.loads(sys.argv[1]), dtype='int')
    stream_knot_spacings = jnp.array(json.loads(sys.argv[2]), dtype='int')
    offtrack_dx = jnp.array(json.loads(sys.argv[3]))

    ## Create filenames for checking later
    svi_results_dir = '/Users/Tavangar/Work/gd1-dr3/svi_results/'
    bkg_filename = svi_results_dir + 'bkg_{}_{}_{}.pkl'.format(*bkg_knot_spacings)
    stream_filename = svi_results_dir + 'stream_{}_{}_{}_{}.pkl'.format(*stream_knot_spacings)
    
    all_knot_spacings = jnp.concatenate([bkg_knot_spacings, stream_knot_spacings])
    stream_bkg_mm_filename = svi_results_dir + 'mm_bkg{}_{}_{}_stream{}_{}_{}_{}.pkl'.format(*all_knot_spacings)
    
    specifications = jnp.concatenate([bkg_knot_spacings, stream_knot_spacings, offtrack_dx])
    full_filename = svi_results_dir + 'full_mm_bkg{}_{}_{}_stream{}_{}_{}_{}_off{}_{}.pkl'.format(*specifications)
    full_file = Path(full_filename)
    if full_file.exists():
        print('A model with these specifications already exists at ' + full_filename)
        print('If you would like to run this model anyway, please delete or change the nameof the existing file and rerun')

    else:
        main(bkg_knot_spacings, stream_knot_spacings, offtrack_dx, 
             bkg_filename, stream_filename, stream_bkg_mm_filename, full_filename)
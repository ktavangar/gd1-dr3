import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoDiagonalNormal
import optax

from stream_membership import ComponentMixtureModel, ModelComponent
from stream_membership.distributions import IndependentGMM, TruncatedNormalSpline, DirichletSpline, TruncatedNormal1DSplineMixture, TruncatedNormalGMMConditional
from stream_membership.plot import plot_data_projections

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

# Offtrack spatial-term options, forwarded to
# `initialize_stream.make_offtrack_model_component`. Defaults reproduce the
# original model (rectangular lattice, sampled widths, Dirichlet weights).
# New setup: lattice="diamond", fixed_scales=True, weight_prior="softmax"
# (and pass a larger offtrack_dx for the same node count -- see that
# function's docstring).
OFFTRACK_OPTS = dict(
    lattice="rectangular",
    fixed_scales=False,
    sigma_frac=1.0,
    weight_prior="dirichlet",
    logit_scale=2.0,
)


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
    _, _, _, _, pprint = iso_obj_.simpleSln(mass_thresh=0.83)
    
    pmsel = ProperMotionSelection(cat, inputs[stream_name], pprint,
                                  n_dispersion_phi1=3, n_dispersion_phi2=3, cutoff=0.1)

    iso_obj = Isochrone(cat, inputs[stream_name], pawprint=pmsel.pawprint,)
    _, iso_mask, _, hb_mask, pprint = iso_obj.simpleSln(mass_thresh=0.83)

    
    ##### rough pm cut to start with (this comes only from the galstreams proper motion tracks)
    pawprint.pmprint, pm_mask = rough_pm_poly(pawprint, cat, buffer=2)

    return pawprint, iso_obj, iso_mask, hb_mask, pmsel, pm_mask


def run_SVI(model, init_params, data, err=None, 
            num_steps=20_000, GuideFunction=AutoNormal, 
            adaptive_lr=True, init_lr=5e-3, num_particles=3,
            keys=jax.random.split(jax.random.PRNGKey(42), num=2)
            ):

    if adaptive_lr:
        # Define step intervals
        eighth_steps = num_steps // 8
        quarter_steps = num_steps // 4
        half_steps = num_steps // 2

        lr_schedule = optax.join_schedules(
                schedules=[
                    optax.constant_schedule(init_lr),
                    optax.cosine_decay_schedule(init_value=init_lr, decay_steps=quarter_steps+eighth_steps, alpha=1e-2),
                    # optax.constant_schedule(init_lr/100),
                    optax.cosine_decay_schedule(init_value=init_lr/100, decay_steps=quarter_steps+eighth_steps, alpha=1e-2),
                ],
                boundaries=[eighth_steps, 
                            # eighth_steps+quarter_steps,
                            half_steps]
            )
        
        optimizer = optax.chain(optax.clip(10.),
                                optax.adam(learning_rate=lr_schedule)
                                )
    else:
        optimizer = numpyro.optim.ClippedAdam(init_lr)

    # Define the guide
    guide = GuideFunction(
        model, init_loc_fn=numpyro.infer.init_to_value(values=init_params)
    )

    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles))
    with numpyro.validation_enabled(), jax.debug_nans():
        # Initialize and run the SVI optimization
        init_state = svi.init(keys[0], init_params=init_params, data=data, err=err)
        svi_results = svi.run(
            rng_key=keys[1], num_steps=num_steps, init_state=init_state, 
            data=data, err=err
        )
    
    return svi_results, guide


def get_svi_params(model, data, svi_results, guide, num_samples=1, key=jax.random.PRNGKey(12345)):

    pred_dist = Predictive(guide, params=svi_results.params, num_samples=num_samples)
    pars_ = pred_dist(key, data=data)
    # The old single joint site "mixture:modeldata" is now split into
    # multiple per-coordinate sites named f"{coord}:modeldata" (e.g.
    # "pm1:modeldata", "PS_g-PS_r:modeldata") by the ragged-plate
    # ComponentMixtureModel.__call__, so filter by suffix rather than exact
    # match to drop all of them.
    pars = {k: jnp.median(v, axis=0) for k, v in pars_.items() if (k != '_auto_latent' and not k.endswith(':modeldata'))}
    pars_expanded = model.expand_numpyro_params(pars)

    return pars_expanded

def make_lp_dict(model, data, data_err, params):
    
    lp_dict = {} # create dictionary of log probabilities for each component
    
    for n_comp, component in enumerate(model.components):
    
        ## Get distributions for every coordinate in the component
        dists = component.make_dists(params[component.name])
    
        ## Create empty array of log probabilities for each data point and coordinate
        component_lp_stack = jnp.empty((len(dists), len(data['phi1'])))
    
        for i, (coord_name, dist_) in enumerate(dists.items()):
            if isinstance(coord_name, tuple): # (phi1, phi2) offtrack, or the CMD term
                _data = jnp.stack([data[k] for k in coord_name], axis=-1)
            else:
                _data = jnp.asarray(data[coord_name])
            # NOTE: `_errors` is deliberately *not* computed here for every
            # coordinate -- `data_err` (from `table_to_arrays`) only has
            # entries for the ragged pm1/pm2/radial_velocity coordinates.
            # phi1/phi2 (position space) and the CMD magnitudes are treated
            # as noiseless, so they have no `data_err` entry at all; eagerly
            # looking one up for them raised `KeyError`. Each branch below
            # that actually needs an error term looks it up itself.

            if isinstance(dist_, (IndependentGMM, dist.Uniform)):
                # IndependentGMM is only used for position space at the moment so ignoring errors is fine
                variable_lp  = dist_.log_prob(value=_data)

            elif isinstance(dist_, TruncatedNormalGMMConditional):
                # pm1/pm2/radial_velocity for the offtrack component: ragged
                # (not every star has this measurement) and conditioned on
                # this coordinate's own valid-star (phi1, phi2) pairs --
                # exactly the `x` convention `make_offtrack_model_component`
                # built at fit time (see `init_stream._coord_x`/`_coord_x2`),
                # so the full 2D joint responsibility is used here too,
                # rather than falling back to a phi1-only marginal.
                _x = jnp.stack(
                    [init_stream._coord_x(data, coord_name), init_stream._coord_x2(data, coord_name)],
                    axis=-1,
                )
                variable_lp = dist_.log_prob(value=_data, x=_x)

            elif isinstance(dist_, TruncatedNormalSpline): # One truncated spline
                # Get the spline mean and scale at every data point's phi1
                #  the scale should the the rms of the model scale and the error on that point
                # NOTE: not every TruncatedNormalSpline coordinate has a real
                # per-star error in `data_err` -- only the ragged pm1/pm2/
                # radial_velocity measurements do. Other coordinates using
                # this same distribution class (e.g. the stream's "phi2"
                # track) are treated as noiseless, so default to 0 error
                # rather than assuming a `data_err` entry exists.
                _errors = jnp.asarray(data_err.get(coord_name, 0.0))  # Include errors, if any
                # Ragged coordinates (pm1/pm2/radial_velocity) have their own,
                # generally-shorter valid-star phi1 subsample -- `_coord_x`
                # returns that if present, else falls back to `data['phi1']`.
                _phi1 = init_stream._coord_x(data, coord_name)
                spline_mean = dist_._loc_spl(_phi1)
                adjusted_spline_scale = jnp.sqrt(dist_._scale_spl(_phi1)**2 + _errors**2)

                # Get lower and upper bounds of truncated normal (in terms of number of stds from mean)
                lower = (dist_.low - spline_mean) / adjusted_spline_scale
                higher = (dist_.high - spline_mean) / adjusted_spline_scale

                # Create a truncated normal per data point
                variable_lp = jax.scipy.stats.truncnorm.logpdf(
                                    x=_data,
                                    loc=spline_mean,
                                    scale=adjusted_spline_scale,
                                    a=lower,
                                    b=higher
                                )
            elif isinstance(dist_, TruncatedNormal1DSplineMixture): # mixture of truncated splines
                # Similar to above
                #  Create the mean and scale at each data point's 'phi1' for each of the truncated splines (shape (K, len(data)))
                # NOTE: see the matching NOTE in the TruncatedNormalSpline
                # branch above -- default to 0 error for any coordinate that
                # has no real per-star error in `data_err`.
                _errors = jnp.asarray(data_err.get(coord_name, 0.0))  # Include errors, if any
                _phi1 = init_stream._coord_x(data, coord_name)
                spline_mean = jnp.stack([dist_._component_distributions[i]._loc_spl(_phi1) for i in range(dist_._n_components)])
                spline_scale = jnp.stack([dist_._component_distributions[i]._scale_spl(_phi1) for i in range(dist_._n_components)])
                adjusted_spline_scale = jnp.sqrt(spline_scale**2 + _errors**2)
                lower = (dist_.low - spline_mean) / adjusted_spline_scale
                higher = (dist_.high - spline_mean) / adjusted_spline_scale

                component_lp = jax.scipy.stats.truncnorm.logpdf(
                                    x=_data[None, :],
                                    loc=spline_mean,
                                    scale=adjusted_spline_scale,
                                    a=lower,
                                    b=higher
                                )
                # Mixing weights are now a spline function of phi1 (see
                # spline_mixture.py's `mixing_vals` support) instead of one
                # global weight, so `params[...]['mixing_distribution']` no
                # longer exists. Evaluate the per-component mixing logits at
                # each data point's phi1 -- mirroring exactly how
                # spline_mean/spline_scale are evaluated above -- and
                # softmax over components to get properly-normalized,
                # per-datapoint weights (shape (K, len(data))).
                mixing_logits = jnp.stack([spl(_phi1) for spl in dist_._mixing_spls])
                log_mixture_weights = jax.nn.log_softmax(mixing_logits, axis=0)
                weighted_comp_lp = component_lp + log_mixture_weights

                variable_lp = jax.scipy.special.logsumexp(weighted_comp_lp, axis=0)

            else: # shouldn't be used but is here just in case
                variable_lp = dist_.log_prob(value=_data, x=jnp.asarray(data['phi1']))

            # Ragged coordinates (pm1/pm2/radial_velocity) produce a
            # shorter-than-full-length `variable_lp` here -- scatter it back
            # into a full-length array (0 for stars missing this coordinate,
            # so they simply don't contribute this term to the summed total
            # log-posterior below) before stacking. Full-length coordinates
            # (phi1, phi2, and tuple-keyed terms like the offtrack spatial
            # grid or the CMD term) have no valid-mask entry in `data`, so
            # this is a no-op for them.
            mask_key = None if isinstance(coord_name, tuple) else init_stream._valid_mask_key(coord_name)
            if mask_key is not None and mask_key in data:
                variable_lp = init_stream._scatter_ragged_lp(len(data['phi1']), data[mask_key], variable_lp)

            # add log prob from this coordinate to the array for this component log prob
            component_lp_stack = component_lp_stack.at[i].set(variable_lp)
    
            
        # Get total log probability for this component and put into original dictionary
        comp_lp = jnp.sum(component_lp_stack, axis=0) + jnp.log(params['mixture-probs'][n_comp])
        lp_dict[component.name] = comp_lp
        
    return lp_dict

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
        data = at.Table.read("/Users/Tavangar/Work/gd1-dr3/data/GD1-region-alldata.fits")
    
        pawprint,iso_obj,iso_mask,hb_mask,pmsel,pm_mask = run_CATS(data, stream_name='GD-1', phi1_lim=[-100,20])
        run_data_ = iso_obj.cat[pm_mask & (iso_mask | hb_mask)]
        bkg_data_ = iso_obj.cat[pm_mask & (iso_mask | hb_mask) & ~iso_obj.on_skymask]
        stream_data_ = iso_obj.cat[pmsel.pm12_mask & (iso_mask | hb_mask) & iso_obj.on_skymask]

    #####################
    ## Setup the Model ##
    #####################

    def _table_to_ragged_data(tbl_):
        """Ragged, per-coordinate-plate data dict for a CATS-processed
        table. pm1/pm2/rv can be masked columns (missing measurements), so
        -- unlike the old sentinel-fill convention (rv=0, rv_err=1e4) --
        each is subset to its own valid-only rows independently, and each
        ragged coordinate's own valid-star phi1 (and phi2) subsample is
        tagged alongside it via `init_stream._phi1_key`/`_phi2_key` (mirrors
        `svi_helpers.table_to_arrays`). The phi2 tagging is needed for
        `make_offtrack_model_component`'s D=2 (phi1, phi2) grid, whose
        pm1/pm2/radial_velocity conditioning arrays are joint (phi1, phi2)
        pairs, not phi1 alone -- see `init_stream._coord_x2`. Note the
        coordinate is renamed "rv" -> "radial_velocity" here to match the
        key `initialize_stream.make_bkg_model_component`/
        `make_stream_model_component`/`ComponentMixtureModel.__call__`
        actually look for -- under the old shared-plate architecture a
        stray "rv"/"radial_velocity" mismatch just meant the rv term was
        silently absent from `coord_parameters`; under the new ragged-plate
        architecture a missing coordinate is *silently skipped* by design
        (see `ComponentMixtureModel.__call__`), so keeping the mismatch
        here would now silently drop the radial-velocity term from every
        model built by this script instead of merely being inert.

        Also tags each ragged coordinate's full-length valid-star boolean
        mask via `init_stream._valid_mask_key` (mirrors
        `svi_helpers.table_to_arrays`), so downstream per-star log-posterior
        code (`make_lp_dict`) can scatter that coordinate's ragged log-probs
        back into a full-length array aligned with `data_["phi1"]`.
        """
        data_ = {
            "phi1": jnp.array(tbl_["phi1"], dtype="f8"),
            "phi2": jnp.array(tbl_["phi2"], dtype="f8"),
        }
        err_ = {}
        for coord, col, err_col in [
            ("pm1", "pm1", "pm1_error"),
            ("pm2", "pm2", "pm2_error"),
            ("radial_velocity", "rv", "rv_err"),
        ]:
            values = tbl_[col]
            errors = tbl_[err_col]
            valid = jnp.ones(len(tbl_), dtype=bool)
            if hasattr(values, "mask"):
                valid = valid & jnp.array(~values.mask)
            if hasattr(errors, "mask"):
                valid = valid & jnp.array(~errors.mask)
            data_[coord] = jnp.array(values, dtype="f8")[valid]
            err_[coord] = jnp.array(errors, dtype="f8")[valid]
            data_[init_stream._phi1_key(coord)] = data_["phi1"][valid]
            data_[init_stream._phi2_key(coord)] = data_["phi2"][valid]
            # Full-length valid-star mask, needed to scatter this
            # coordinate's ragged per-star values (e.g. log-probs) back into
            # a full-length array in `make_lp_dict`. See
            # `init_stream._valid_mask_key`/`_scatter_ragged_lp`.
            data_[init_stream._valid_mask_key(coord)] = valid
        return data_, err_

    run_data, run_data_err = _table_to_ragged_data(run_data_)
    bkg_data, bkg_data_err = _table_to_ragged_data(bkg_data_)
    stream_data, stream_data_err = _table_to_ragged_data(stream_data_)

    coord_bounds, _ = init_stream.get_bounds_and_grids(run_data, pawprint)
    
    phi1_lim = coord_bounds['phi1']

    guide_function = AutoNormal

    ##########################################
    ## Create and Optimize Background Model ##
    ##########################################
    
    n_mixture = 2 # number of mixture model components in the background proper motion models
        
    bkg_model = init_stream.make_bkg_model_component(coords=['phi2', 'pm1', 'pm2', 'radial_velocity'],
                                                     knot_spacings=bkg_knot_spacings, n_mixture=n_mixture,
                                                     coord_bounds=coord_bounds, data=bkg_data, stream_data=stream_data)
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
        n_rv_knots = bkg_model.coord_parameters['radial_velocity']['knots'].shape[-1]
        
        bkg_init_params = {
            "phi1": {
                "mixing_distribution": jnp.ones(n_phi1_knots) / n_phi1_knots,
                "scales": jnp.full(n_phi1_knots, bkg_knot_spacings[0])
            },
            "phi2": {},
            "pm1": {
                # mixing_vals are per-knot logits (shape (n_mixture, n_knots)),
                # not a probability simplex -- all-zeros is the "uninformative"
                # initialization here, since softmax(0) is a uniform
                # distribution over components at every phi1 knot (equivalent
                # to the old flat jnp.ones(n_mixture) / n_mixture init).
                "mixing_vals": jnp.full((n_mixture, n_pm1_knots), 0.0),
                "loc_vals": jnp.full((n_mixture, n_pm1_knots), 0),
                "scale_vals": jnp.full((n_mixture, n_pm1_knots), 5.0),
            },
            "pm2": {
                "mixing_vals": jnp.full((n_mixture, n_pm2_knots), 0.0),
                "loc_vals": jnp.full((n_mixture, n_pm2_knots), -3.0),
                "scale_vals": jnp.full((n_mixture, n_pm2_knots), 3.0),
            },
            "rv": {
                "mixing_vals": jnp.full((n_mixture, n_rv_knots), 0.0),
                "loc_vals": jnp.full((n_mixture, n_rv_knots), -100),
                "scale_vals": jnp.full((n_mixture,n_rv_knots), 50),
            },
        }
    
        keys = jax.random.split(jax.random.PRNGKey(42), num=2)
        bkg_svi_results, bkg_guide = run_SVI(bkg_model, bkg_init_params, bkg_data, bkg_data_err,
                                             num_steps=2_000, GuideFunction=guide_function, keys=keys,
                                             init_lr=1e-2, adaptive_lr=False, num_particles=3)
        bkg_dict = {'svi_results': bkg_svi_results,
                    'guide': bkg_guide,
                   }
        with open(bkg_filename, 'wb') as param_file:
            pickle.dump(bkg_dict, param_file)
        

    bkg_params = get_svi_params(model=bkg_model, data=bkg_data, svi_results=bkg_svi_results, 
                                guide=bkg_guide, num_samples=1, key=jax.random.PRNGKey(12345))

    ######################################
    ## Create and Optimize Stream Model ##
    ######################################
    
    # `make_stream_model_component` needs the phi1 range spanned by stars
    # with valid radial-velocity measurements to lay down its rv knots --
    # use the ragged radial_velocity coordinate's own phi1 subsample (see
    # `_table_to_ragged_data` above), matching the pattern already used in
    # the notebooks (e.g. `4-svi_runs.ipynb`).
    rv_min = jnp.min(stream_data[init_stream._phi1_key("radial_velocity")])
    rv_max = jnp.max(stream_data[init_stream._phi1_key("radial_velocity")])

    stream_model = init_stream.make_stream_model_component(knot_spacings=stream_knot_spacings,
                                                           coord_bounds=coord_bounds, data=stream_data,
                                                           rv_min=rv_min, rv_max=rv_max)
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
        stream_rv_knots = stream_model.coord_parameters['radial_velocity']['knots']
        
        _interp_dict = init_stream.interpolate_stream_tracks(stream_data, phi1_lim)
        eval_interp_phi2 = jnp.array(_interp_dict['phi2'](stream_phi2_knots))
        eval_interp_pm1 = jnp.array(_interp_dict['pm1'](stream_pm1_knots))
        eval_interp_pm2 = jnp.array(_interp_dict['pm2'](stream_pm2_knots))
        eval_interp_rv = jnp.array(_interp_dict['radial_velocity'](stream_rv_knots))
        
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
            "rv": {
                "loc_vals": eval_interp_rv,
                "scale_vals": jnp.full(stream_rv_knots.shape[0], 4),
            },
        }
    
        keys = jax.random.split(jax.random.PRNGKey(42), num=2)
        stream_svi_results, stream_guide = run_SVI(stream_model, stream_init_params, stream_data, stream_data_err,
                                                   num_steps=50_000, GuideFunction=guide_function, keys=keys,
                                                   init_lr=1e-2, adaptive_lr=False, num_particles=3)
        
        stream_dict = {'svi_results': stream_svi_results,
                       'guide': stream_guide,
                      }
        with open(stream_filename, 'wb') as param_file:
            pickle.dump(stream_dict, param_file)

    stream_params = get_svi_params(model=stream_model, data=stream_data, svi_results=stream_svi_results, 
                                   guide=stream_guide, num_samples=1, key=jax.random.PRNGKey(12345))

    #########################################################
    ## Create and Optimize Background+Stream Mixture Model ##
    #########################################################

    bkg_model_mm = init_stream.make_bkg_model_component(coords=['phi2', 'pm1', 'pm2', 'radial_velocity'],
                                                        knot_spacings=bkg_knot_spacings, n_mixture=n_mixture,
                                                        coord_bounds=coord_bounds, data=run_data, stream_data=stream_data)
    stream_model_mm = init_stream.make_stream_model_component(knot_spacings=stream_knot_spacings,
                                                              coord_bounds=coord_bounds, data=run_data,
                                                              rv_min=rv_min, rv_max=rv_max)
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
    
        filtered_bkg_results = {key: value for key, value in bkg_svi_results.params.items() if not "modeldata" in key}
        filtered_stream_results = {key: value for key, value in stream_svi_results.params.items() if not "modeldata" in key}
        packed_params = filtered_bkg_results | filtered_stream_results
        packed_params["mixture-probs"] = jnp.array([1-f_stream, f_stream])

        # The old single joint "mixture:modeldata_auto_loc"/"_auto_scale"
        # guide-param pair is now one pair PER coordinate (e.g.
        # "pm1:modeldata_auto_loc", "PS_g-PS_r:modeldata_auto_scale", ...),
        # since ComponentMixtureModel.__call__ now samples a per-coordinate
        # ":modeldata" site rather than one shared "mixture:modeldata" site.
        # `previous_svi_results` is loaded from an old, pre-refactor pickle
        # (see `if __name__ == '__main__'` below), so it will only ever have
        # the old-style key(s); this loop is a no-op warm-start for it (the
        # new per-coordinate sites just fall back to the guide's default
        # init) but is written generically so it also works once
        # `previous_svi_results` is regenerated with the new architecture.
        for key_, value_ in previous_svi_results.params.items():
            if key_.endswith("modeldata_auto_loc") or key_.endswith("modeldata_auto_scale"):
                packed_params[key_] = value_

        keys = jax.random.split(jax.random.PRNGKey(42), num=2)
        no_off_svi_results, no_off_guide = run_SVI(stream_bkg_mm, packed_params, run_data, run_data_err,
                                                   num_steps=2_500, GuideFunction=guide_function, keys=keys,
                                                   init_lr=5e-4, adaptive_lr=True, num_particles=3)
        
        no_off_dict = {'svi_results': no_off_svi_results,
                       'guide': no_off_guide,
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
    
    offtrack_model, offtrack_phi12_locs = init_stream.make_offtrack_model_component(
        offtrack_dx, stream_model, coord_bounds, data=run_data,
        **OFFTRACK_OPTS,
    )

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
    offtrack_model.coord_parameters['radial_velocity']['loc_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['radial_velocity']['loc_vals'],
        scale=no_off_params['stream']['radial_velocity']['scale_vals'],
        low=no_off_params['stream']['radial_velocity']['loc_vals']-3*no_off_params['stream']['radial_velocity']['scale_vals'],
        high=no_off_params['stream']['radial_velocity']['loc_vals']+3*no_off_params['stream']['radial_velocity']['scale_vals'])

    offtrack_model.coord_parameters['pm1']['scale_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['pm1']['scale_vals'],
        scale=no_off_params['stream']['pm1']['scale_vals'] / 3,
        low=0, high=2*no_off_params['stream']['pm1']['scale_vals'])
    offtrack_model.coord_parameters['pm2']['scale_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['pm2']['scale_vals'],
        scale=no_off_params['stream']['pm2']['scale_vals'] / 3,
        low=0, high=2*no_off_params['stream']['pm2']['scale_vals'])
    offtrack_model.coord_parameters['radial_velocity']['scale_vals'] = dist.TruncatedNormal(
        loc=no_off_params['stream']['radial_velocity']['scale_vals'],
        scale=no_off_params['stream']['radial_velocity']['scale_vals'] / 3,
        low=0, high=2*no_off_params['stream']['radial_velocity']['scale_vals'])
    # NOTE: no manual 'x' override needed here anymore -- `data=run_data` is
    # now passed directly into `make_offtrack_model_component` above, so
    # pm1/pm2/radial_velocity's joint (phi1, phi2) conditioning arrays are
    # already built correctly (via `_coord_x`/`_coord_x2`) at construction
    # time. A stale override recomputing this manually would in fact be
    # *wrong* now, since offtrack's `conditioning_dist` is D=2 and expects
    # (N, 2) joint (phi1, phi2) pairs, not the old 1D phi1-only array this
    # override used to assign.

    # Uniform initial node weights: probabilities for the Dirichlet prior, or
    # zero log-amplitudes (softmax -> uniform) for the softmax prior.
    if OFFTRACK_OPTS["weight_prior"] == "softmax":
        _init_mix = jnp.zeros(len(offtrack_phi12_locs))
    else:
        _init_mix = jnp.ones(len(offtrack_phi12_locs))
        _init_mix /= _init_mix.sum()

    _spatial_init = {"mixing_distribution": _init_mix}
    if not OFFTRACK_OPTS["fixed_scales"]:
        # (scales are only a sampled site when not fixed)
        _spatial_init["scales"] = jnp.array([offtrack_dx] * len(offtrack_phi12_locs)).T

    offtrack_init_params = {
        ("phi1", "phi2"): _spatial_init,
        "pm1": no_off_params['stream']['pm1'],
        "pm2": no_off_params['stream']['pm2'],
        "radial_velocity": no_off_params['stream']['radial_velocity'],
    }

    stream_probs = no_off_params['mixture-probs'][-1]
    off_probs = 0.25*stream_probs
    bkg_probs = 1-stream_probs-off_probs
    mm = ComponentMixtureModel(dist.Dirichlet(jnp.array([1.0, 1.0, 1.0])),
                               components=[bkg_model_mm, stream_model_mm, offtrack_model],
                            #    tied_coordinates={"offtrack": {"pm1": "stream", "pm2": "stream"}},
                              )

    mm_init_params_off = {
        "offtrack": offtrack_init_params,
    }
    # Filter the dictionary to remove keys starting with "mixture"
    filtered_no_off_svi_results = {key: value for key, value in no_off_svi_results.params.items() if not key.startswith("mixture")}
    mm_packed_params = filtered_no_off_svi_results | mm.pack_params(mm_init_params_off)
    mm_packed_params["mixture-probs"] = jnp.array([bkg_probs, stream_probs, off_probs])

    # Same per-coordinate ":modeldata" site rename as above -- `no_off_svi_results`
    # is a fresh result from this same run (already using the new
    # architecture), so this transplants every per-coordinate modeldata
    # guide param it has (e.g. "pm1:modeldata_auto_loc") into the bigger
    # (bkg+stream+offtrack) mixture's init params.
    for key_, value_ in no_off_svi_results.params.items():
        if key_.endswith("modeldata_auto_loc") or key_.endswith("modeldata_auto_scale"):
            mm_packed_params[key_] = value_

    keys = jax.random.split(jax.random.PRNGKey(42), num=2)
    full_svi_results, full_guide = run_SVI(mm, mm_packed_params, run_data, run_data_err,
                                           num_steps=20_000, GuideFunction=guide_function, keys=keys,
                                           init_lr=1e-5, adaptive_lr=True, num_particles=3)

    results = {'svi_results': full_svi_results,
               'guide': full_guide,
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
    bkg_filename = svi_results_dir + 'bkg_{}_{}_{}_{}.pkl'.format(*bkg_knot_spacings)
    stream_filename = svi_results_dir + 'stream_{}_{}_{}_{}_{}.pkl'.format(*stream_knot_spacings)
    
    all_knot_spacings = jnp.concatenate([bkg_knot_spacings, stream_knot_spacings])
    stream_bkg_mm_filename = svi_results_dir + 'mm_bkg{}_{}_{}_{}_stream{}_{}_{}_{}_{}.pkl'.format(*all_knot_spacings)
    
    specifications = jnp.concatenate([bkg_knot_spacings, stream_knot_spacings, offtrack_dx])
    full_filename = svi_results_dir + 'full_mm_bkg{}_{}_{}_{}_stream{}_{}_{}_{}_{}_off{}_{}.pkl'.format(*specifications)
    full_file = Path(full_filename)
    if full_file.exists():
        print('A model with these specifications already exists at ' + full_filename)
        print('If you would like to run this model anyway, please delete or change the nameof the existing file and rerun')

    else:
        spec = [40,40,40,40,5,20,20,20,20] # picked at random, only using to get modeldata_auto_loc and modeldata_auto_scale

        svi_results_dir = '/Users/Tavangar/Work/gd1-dr3/svi_results/old_init/'
        filename = svi_results_dir + 'mm_bkg{}_{}_{}_{}_stream{}_{}_{}_{}_{}.pkl'.format(*spec)
        
        with open(filename, 'rb') as input_file_:
            full_dict = pickle.load(input_file_)
            
        previous_svi_results = full_dict['svi_results']
        
        main(bkg_knot_spacings, stream_knot_spacings, offtrack_dx, 
             bkg_filename, stream_filename, stream_bkg_mm_filename, full_filename)
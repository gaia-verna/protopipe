#!/usr/bin/env python

import logging
import os
import astropy.units as u
from astropy import table
from astropy.table import QTable, unique
from astropy.io import fits
import argparse
import pandas as pd
import numpy as np
import operator

from pyirf.binning import (
    add_overflow_bins,
)
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.simulations import SimulatedEventsInfo
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)
from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from protopipe.pipeline.utils import load_config

# We would like to add this function in pyirf
def read_protopipe_hdf5(infile, run_header):
    """
    Read a DL2 HDF5 file as produced by the protopipe pipeline:
    https://github.com/cta-observatory/protopipe

    Parameters
    ----------
    infile: str or pathlib.Path
        Path to the input fits file
        
    run_header: dict
        Dictionary with info about simulated particle informations

    Returns
    -------
    events: astropy.QTable
        Astropy Table object containing the reconstructed events information.
    simulated_events: ``~pyirf.simulations.SimulatedEventsInfo``

    """
    log = logging.getLogger("pyirf")
    log.debug(f"Reading {infile}")
    df = pd.read_hdf(infile, "/reco_events")
    
    # These values are hard-coded at the moment
    true_alt = [70] * len(df)
    true_az = [180] * len(df)
    pointing_alt = [70] * len(df)
    pointing_az = [180] * len(df)

    events = QTable([list(df['obs_id']),
                     list(df['event_id']),
                     list(df['xi']) * u.deg, 
                     list(df['mc_energy']) * u.TeV, 
                     list(df['reco_energy']) * u.TeV,
                     list(df['gammaness']),
                     list(df['NTels_reco']),
                     list(df['reco_alt']) * u.deg,
                     list(df['reco_az']) * u.deg,
                     true_alt * u.deg,
                     true_az * u.deg,
                     pointing_alt * u.deg,
                     pointing_az * u.deg,
                    ],
                    names=('obs_id',
                           'event_id',
                           'theta',
                           'true_energy', 
                           'reco_energy', 
                           'gh_score',
                           'multiplicity',
                           'reco_alt',
                           'reco_az',
                           'true_alt',
                           'true_az',
                           'pointing_alt',
                           'pointing_az',   
                          ),
                   )
    
    n_runs = len(set(events['obs_id']))
    log.info(f"Estimated number of runs from obs ids: {n_runs}")

    n_showers = n_runs * run_header["num_use"] * run_header["num_showers"]
    log.debug(f"Number of events from n_runs and run header: {n_showers}")

    sim_info = SimulatedEventsInfo(
        n_showers=n_showers,
        energy_min=u.Quantity(run_header["e_min"], u.TeV),
        energy_max=u.Quantity(run_header["e_max"], u.TeV),
        max_impact=u.Quantity(run_header["gen_radius"], u.m),
        spectral_index=run_header["gen_gamma"],
        viewcone=u.Quantity(run_header["diff_cone"], u.deg),
    )
    return events, sim_info

def main():
    
    log = logging.getLogger("pyirf")
    
    # Read arguments
    parser = argparse.ArgumentParser(description='Make performance files')
    parser.add_argument('--config_file', type=str, required=True, help='')
    parser.add_argument(
        '--obs_time',
        type=str,
        required=True,
        help='Observation time, should be given as a string, value and astropy unit separated by an empty space'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--wave', dest="mode", action='store_const',
                            const="wave", default="tail",
                            help="if set, use wavelet cleaning")
    mode_group.add_argument('--tail', dest="mode", action='store_const',
                            const="tail",
                            help="if set, use tail cleaning, otherwise wavelets")
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Add obs. time in configuration file
    str_obs_time = args.obs_time.split()
    cfg['analysis']['obs_time'] = {'value': float(str_obs_time[0]), 'unit': str(str_obs_time[-1])}

    # Create output directory if necessary
    outdir = os.path.join(cfg['general']['outdir'] + '_pyirf', 'irf_{}_ThSq_{}_Time{:.2f}{}'.format(
        args.mode,
        cfg['analysis']['thsq_opt']['type'],
        cfg['analysis']['obs_time']['value'],
        cfg['analysis']['obs_time']['unit'])
    )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    indir = cfg['general']['indir']
    template_input_file = cfg['general']['template_input_file']
    
    T_OBS = cfg['analysis']['obs_time']['value'] * u.Unit(cfg['analysis']['obs_time']['unit'])
    # scaling between on and off region.
    # Make off region 5 times larger than on region for better
    # background statistics
    ALPHA = cfg['analysis']['alpha']
    # Radius to use for calculating bg rate
    MAX_BG_RADIUS = cfg['analysis']['max_bg_radius'] * u.deg
    
    particles = {
        "gamma": {
            "file": os.path.join(indir, template_input_file.format(args.mode, "gamma")),
            "target_spectrum": CRAB_HEGRA,
            "run_header": cfg['particle_information']['gamma']
        },
        "proton": {
            "file": os.path.join(indir, template_input_file.format(args.mode, "proton")),
            "target_spectrum": IRFDOC_PROTON_SPECTRUM,
            "run_header": cfg['particle_information']['proton']
        },
        "electron": {
            "file": os.path.join(indir, template_input_file.format(args.mode, "electron")),
            "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
            "run_header": cfg['particle_information']['electron']
        },
    }
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)
    
    for k, p in particles.items():
        log.info(f"Simulated {k.title()} Events:")
        p["events"], p["simulation_info"] = read_protopipe_hdf5(p["file"],p["run_header"])

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        # Weight events
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        # on axis observation
        p["events"]["source_fov_offset"] = calculate_source_fov_offset(p["events"])
        # calculate theta / distance between reco and assuemd source positoin
        # we handle only ON observations here, so the assumed source pos
        # is the pointing position
        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=p["events"]["pointing_az"],
            assumed_source_alt=p["events"]["pointing_alt"],
        )
        log.info(p["simulation_info"])
        log.info("")
        
    gammas = particles["gamma"]["events"]
    # background table composed of both electrons and protons
    background = table.vstack(
        [particles["proton"]["events"], particles["electron"]["events"]]
    )

    # Reco energy binning
    cfg_binning_reco = cfg['analysis']['ereco_binning']
    ereco = np.logspace(np.log10(cfg_binning_reco['emin']),
                        np.log10(cfg_binning_reco['emax']),
                        cfg_binning_reco['nbin'] + 1) * u.TeV
    ereco_bins = add_overflow_bins(ereco)
    # True energy binning
    cfg_binning_true = cfg['analysis']['etrue_binning']
    etrue = np.logspace(np.log10(cfg_binning_true['emin']),
                        np.log10(cfg_binning_true['emax']),
                        cfg_binning_true['nbin'] + 1) * u.TeV
    etrue_bins = add_overflow_bins(etrue)
    
    # Handle theta square cut optimisation
    # (compute 68 % containment radius PSF if necessary)
    thsq_opt_type = cfg['analysis']['thsq_opt']['type']
    if thsq_opt_type in 'fixed':
        pass
        #thsq_values = np.array([cfg['analysis']['thsq_opt']['value']]) * u.deg
        #print('Using fixed theta cut: {}'.format(thsq_values))
    elif thsq_opt_type in 'opti':
        pass
        #thsq_values = np.arange(0.05, 0.40, 0.01) * u.deg
        #print('Optimising theta cut for: {}'.format(thsq_values))
    elif thsq_opt_type in 'r68':
        print('Using R68% theta cut')
        print('Computing...')
        theta_cuts = calculate_percentile_cut(
            gammas["theta"],
            gammas["reco_energy"],
            bins=ereco_bins,
            min_value=0.05 * u.deg,
            fill_value=0.32 * u.deg,
            max_value=0.32 * u.deg,
            percentile=68,
        )
    
        gammas["selected_theta"] = evaluate_binned_cut(
            gammas["theta"], 
            gammas["reco_energy"],
            theta_cuts, 
            operator.le
        )

        # Scan eff_bkg efficiency (going from 0.05 to 0.5, 10 bins as in MARS analysis)
        # fixed_bkg_eff = np.linspace(0.05, 0.5, 15)
        # Scan fixed gh_cut values
        # gh_cut_values = np.arange(-1.0, 1.005, 0.05)
        gh_cut_values = np.arange(0, 1.00, 0.005)
        
        # Find best cutoff to reach best sensitivity
        print('- Estimating cutoffs...')
        sensitivity, gh_cuts = optimize_gh_cut(
            gammas[gammas["selected_theta"]],
            background,
            reco_energy_bins=ereco_bins,
            gh_cut_values= gh_cut_values,
            theta_cuts=theta_cuts,
            op=operator.ge,
            alpha=ALPHA,
            background_radius=MAX_BG_RADIUS
        )
        #cut_optimiser.find_best_cutoff(energy_values=ereco, angular_values=thsq_values)

        # evaluate the gh_score cut
        for tab in (gammas, background):
            tab["selected_gh"] = evaluate_binned_cut(
                tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
            )
            
        gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]
          
        # scale relative sensitivity by Crab flux to get the flux sensitivity
        spectrum = particles['gamma']['target_spectrum']

        sensitivity["flux_sensitivity"] = (
                sensitivity["relative_sensitivity"] * spectrum(sensitivity['reco_energy_center'])
            )

        log.info('Calculating IRFs')
        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
            fits.BinTableHDU(theta_cuts, name="THETA_CUTS_R68"),
            fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
        ]

        masks = {
            "": gammas["selected"],
            "_NO_CUTS": slice(None),
            "_ONLY_GH": gammas["selected_gh"],
            "_ONLY_THETA": gammas["selected_theta"],
        }

        fov_offset_bins = [0, 0.5] * u.deg
        source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
        energy_migration_bins = np.geomspace(0.2, 5, 200)

        for label, mask in masks.items():
            effective_area = effective_area_per_energy(
                gammas[mask],
                particles["gamma"]["simulation_info"],
                true_energy_bins=etrue_bins,
            )
            hdus.append(
                create_aeff2d_hdu(
                    effective_area[..., np.newaxis],  # add one dimension for FOV offset
                    etrue_bins,
                    fov_offset_bins,
                    extname="EFFECTIVE_AREA" + label,
                )
            )
            edisp = energy_dispersion(
                gammas[mask],
                true_energy_bins=etrue_bins,
                fov_offset_bins=fov_offset_bins,
                migration_bins=energy_migration_bins,
            )
            hdus.append(
                create_energy_dispersion_hdu(
                    edisp,
                    true_energy_bins=etrue_bins,
                    migration_bins=energy_migration_bins,
                    fov_offset_bins=fov_offset_bins,
                    extname="ENERGY_DISPERSION" + label,
                )
            )

        bias_resolution = energy_bias_resolution(
            gammas[gammas["selected"]], etrue_bins,
        )
        ang_res = angular_resolution(gammas[gammas["selected_gh"]], etrue_bins,)
        psf = psf_table(
            gammas[gammas["selected_gh"]],
            etrue_bins,
            fov_offset_bins=fov_offset_bins,
            source_offset_bins=source_offset_bins,
        )

        background_rate = background_2d(
            background[background['selected_gh']],
            ereco_bins,
            fov_offset_bins=np.arange(0, 11) * u.deg,
            t_obs=T_OBS,
        )

        hdus.append(create_background_2d_hdu(
            background_rate,
            ereco_bins,
            fov_offset_bins=np.arange(0, 11) * u.deg,
        ))
        hdus.append(create_psf_table_hdu(
                psf, etrue_bins, source_offset_bins, fov_offset_bins,
        ))
        hdus.append(create_rad_max_hdu(
            theta_cuts["cut"][:, np.newaxis], ereco_bins, fov_offset_bins
        ))
        hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
        hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

        log.info('Writing outputfile')
        fits.HDUList(hdus).writeto("./pyirf_protopipe.fits.gz", overwrite=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import logging
import os
import astropy.units as u
from astropy import table
from astropy.io import fits
import argparse
import pandas as pd
import numpy as np
import operator

from protopipe.pipeline.utils import load_config
from protopipe.perf.utils import read_DL2_pyirf

from pyirf.binning import (
    add_overflow_bins,
    create_bins_per_decade,
    create_histogram_table,
)
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)

from pyirf.sensitivity import calculate_sensitivity, estimate_background 

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

log = logging.getLogger("pyirf")

def main():
    
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
    outdir = os.path.join(cfg['general']['outdir'], 'irf_{}_Time{:.2f}{}'.format(
        args.mode,
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
    
    # Number of boostrap iterations
    N_Boostrap = cfg['analysis']['n_boostrap']
    
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
    
    for particle_type, p in particles.items():
        log.info(f"Simulated {particle_type.title()} Events:")
        p["events"], p["simulation_info"] = read_DL2_pyirf(p["file"],p["run_header"])
        
        # Multiplicity cut
        p["events"] = p["events"][p["events"]["multiplicity"] >= cfg['analysis']['cut_on_multiplicity']].copy()

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        # Weight events
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        
        for prefix in ('true', 'reco'):
            k = f"{prefix}_source_fov_offset"
            p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)
        
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
    
    MAX_GH_CUT_EFFICIENCY = 0.8
    GH_CUT_EFFICIENCY_STEP = 0.01
    
    # gh cut used for first calculation of the binned theta cuts
    INITIAL_GH_CUT_EFFICENCY = 0.4
    
    INITIAL_GH_CUT = np.quantile(gammas['gh_score'], (1 - INITIAL_GH_CUT_EFFICENCY))
    log.info(f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts")

    # event display uses much finer bins for the theta cut than
    # for the sensitivity
    theta_bins = add_overflow_bins(
        create_bins_per_decade(10 ** (-1.9) * u.TeV, 10 ** 2.3005 * u.TeV, 50,)
    )
    
    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = gammas["gh_score"] >= INITIAL_GH_CUT
    theta_cuts = calculate_percentile_cut(
        gammas["theta"][mask_theta_cuts],
        gammas["reco_energy"][mask_theta_cuts],
        bins=theta_bins,
        min_value=0.05 * u.deg,
        fill_value=0.32 * u.deg,
        max_value=0.32 * u.deg,
        percentile=68,
    )
    
    # same bins as event display uses
    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, bins_per_decade=5
        )
    )

    log.info("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(
        GH_CUT_EFFICIENCY_STEP,
        MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
        GH_CUT_EFFICIENCY_STEP
    )
    sensitivity_step_2, gh_cuts = optimize_gh_cut(
        gammas,
        background,
        reco_energy_bins=sensitivity_bins,
        gh_cut_efficiencies=gh_cut_efficiencies,
        op=operator.ge,
        theta_cuts=theta_cuts,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    
     # now that we have the optimized gh cuts, we recalculate the theta
    # cut as 68 percent containment on the events surviving these cuts.
    log.info('Recalculating theta cut for optimized GH Cuts')
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    theta_cuts_opt = calculate_percentile_cut(
        gammas[gammas['selected_gh']]["theta"],
        gammas[gammas['selected_gh']]["reco_energy"],
        theta_bins,
        percentile=68,
        fill_value=0.32 * u.deg,
        max_value=0.32 * u.deg,
        min_value=0.05 * u.deg,
    )

    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

    # calculate sensitivity
    signal_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    background_hist = estimate_background(
        background[background["selected_gh"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts_opt,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity = calculate_sensitivity(
        signal_hist, background_hist, alpha=ALPHA
    )

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles['gamma']['target_spectrum']
    for s in (sensitivity_step_2, sensitivity):
        s["flux_sensitivity"] = (
            s["relative_sensitivity"] * spectrum(s['reco_energy_center'])
        )
        
    log.info('Calculating IRFs')
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(theta_cuts_opt, name="THETA_CUTS_OPT"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]

    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
        "_ONLY_THETA": gammas["selected_theta"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, 10)
    )
    reco_energy_bins = add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, 5)
    )
    fov_offset_bins = [0, 0.5] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 200)

    for label, mask in masks.items():
        effective_area = effective_area_per_energy(
            gammas[mask],
            particles["gamma"]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one dimension for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
            )
        )
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]], true_energy_bins,
    )
    ang_res = angular_resolution(gammas[gammas["selected_gh"]], true_energy_bins,)
    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )

    background_rate = background_2d(
        background[background['selected_gh']],
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
        t_obs=T_OBS,
    )

    hdus.append(create_background_2d_hdu(
        background_rate,
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
    ))
    hdus.append(create_psf_table_hdu(
        psf, true_energy_bins, source_offset_bins, fov_offset_bins,
    ))
    hdus.append(create_rad_max_hdu(
        theta_cuts_opt["cut"][:, np.newaxis], theta_bins, fov_offset_bins
    ))
    hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
    hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

    log.info('Writing outputfile')
    fits.HDUList(hdus).writeto( outdir + "/pyirf_protopipe.fits.gz", overwrite=True)
    
    if N_Boostrap != 1:
        
        print()
        log.info(f"------- Starting Boostrap {N_Boostrap} iterations ---------") 
        print()
        
        for i in range(N_Boostrap):
            print()
            log.info(f"Starting Boostrap iteration n.{i}")
            print()
            
            # bootstrap gammas table
            gammas_sample = gammas[np.random.choice(len(gammas),len(gammas),replace = True)]
            # bootstrap background table
            background_sample = background[np.random.choice(len(background),len(background),replace = True)]
            
            # theta cut is 68 percent containmente of the gammas
            # for now with a fixed global, unoptimized score cut
            mask_theta_cuts = gammas_sample["gh_score"] >= INITIAL_GH_CUT
            theta_cuts = calculate_percentile_cut(
                gammas_sample["theta"][mask_theta_cuts],
                gammas_sample["reco_energy"][mask_theta_cuts],
                bins=theta_bins,
                min_value=0.05 * u.deg,
                fill_value=0.32 * u.deg,
                max_value=0.32 * u.deg,
                percentile=68,
            )

            sensitivity_step_2, gh_cuts = optimize_gh_cut(
                gammas_sample,
                background_sample,
                reco_energy_bins=sensitivity_bins,
                gh_cut_efficiencies=gh_cut_efficiencies,
                op=operator.ge,
                theta_cuts=theta_cuts,
                alpha=ALPHA,
                background_radius=MAX_BG_RADIUS,
            )

            # now that we have the optimized gh cuts, we recalculate the theta
            # cut as 68 percent containment on the events surviving these cuts.
            log.info('Recalculating theta cut for optimized GH Cuts')
            for tab in (gammas_sample, background_sample):
                tab["selected_gh"] = evaluate_binned_cut(
                    tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
                )

            theta_cuts_opt = calculate_percentile_cut(
                gammas_sample[gammas_sample['selected_gh']]["theta"],
                gammas_sample[gammas_sample['selected_gh']]["reco_energy"],
                theta_bins,
                percentile=68,
                fill_value=0.32 * u.deg,
                max_value=0.32 * u.deg,
                min_value=0.05 * u.deg,
            )

            gammas_sample["selected_theta"] = evaluate_binned_cut(
                gammas_sample["theta"], gammas_sample["reco_energy"], theta_cuts_opt, operator.le
            )
            gammas_sample["selected"] = gammas_sample["selected_theta"] & gammas_sample["selected_gh"]

            # calculate sensitivity
            signal_hist = create_histogram_table(
                gammas_sample[gammas_sample["selected"]], bins=sensitivity_bins
            )
            background_hist = estimate_background(
                background_sample[background_sample["selected_gh"]],
                reco_energy_bins=sensitivity_bins,
                theta_cuts=theta_cuts_opt,
                alpha=ALPHA,
                background_radius=MAX_BG_RADIUS,
            )
            sensitivity = calculate_sensitivity(
                signal_hist, background_hist, alpha=ALPHA
            )

            # scale relative sensitivity by Crab flux to get the flux sensitivity
            spectrum = particles['gamma']['target_spectrum']
            for s in (sensitivity_step_2, sensitivity):
                s["flux_sensitivity"] = (
                    s["relative_sensitivity"] * spectrum(s['reco_energy_center'])
                )
                
            log.info('Calculating IRFs')
    
            hdus = [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
                    fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
                    fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
                    fits.BinTableHDU(theta_cuts_opt, name="THETA_CUTS_OPT"),
                    fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
            ]

            masks = {
                "": gammas_sample["selected"],
                "_NO_CUTS": slice(None),
                "_ONLY_GH": gammas_sample["selected_gh"],
                "_ONLY_THETA": gammas_sample["selected_theta"],
            }

            for label, mask in masks.items():
                effective_area = effective_area_per_energy(
                    gammas_sample[mask],
                    particles["gamma"]["simulation_info"],
                    true_energy_bins=true_energy_bins,
                )
                hdus.append(
                    create_aeff2d_hdu(
                        effective_area[..., np.newaxis],  # add one dimension for FOV offset
                        true_energy_bins,
                        fov_offset_bins,
                        extname="EFFECTIVE_AREA" + label,
                    )
                )
                edisp = energy_dispersion(
                    gammas_sample[mask],
                    true_energy_bins=true_energy_bins,
                    fov_offset_bins=fov_offset_bins,
                    migration_bins=energy_migration_bins,
                )
                hdus.append(
                    create_energy_dispersion_hdu(
                        edisp,
                        true_energy_bins=true_energy_bins,
                        migration_bins=energy_migration_bins,
                        fov_offset_bins=fov_offset_bins,
                        extname="ENERGY_DISPERSION" + label,
                    )
                )

            bias_resolution = energy_bias_resolution(
                gammas_sample[gammas_sample["selected"]], true_energy_bins,
            )
            ang_res = angular_resolution(gammas_sample[gammas_sample["selected_gh"]], true_energy_bins,)
            psf = psf_table(
                gammas_sample[gammas_sample["selected_gh"]],
                true_energy_bins,
                fov_offset_bins=fov_offset_bins,
                source_offset_bins=source_offset_bins,
            )

            background_rate = background_2d(
                background_sample[background_sample['selected_gh']],
                reco_energy_bins,
                fov_offset_bins=np.arange(0, 11) * u.deg,
                t_obs=T_OBS,
            )

            hdus.append(create_background_2d_hdu(
                background_rate,
                reco_energy_bins,
                fov_offset_bins=np.arange(0, 11) * u.deg,
            ))
            hdus.append(create_psf_table_hdu(
                psf, true_energy_bins, source_offset_bins, fov_offset_bins,
            ))
            hdus.append(create_rad_max_hdu(
                theta_cuts_opt["cut"][:, np.newaxis], theta_bins, fov_offset_bins
            ))
            hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
            hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))
            
            log.info('Writing outputfile')
            fits.HDUList(hdus).writeto( outdir + f"/pyirf_protopipe_{i}.fits.gz", overwrite=True)

if __name__ == '__main__':
    main()
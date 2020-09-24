#!/usr/bin/env python
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation
from astropy.io.ascii import Csv

from sys import exit
from glob import glob
import signal
import tables as tb

from ctapipe.utils.CutFlow import CutFlow
from ctapipe.io import event_source
from ctapipe.reco.energy_regressor import *

from protopipe.pipeline import EventPreparer
from protopipe.pipeline.utils import (
    make_argparser,
    prod3b_array,
    str2bool,
    load_config,
    SignalHandler,
)

def main():

    # Argument parser
    parser = make_argparser()

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save images in images.h5 (one file testing)",
    )

    parser.add_argument(
        "--estimate_energy", action="store_true", help="Estimate energy"
    )
    parser.add_argument(
        "--regressor_dir_STD", type=str, default="./", help="regressors directory STD analysis"
    )
    parser.add_argument(
        "--regressor_dir_FIT", type=str, default="./", help="regressors directory FIT analysis"
    )
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    try:  # If the user didn't specify a site and/or and array...
        site = cfg["General"]["site"]
        array = cfg["General"]["array"]
    except KeyError:  # ...raise an error and exit.
        print(
            "\033[91m ERROR: make sure that both 'site' and 'array' are "
            "specified in the analysis configuration file! \033[0m"
        )
        exit()

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    else:
        raise ValueError("don't know which input to use...")

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
    else:
        print("found {} files".format(len(filenamelist)))

    # Get the IDs of the involved telescopes and associated cameras together
    # with the equivalent focal lengths from the first event
    allowed_tels, cams_and_foclens, subarray = prod3b_array(
        filenamelist[0], site, array
    )

    # keeping track of events and where they were rejected
    evt_cutflow = CutFlow("EventCutFlow")
    img_cutflow = CutFlow("ImageCutFlow")

    preper = EventPreparer(
        config=cfg,
        subarray=subarray,
        cams_and_foclens=cams_and_foclens,
        mode=args.mode,
        event_cutflow=evt_cutflow,
        image_cutflow=img_cutflow,
    )

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # Regressor method
    regressor_method = cfg["EnergyRegressor"]["method_name"]

    # wrapper for the scikit-learn regressor
    if args.estimate_energy is True:
        
        regressor_files_STD = (
            args.regressor_dir_STD + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        )
        reg_file_STD = regressor_files_STD.format(
            **{
                "mode": args.mode,
                "wave_args": "mixed",  # ToDo, control
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )

        regressor_STD = EnergyRegressor.load(reg_file_STD, cam_id_list=cams_and_foclens.keys())
        
        regressor_files_FIT = (
            args.regressor_dir_FIT + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        )
        reg_file_FIT = regressor_files_FIT.format(
            **{
                "mode": args.mode,
                "wave_args": "mixed",  # ToDo, control
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )

        regressor_FIT = EnergyRegressor.load(reg_file_FIT, cam_id_list=cams_and_foclens.keys())

    # Declaration of the column descriptor for the (possible) images file
    # For the moment works only on LSTCam and NectarCam.
    class StoredImages(tb.IsDescription):
        event_id = tb.Int32Col(dflt=1, pos=0)
        tel_id = tb.Int16Col(dflt=1, pos=1)
        dl1_phe_image=tb.Float32Col(shape=(1855), pos=2)
        dl1_phe_image_mask_reco=tb.BoolCol(shape=(1855), pos=3)
        mc_phe_image = tb.Float32Col(shape=(1855), pos=4)
        mc_energy = tb.Float32Col(dflt=1, pos=5)
        dl1_phe_image = tb.Float32Col(shape=(1855), pos=2)
        dl1_phe_image_1stPass = tb.Float32Col(shape=(1855), pos=3)
        calibration_status = tb.Int16Col(dflt=1, pos=4)
        mc_phe_image = tb.Float32Col(shape=(1855), pos=5)
        mc_energy = tb.Float32Col(dflt=1, pos=6)

    # Declaration of the column descriptor for the file containing DL1 data
    class EventFeatures(tb.IsDescription):
        impact_dist_STD = tb.Float32Col(dflt=1, pos=0)
        impact_dist_FIT = tb.Float32Col(dflt=1, pos=1)
        
        sum_signal_evt = tb.Float32Col(dflt=1, pos=2)
        max_signal_cam = tb.Float32Col(dflt=1, pos=3)
        sum_signal_cam = tb.Float32Col(dflt=1, pos=4)
        N_LST = tb.Int16Col(dflt=1, pos=5)
        N_MST = tb.Int16Col(dflt=1, pos=6)
        N_SST = tb.Int16Col(dflt=1, pos=7)
        
        N_LST_truncated = tb.Int16Col(dflt=1, pos=8)
        N_MST_truncated = tb.Int16Col(dflt=1, pos=9)
        N_SST_truncated = tb.Int16Col(dflt=1, pos=10)
        
        width = tb.Float32Col(dflt=1, pos=11)
        length = tb.Float32Col(dflt=1, pos=12)
        skewness = tb.Float32Col(dflt=1, pos=13)
        kurtosis = tb.Float32Col(dflt=1, pos=14)
        psi = tb.Float32Col(dflt=1, pos=15)
        
        h_max_STD = tb.Float32Col(dflt=1, pos=16)
        h_max_FIT = tb.Float32Col(dflt=1, pos=17)
        
        err_est_pos = tb.Float32Col(dflt=1, pos=18)
        err_est_dir = tb.Float32Col(dflt=1, pos=19)
        mc_energy = tb.FloatCol(dflt=1, pos=20)
        local_distance = tb.Float32Col(dflt=1, pos=21)
        n_pixel = tb.Int16Col(dflt=1, pos=22)
        n_cluster = tb.Int16Col(dflt=-1, pos=23)
        obs_id = tb.Int16Col(dflt=1, pos=24)
        event_id = tb.Int32Col(dflt=1, pos=25)
        tel_id = tb.Int16Col(dflt=1, pos=26)
        
        xi_STD = tb.Float32Col(dflt=np.nan, pos=27)
        xi_FIT = tb.Float32Col(dflt=np.nan, pos=28)
        
        reco_energy_STD = tb.FloatCol(dflt=np.nan, pos=29)
        reco_energy_FIT = tb.FloatCol(dflt=np.nan, pos=30)
        
        ellipticity = tb.FloatCol(dflt=1, pos=31)
        n_tel_reco = tb.Int16Col(dflt=1, pos=32)
        n_tel_reco_truncated = tb.Int16Col(dflt=1, pos=33)
        
        n_tel_discri = tb.FloatCol(dflt=1, pos=34)
        mc_core_x = tb.FloatCol(dflt=1, pos=35)
        mc_core_y = tb.FloatCol(dflt=1, pos=36)
        
        reco_core_x_STD = tb.FloatCol(dflt=1, pos=37)
        reco_core_y_STD = tb.FloatCol(dflt=1, pos=38)
        reco_core_x_FIT = tb.FloatCol(dflt=1, pos=39)
        reco_core_y_FIT = tb.FloatCol(dflt=1, pos=40)
        
        mc_h_first_int = tb.FloatCol(dflt=1, pos=41)
        
        offset_STD = tb.Float32Col(dflt=np.nan, pos=42)
        offset_FIT = tb.Float32Col(dflt=np.nan, pos=43)
        
        mc_x_max = tb.Float32Col(dflt=np.nan, pos=44)

        alt_STD = tb.Float32Col(dflt=np.nan, pos=45)
        az_STD = tb.Float32Col(dflt=np.nan, pos=46)
        alt_FIT = tb.Float32Col(dflt=np.nan, pos=47)
        az_FIT = tb.Float32Col(dflt=np.nan, pos=48)
        
        reco_energy_tel_STD = tb.Float32Col(dflt=np.nan, pos=49)
        reco_energy_tel_FIT = tb.Float32Col(dflt=np.nan, pos=50)
        
        # from hillas_reco_STD
        ellipticity_reco_STD = tb.FloatCol(dflt=1, pos=51)
        local_distance_reco_STD = tb.Float32Col(dflt=1, pos=52)
        skewness_reco_STD = tb.Float32Col(dflt=1, pos=53)
        kurtosis_reco_STD = tb.Float32Col(dflt=1, pos=54)
        width_reco_STD = tb.Float32Col(dflt=1, pos=55)
        length_reco_STD = tb.Float32Col(dflt=1, pos=56)
        cog_r_reco_STD = tb.Float32Col(dflt=1, pos=57)
        cog_x_reco_STD = tb.Float32Col(dflt=1, pos=58)
        cog_y_reco_STD = tb.Float32Col(dflt=1, pos=59)
        psi_reco_STD = tb.Float32Col(dflt=1, pos=60)
        intensity_STD = tb.Float32Col(dflt=1, pos=61)
        
        # from hillas_reco_FIT
        ellipticity_reco_FIT = tb.FloatCol(dflt=1, pos=62)
        local_distance_reco_FIT = tb.Float32Col(dflt=1, pos=63)
        width_reco_FIT = tb.Float32Col(dflt=1, pos=64)
        length_reco_FIT = tb.Float32Col(dflt=1, pos=65)
        cog_r_reco_FIT = tb.Float32Col(dflt=1, pos=66)
        cog_x_reco_FIT = tb.Float32Col(dflt=1, pos=67)
        cog_y_reco_FIT = tb.Float32Col(dflt=1, pos=68)
        psi_reco_FIT = tb.Float32Col(dflt=1, pos=69)
        intensity_FIT = tb.Float32Col(dflt=1, pos=70)
        
        fval_FIT= tb.Float32Col(dflt=1, pos=71)
        dof_FIT= tb.Float32Col(dflt=1, pos=72)
        invalid_FIT= tb.BoolCol(dflt=False, pos=73)
        
        truncated_image= tb.BoolCol(dflt=False, pos=74)
        pixels_width_1 = tb.Float32Col(dflt=1, pos=75)
        pixels_width_2 = tb.Float32Col(dflt=1, pos=76)
        intensity_width_1 = tb.Float32Col(dflt=1, pos=77)
        intensity_width_2 = tb.Float32Col(dflt=1, pos=78)


    feature_outfile = tb.open_file(args.outfile, mode="w")
    feature_table = {}
    feature_events = {}

    # Create the images file only if the user want to store the images
    if args.save_images is True:
        images_outfile = tb.open_file("images.h5", mode="w")
        images_table = {}
        images_phe = {}

    for i, filename in enumerate(filenamelist):

        print("file: {} filename = {}".format(i, filename))

        source = event_source(
            input_url=filename, allowed_tels=allowed_tels, max_events=args.max_events
        )

        # loop that cleans and parametrises the images and performs the
        # reconstruction for each event
        for(
            event,
            dl1_phe_image,
            dl1_phe_image_mask_reco,
            dl1_phe_image_1stPass,
            calibration_status,
            mc_phe_image,
            n_pixel_dict,
            truncated_image,
            leak_reco,
            hillas_dict,
            hillas_dict_reco_STD,
            hillas_dict_reco_FIT,
            info_fit,
            n_tels,
            n_tels_truncated,
            tot_signal,
            max_signals,
            n_cluster_dict,
            reco_result_STD,
            reco_result_FIT,
            impact_dict_STD,
            impact_dict_FIT,
        ) in preper.prepare_event(source, save_images=args.save_images):

            # Run
            n_run=event.r0.obs_id

            # Angular quantities
            run_array_direction = event.mcheader.run_array_direction

            n_truncated = np.count_nonzero(list(truncated_image.values()))

            xi_STD = angular_separation(
                event.mc.az, event.mc.alt, reco_result_STD.az, reco_result_STD.alt
            )
            xi_FIT = angular_separation(
                event.mc.az, event.mc.alt, reco_result_FIT.az, reco_result_FIT.alt
            )

            offset_STD = angular_separation(
                run_array_direction[0],  # az
                run_array_direction[1],  # alt
                reco_result_STD.az,
                reco_result_STD.alt,
            )
            
            offset_FIT = angular_separation(
                run_array_direction[0],  # az
                run_array_direction[1],  # alt
                reco_result_FIT.az,
                reco_result_FIT.alt,
            )

            # Impact parameter STD
            reco_core_x_STD = reco_result_STD.core_x
            reco_core_y_STD = reco_result_STD.core_y
            
            # Impact parameter FIT
            reco_core_x_FIT = reco_result_FIT.core_x
            reco_core_y_FIT = reco_result_FIT.core_y

            # Height of shower maximum STD
            h_max_STD = reco_result_STD.h_max
            
            # Height of shower maximum STD
            h_max_FIT = reco_result_FIT.h_max
            
            # Todo add conversion in number of radiation length,
            # need an atmosphere profile
            reco_energy_STD = np.nan
            reco_energy_FIT = np.nan

            reco_energy_tel_STD = dict()
            reco_energy_tel_FIT = dict()
            
            
            # Not optimal at all, two loop on tel!!!
            # For energy estimation
            if args.estimate_energy is True:
                weight_tel = np.zeros(len(hillas_dict.keys()))
                energy_tel_STD = np.zeros(len(hillas_dict.keys()))
                energy_tel_FIT = np.zeros(len(hillas_dict.keys()))

                for idx, tel_id in enumerate(hillas_dict.keys()):
                    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                    moments_reco_STD = hillas_dict_reco_STD[tel_id]
                    moments_reco_FIT = hillas_dict_reco_FIT[tel_id]
                    model_STD = regressor_STD.model_dict[cam_id]
                    model_FIT = regressor_FIT.model_dict[cam_id]


                    features_img_STD = np.array(
                        [
                            np.log10(moments.intensity),
                            np.log10(impact_dict_STD[tel_id].value),
                            moments.width.value,
                            moments.length.value,
                            h_max_STD.value,
                            moments_reco_STD.r.value
                        ]
                    )

                    features_img_FIT = np.array(
                        [
                            np.log10(moments.intensity),
                            np.log10(impact_dict_FIT[tel_id].value),
                            moments.width.value,
                            moments.length.value,
                            h_max_FIT.value,
                            moments_reco_FIT.r.value
                        ]
                    )

                    energy_tel_STD[idx] = model_STD.predict([features_img_STD])
                    energy_tel_FIT[idx] = model_FIT.predict([features_img_FIT])
                    
                    weight_tel[idx] = moments.intensity
                    
                    reco_energy_tel_STD[tel_id] = energy_tel_STD[idx]
                    reco_energy_tel_FIT[tel_id] = energy_tel_FIT[idx]
                    

                reco_energy_STD = np.sum(weight_tel * energy_tel_STD) / sum(weight_tel)
                reco_energy_FIT = np.sum(weight_tel * energy_tel_FIT) / sum(weight_tel)
                
            else:
                for idx, tel_id in enumerate(hillas_dict.keys()):
                    reco_energy_tel_STD[tel_id] = np.nan
                    reco_energy_tel_FIT[tel_id] = np.nan
                    
            for idx, tel_id in enumerate(hillas_dict.keys()):
                cam_id = event.inst.subarray.tel[tel_id].camera.cam_id

                if cam_id not in feature_events:
                    feature_table[cam_id] = feature_outfile.create_table(
                        "/", "_".join(["feature_events", cam_id]), EventFeatures
                    )
                    feature_events[cam_id] = feature_table[cam_id].row

                if args.save_images is True:
                    if cam_id not in images_phe:
                        images_table[cam_id] = images_outfile.create_table(
                            "/", "_".join(["images", cam_id]), StoredImages
                        )
                        images_phe[cam_id] = images_table[cam_id].row

                moments = hillas_dict[tel_id]
                moments_reco_STD = hillas_dict_reco_STD[tel_id]
                moments_reco_FIT = hillas_dict_reco_FIT[tel_id]
                
                ellipticity =  moments.width / moments.length
                ellipticity_reco_STD = moments_reco_STD.width / moments_reco_STD.length
                ellipticity_reco_FIT = moments_reco_FIT.width / moments_reco_FIT.length
                
                leak = leak_reco[tel_id]
                # Write to file also the Hillas parameters that have been used
                # to calculate reco_results

                feature_events[cam_id]["impact_dist_STD"] = (
                    impact_dict_STD[tel_id].to("m").value
                )
                feature_events[cam_id]["impact_dist_FIT"] = (
                    impact_dict_FIT[tel_id].to("m").value
                )
                feature_events[cam_id]["sum_signal_evt"] = tot_signal
                feature_events[cam_id]["max_signal_cam"] = max_signals[tel_id]
                feature_events[cam_id]["sum_signal_cam"] = moments.intensity
                feature_events[cam_id]["N_LST"] = n_tels["LST_LST_LSTCam"]
                feature_events[cam_id]["N_MST"] = (
                    n_tels["MST_MST_NectarCam"]
                    + n_tels["MST_MST_FlashCam"]
                    + n_tels["MST_SCT_SCTCam"]
                )
                feature_events[cam_id]["N_SST"] = (
                    n_tels["SST_1M_DigiCam"]
                    + n_tels["SST_ASTRI_ASTRICam"]
                    + n_tels["SST_GCT_CHEC"]
                )
                feature_events[cam_id]["N_LST_truncated"] = n_tels_truncated["LST_LST_LSTCam"]
                feature_events[cam_id]["N_MST_truncated"] = (
                    n_tels_truncated["MST_MST_NectarCam"]
                    + n_tels_truncated["MST_MST_FlashCam"]
                    + n_tels_truncated["MST_SCT_SCTCam"]
                )
                feature_events[cam_id]["N_SST_truncated"] = (
                    n_tels_truncated["SST_1M_DigiCam"]
                    + n_tels_truncated["SST_ASTRI_ASTRICam"]
                    + n_tels_truncated["SST_GCT_CHEC"]
                )
                # Variables from hillas_dict
                feature_events[cam_id]["ellipticity"] = ellipticity.value
                feature_events[cam_id]["width"] = moments.width.to("m").value
                feature_events[cam_id]["length"] = moments.length.to("m").value
                feature_events[cam_id]["psi"] = moments.psi.to("deg").value
                feature_events[cam_id]["skewness"] = moments.skewness
                feature_events[cam_id]["kurtosis"] = moments.kurtosis
                
                feature_events[cam_id]["h_max_STD"] = h_max_STD.to("m").value
                feature_events[cam_id]["h_max_FIT"] = h_max_FIT.to("m").value
                
                feature_events[cam_id]["err_est_pos"] = np.nan
                feature_events[cam_id]["err_est_dir"] = np.nan
                feature_events[cam_id]["mc_energy"] = event.mc.energy.to("TeV").value
                feature_events[cam_id]["local_distance"] = moments.r.to("m").value
                feature_events[cam_id]["n_pixel"] = n_pixel_dict[tel_id]
                feature_events[cam_id]["obs_id"] = event.r0.obs_id
                feature_events[cam_id]["event_id"] = event.r0.event_id
                feature_events[cam_id]["tel_id"] = tel_id
                
                feature_events[cam_id]["xi_STD"] = xi_STD.to("deg").value
                feature_events[cam_id]["xi_FIT"] = xi_FIT.to("deg").value
                
                feature_events[cam_id]["reco_energy_STD"] = reco_energy_STD   
                feature_events[cam_id]["reco_energy_FIT"] = reco_energy_FIT                
                
                feature_events[cam_id]["n_cluster"] = n_cluster_dict[tel_id]
                
                feature_events[cam_id]["n_tel_reco"] = int(n_tels["reco"])
                feature_events[cam_id]["n_tel_reco_truncated"] = int(n_truncated)
                
                feature_events[cam_id]["n_tel_discri"] = n_tels["discri"]
                
                feature_events[cam_id]["mc_core_x"] = event.mc.core_x.to("m").value
                feature_events[cam_id]["mc_core_y"] = event.mc.core_y.to("m").value
                
                feature_events[cam_id]["reco_core_x_STD"] = reco_core_x_STD.to("m").value
                feature_events[cam_id]["reco_core_y_STD"] = reco_core_y_STD.to("m").value
                feature_events[cam_id]["reco_core_x_FIT"] = reco_core_x_FIT.to("m").value
                feature_events[cam_id]["reco_core_y_FIT"] = reco_core_y_FIT.to("m").value
                
                
                feature_events[cam_id]["mc_h_first_int"] = event.mc.h_first_int.to(
                    "m"
                ).value
                feature_events[cam_id]["offset_STD"] = offset_STD.to("deg").value
                feature_events[cam_id]["offset_FIT"] = offset_FIT.to("deg").value
                
                feature_events[cam_id]["mc_x_max"] = event.mc.x_max.value  # g / cm2
                
                feature_events[cam_id]["alt_STD"] = reco_result_STD.alt.to("deg").value
                feature_events[cam_id]["az_STD"] = reco_result_STD.az.to("deg").value
                feature_events[cam_id]["alt_FIT"] = reco_result_FIT.alt.to("deg").value
                feature_events[cam_id]["az_FIT"] = reco_result_FIT.az.to("deg").value
                
                feature_events[cam_id]["reco_energy_tel_STD"] = reco_energy_tel_STD[tel_id]
                feature_events[cam_id]["reco_energy_tel_FIT"] = reco_energy_tel_FIT[tel_id]
                
                # Variables from hillas_dict_reco_STD
                feature_events[cam_id]["ellipticity_reco_STD"] = ellipticity_reco_STD.value
                feature_events[cam_id]["local_distance_reco_STD"] = moments_reco_STD.r.to(
                    "m"
                ).value
                feature_events[cam_id]["skewness_reco_STD"] = moments_reco_STD.skewness
                feature_events[cam_id]["kurtosis_reco_STD"] = moments_reco_STD.kurtosis
                feature_events[cam_id]["width_reco_STD"] = moments_reco_STD.width.to("m").value
                feature_events[cam_id]["length_reco_STD"] = moments_reco_STD.length.to(
                    "m"
                ).value
                feature_events[cam_id]["cog_r_reco_STD"] = moments_reco_STD.r.to("m").value
                feature_events[cam_id]["cog_x_reco_STD"] = moments_reco_STD.x.to("m").value
                feature_events[cam_id]["cog_y_reco_STD"] = moments_reco_STD.y.to("m").value
                feature_events[cam_id]["psi_reco_STD"] = moments_reco_STD.psi.to("deg").value
                feature_events[cam_id]["intensity_STD"] = moments_reco_STD.intensity
                
                # Variables from hillas_dict_reco_FIT
                feature_events[cam_id]["ellipticity_reco_FIT"] = ellipticity_reco_FIT.value
                feature_events[cam_id]["local_distance_reco_FIT"] = moments_reco_FIT.r.to(
                    "m"
                ).value
                feature_events[cam_id]["width_reco_FIT"] = moments_reco_FIT.width.to("m").value
                feature_events[cam_id]["length_reco_FIT"] = moments_reco_FIT.length.to(
                    "m"
                ).value
                feature_events[cam_id]["cog_r_reco_FIT"] = moments_reco_FIT.r.to("m").value
                feature_events[cam_id]["cog_x_reco_FIT"] = moments_reco_FIT.x.to("m").value
                feature_events[cam_id]["cog_y_reco_FIT"] = moments_reco_FIT.y.to("m").value
                feature_events[cam_id]["psi_reco_FIT"] = moments_reco_FIT.psi.to("deg").value
                feature_events[cam_id]["intensity_FIT"] = moments_reco_FIT.intensity
                feature_events[cam_id]["fval_FIT"] = info_fit[tel_id]['fval']
                feature_events[cam_id]["dof_FIT"] = info_fit[tel_id]['dof']
                feature_events[cam_id]["invalid_FIT"] = info_fit[tel_id]['fit_invalid']
                
                feature_events[cam_id]["truncated_image"] = truncated_image[tel_id]
                feature_events[cam_id]["pixels_width_1"] = leak.leakage1_pixel
                feature_events[cam_id]["pixels_width_2"] = leak.leakage2_pixel
                feature_events[cam_id]["intensity_width_1"] = leak.leakage1_intensity
                feature_events[cam_id]["intensity_width_2"] = leak.leakage2_intensity

                feature_events[cam_id].append()

                if args.save_images is True:
                    images_phe[cam_id]["event_id"] = event.r0.event_id
                    images_phe[cam_id]["tel_id"] = tel_id
                    images_phe[cam_id]["dl1_phe_image"] = dl1_phe_image[tel_id] 
                    images_phe[cam_id]["dl1_phe_image_mask_reco"] = dl1_phe_image_mask_reco[tel_id] 
                    images_phe[cam_id]["dl1_phe_image"] = dl1_phe_image[tel_id]
                    images_phe[cam_id]["dl1_phe_image_1stPass"] = dl1_phe_image_1stPass[
                        tel_id
                    ]
                    images_phe[cam_id]["calibration_status"] = calibration_status[
                        tel_id
                    ]
                    images_phe[cam_id]["mc_phe_image"] = mc_phe_image[tel_id]
                    images_phe[cam_id]["mc_energy"] = event.mc.energy.value  # TeV

                    images_phe[cam_id].append()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break
    # make sure that all the events are properly stored
    for table in feature_table.values():
        table.flush()

    if args.save_images is True:
        for table in images_table.values():
            table.flush()

    evt_cutflow()
    evt_table=evt_cutflow.get_table()
    evt_table.remove_column('Efficiency')
    evt_table.add_row(['Run',n_run])
    evt_table.write('EventCut_Table_run'+str(n_run)+'.csv')


    # Catch specific cases
    triggered_events = evt_cutflow.cuts["min2Tels trig"][1]
    reconstructed_events = evt_cutflow.cuts["min2Tels reco"][1]

    if triggered_events == 0:
        print(
            "\033[93mWARNING: No events have been triggered"
            " by the selected telescopes! \033[0m"
        )
    else:
        img_cutflow()
        img_table=img_cutflow.get_table()
        img_table.remove_column('Efficiency')
        img_table.add_row(['Run',n_run]) 
        img_table.write('ImageCut_Table_run'+str(n_run)+'.csv')

        if reconstructed_events == 0:
            print(
                "\033[93m WARNING: None of the triggered events have been "
                "properly reconstructed by the selected telescopes!\n"
                "DL1 file will be empty! \033[0m"
            )

if __name__ == "__main__":
    main()

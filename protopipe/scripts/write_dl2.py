#!/usr/bin/env python

from sys import exit
import numpy as np
from glob import glob
import signal
from astropy.coordinates.angle_utilities import angular_separation
import tables as tb

# ctapipe
# from ctapipe.io import EventSourceFactory
from ctapipe.io import event_source
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.reco.energy_regressor import EnergyRegressor
from ctapipe.reco.event_classifier import EventClassifier

# Utilities
from protopipe.pipeline import EventPreparer
from protopipe.pipeline.utils import (
    make_argparser,
    prod3b_array,
    str2bool,
    load_config,
    SignalHandler,
)

# from memory_profiler import profile

# @profile
def main():

    # Argument parser
    parser = make_argparser()
    
    parser.add_argument("--regressor_dir_STD", default="./", help="regressors directory STD analysys")
    parser.add_argument("--regressor_dir_STDFIT", default="./", help="regressors directory STDFIT analysys")
     
    parser.add_argument("--classifier_dir_STD", default="./", help="regressors directory STD analysys")
    parser.add_argument("--classifier_dir_STDFIT", default="./", help="regressors directory STDFIT analysys")
    
    
    parser.add_argument(
        "--force_tailcut_for_extended_cleaning",
        type=str2bool,
        default=False,
        help="For tailcut cleaning for energy/score estimation",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save images in images.h5 (one file testing)",
    )
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Read site layout
    site = cfg["General"]["site"]
    array = cfg["General"]["array"]

    # Add force_tailcut_for_extended_cleaning in configuration
    cfg["General"][
        "force_tailcut_for_extended_cleaning"
    ] = args.force_tailcut_for_extended_cleaning
    cfg["General"]["force_mode"] = "tail"
    force_mode = args.mode
    if cfg["General"]["force_tailcut_for_extended_cleaning"] is True:
        force_mode = "tail"
    print("force_mode={}".format(force_mode))
    print("mode={}".format(args.mode))

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
    
    # Get the IDs of the involved telescopes and associated cameras together
    # with the equivalent focal lengths from the first event
    
    allowed_tels, cams_and_foclens, subarray = prod3b_array(filenamelist[0], site, array)
    
    # keeping track of events and where they were rejected
    evt_cutflow = CutFlow("EventCutFlow")
    img_cutflow = CutFlow("ImageCutFlow")
    
    # Event preparer
    #preper = EventPreparer(
    #   config=cfg,
    #   cams_and_foclens=cams_and_foclens,
    #   mode=args.mode,
    #   event_cutflow=evt_cutflow,
    #   image_cutflow=img_cutflow,
    #)

    preper = EventPreparer(
        config=cfg,
        subarray=subarray,
        cams_and_foclens=cams_and_foclens,
        mode=args.mode,
        event_cutflow=evt_cutflow,
        image_cutflow=img_cutflow,
      )
    
    # Regressor and classifier methods
    regressor_method = cfg["EnergyRegressor"]["method_name"]
    classifier_method = cfg["GammaHadronClassifier"]["method_name"]
    use_proba_for_classifier = cfg["GammaHadronClassifier"]["use_proba"]

    if regressor_method in ["None", "none", None]:
        use_regressor = False
    else:
        use_regressor = True

    if classifier_method in ["None", "none", None]:
        use_classifier = False
    else:
        use_classifier = True

    # Classifiers
    if use_classifier:
        classifier_files_STD = (
            args.classifier_dir_STD + "/classifier_{mode}_{cam_id}_{classifier}.pkl.gz"
        )
        
        classifier_files_STDFIT = (
            args.classifier_dir_STDFIT + "/classifier_{mode}_{cam_id}_{classifier}.pkl.gz"
        )
        clf_file_STD = classifier_files_STD.format(
            **{
                "mode": force_mode,
                "wave_args": "mixed",
                "classifier": classifier_method,
                "cam_id": "{cam_id}",
            }
        )
        
        clf_file_STDFIT = classifier_files_STDFIT.format(
            **{
                "mode": force_mode,
                "wave_args": "mixed",
                "classifier": classifier_method,
                "cam_id": "{cam_id}",
            }
        )
        classifier_STD = EventClassifier.load(clf_file_STD, cam_id_list=cams_and_foclens.keys())
        classifier_STDFIT = EventClassifier.load(clf_file_STDFIT, cam_id_list=cams_and_foclens.keys())
        
    # Regressors
    if use_regressor:
        regressor_files_STD = (
            args.regressor_dir_STD + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        )
        regressor_files_STDFIT = (
            args.regressor_dir_STDFIT + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        )
        
        reg_file_STD = regressor_files_STD.format(
            **{
                "mode": force_mode,
                "wave_args": "mixed",
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )
        reg_file_STDFIT = regressor_files_STDFIT.format(
            **{
                "mode": force_mode,
                "wave_args": "mixed",
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )
        regressor_STD = EnergyRegressor.load(reg_file_STD, cam_id_list=cams_and_foclens.keys())
        regressor_STDFIT = EnergyRegressor.load(reg_file_STDFIT, cam_id_list=cams_and_foclens.keys())

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # Declaration of the column descriptor for the (possible) images file
    class StoredImages(tb.IsDescription):
        event_id = tb.Int32Col(dflt=1, pos=0)
        tel_id = tb.Int16Col(dflt=1, pos=1)
        dl1_phe_image = tb.Float32Col(shape=(1855), pos=2)
        mc_phe_image = tb.Float32Col(shape=(1855), pos=3)

    # this class defines the reconstruction parameters to keep track of
    class RecoEvent(tb.IsDescription):
        obs_id = tb.Int16Col(dflt=-1, pos=0)
        event_id = tb.Int32Col(dflt=-1, pos=1)
        NTels_trig = tb.Int16Col(dflt=0, pos=2)
        
        NTels_reco = tb.Int16Col(dflt=0, pos=3)
        NTels_reco_lst = tb.Int16Col(dflt=0, pos=4)
        NTels_reco_mst = tb.Int16Col(dflt=0, pos=5)
        NTels_reco_sst = tb.Int16Col(dflt=0, pos=6)
        
        NTels_reco_truncated = tb.Int16Col(dflt=0, pos=7)
        NTels_reco_lst_truncated = tb.Int16Col(dflt=0, pos=8)
        NTels_reco_mst_truncated = tb.Int16Col(dflt=0, pos=9)
        NTels_reco_sst_truncated = tb.Int16Col(dflt=0, pos=10)      
        
        mc_energy = tb.Float32Col(dflt=np.nan, pos=11)
        
        reco_energy_STD = tb.Float32Col(dflt=np.nan, pos=12)
        reco_energy_STDFIT = tb.Float32Col(dflt=np.nan, pos=13)
        
        reco_alt_STD = tb.Float32Col(dflt=np.nan, pos=14)
        reco_alt_STDFIT = tb.Float32Col(dflt=np.nan, pos=15)
        
        reco_az_STD = tb.Float32Col(dflt=np.nan, pos=16)
        reco_az_STDFIT = tb.Float32Col(dflt=np.nan, pos=17)

        offset_STD = tb.Float32Col(dflt=np.nan, pos=18)
        offset_STDFIT = tb.Float32Col(dflt=np.nan, pos=19)
        
        xi_STD = tb.Float32Col(dflt=np.nan, pos=20)
        xi_STDFIT = tb.Float32Col(dflt=np.nan, pos=21)
        
        ErrEstPos = tb.Float32Col(dflt=np.nan, pos=22)
        ErrEstDir = tb.Float32Col(dflt=np.nan, pos=23)
        
        gammaness_STD = tb.Float32Col(dflt=np.nan, pos=24)
        gammaness_STDFIT = tb.Float32Col(dflt=np.nan, pos=25)
        
        success = tb.BoolCol(dflt=False, pos=26)
        
        score_STD = tb.Float32Col(dflt=np.nan, pos=27)
        score_STDFIT = tb.Float32Col(dflt=np.nan, pos=28)
        
        h_max_STD = tb.Float32Col(dflt=np.nan, pos=29)
        h_max_STDFIT = tb.Float32Col(dflt=np.nan, pos=30)
        
        reco_core_x_STD = tb.Float32Col(dflt=np.nan, pos=31)
        reco_core_x_STDFIT = tb.Float32Col(dflt=np.nan, pos=32)
        
        reco_core_y_STD = tb.Float32Col(dflt=np.nan, pos=33)
        reco_core_y_STDFIT = tb.Float32Col(dflt=np.nan, pos=34)
        
        mc_core_x = tb.Float32Col(dflt=np.nan, pos=35)
        mc_core_y = tb.Float32Col(dflt=np.nan, pos=36)

    reco_outfile = tb.open_file(
        mode="w",
        # if no outfile name is given (i.e. don't to write the event list to disk),
        # need specify two "driver" arguments
        **(
            {"filename": args.outfile}
            if args.outfile
            else {
                "filename": "no_outfile.h5",
                "driver": "H5FD_CORE",
                "driver_core_backing_store": False,
            }
        )
    )

    reco_table = reco_outfile.create_table("/", "reco_events", RecoEvent)
    reco_event = reco_table.row

    # Create the images file only if the user want to store the images
    if args.save_images is True:
        images_outfile = tb.open_file("images.h5", mode="w")
        images_table = {}
        images_phe = {}

    for i, filename in enumerate(filenamelist):

        source = event_source(
            input_url=filename, allowed_tels=allowed_tels, max_events=args.max_events
        )
        # loop that cleans and parametrises the images and performs the reconstruction
        for (
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
            hillas_dict_reco_STDFIT,
            info_fit,
            n_tels,
            n_tels_truncated,
            tot_signal,
            max_signals,
            n_cluster_dict,
            reco_result_STD,
            reco_result_STDFIT,
            impact_dict_STD,
            impact_dict_STDFIT,
        ) in preper.prepare_event(source):
            
            # Run
            n_run=event.r0.obs_id
            
            # N Truncated
            n_truncated = np.count_nonzero(list(truncated_image.values()))
            
            # Angular quantities
            run_array_direction = event.mcheader.run_array_direction

            # Angular separation between true and reco direction
            xi_STD = angular_separation(
                event.mc.az, event.mc.alt, reco_result_STD.az, reco_result_STD.alt
            )
            xi_STDFIT = angular_separation(
                event.mc.az, event.mc.alt, reco_result_STDFIT.az, reco_result_STDFIT.alt
            )

            # Angular separation bewteen the center of the camera and the reco direction.
            offset_STD = angular_separation(
                run_array_direction[0],  # az
                run_array_direction[1],  # alt
                reco_result_STD.az,
                reco_result_STD.alt,
            )
            
            offset_STDFIT = angular_separation(
                run_array_direction[0],  # az
                run_array_direction[1],  # alt
                reco_result_STDFIT.az,
                reco_result_STDFIT.alt,
            )

            # Height of shower maximum
            h_max_STD = reco_result_STD.h_max
            h_max_STDFIT = reco_result_STDFIT.h_max
            

            if hillas_dict is not None:

                # Estimate particle energy
                if use_regressor is True:
                    energy_tel_STD = np.zeros(len(hillas_dict.keys()))
                    energy_tel_STDFIT = np.zeros(len(hillas_dict.keys()))
                    
                    weight_tel = np.zeros(len(hillas_dict.keys()))

                    for idx, tel_id in enumerate(hillas_dict.keys()):
                        cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                        moments = hillas_dict[tel_id]
                        moments_reco = hillas_dict_reco[tel_id]
                        model_STD = regressor_STD.model_dict[cam_id]
                        model_STDFIT = regressor_STDFIT.model_dict[cam_id]       

                        # Features to be fed in the regressor
                        features_img_STD = np.array(
                            [
                                np.log10(moments.intensity),
                                np.log10(impact_dict_STD[tel_id].value),
                                moments.width.value,
                                moments.length.value,
                                h_max_STD.value,
                                moments_reco.local_distance_reco_STD.value
                            ]
                        )
                        features_img_STDFIT = np.array(
                            [
                                np.log10(moments.intensity),
                                np.log10(impact_dict_STDFIT[tel_id].value),
                                moments.width.value,
                                moments.length.value,
                                h_max_STDFIT.value,
                                moments_reco.local_distance_reco_STDFIT.value
                            ]
                        )

                        energy_tel_STD[idx] = model_STD.predict([features_img_STD])
                        energy_tel_STDFIT[idx] = model_STDFIT.predict([features_img_STDFIT])
                        
                        weight_tel[idx] = moments.intensity

                    reco_energy_STD = np.sum(weight_tel * energy_tel_STD) / sum(weight_tel)
                    reco_energy_STDFIT = np.sum(weight_tel * energy_tel_STDFIT) / sum(weight_tel)
                    
                else:
                    reco_energy_STD = np.nan
                    reco_energy_STDFIT = np.nan
                    
                # Estimate particle score/gammaness
                if use_classifier is True:
                    score_tel_STD = np.zeros(len(hillas_dict.keys()))
                    gammaness_tel_STD = np.zeros(len(hillas_dict.keys()))
                    score_tel_STDFIT = np.zeros(len(hillas_dict.keys()))
                    gammaness_tel_STDFIT = np.zeros(len(hillas_dict.keys()))
                    
                    weight_tel = np.zeros(len(hillas_dict.keys()))

                    for idx, tel_id in enumerate(hillas_dict.keys()):
                        cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                        moments = hillas_dict[tel_id]
                        model_STD = classifier_STD.model_dict[cam_id]
                        model_STDFIT = classifier_STD.model_dict[cam_id]
                        
                        # Features to be fed in the classifier
                        features_img_STD = np.array(
                            [
                                np.log10(reco_energy_STD),
                                moments.width.value,
                                moments.length.value,
                                moments.skewness,
                                moments.kurtosis,
                                h_max_STD.value,
                            ]
                        )
                        
                        features_img_STDFIT = np.array(
                            [
                                np.log10(reco_energy_STDFIT),
                                moments.width.value,
                                moments.length.value,
                                moments.skewness,
                                moments.kurtosis,
                                h_max_STDFIT.value,
                            ]
                        )
                        
                        # Output of classifier according to type of classifier
                        if use_proba_for_classifier is False:
                            score_tel_STD[idx] = model_STD.decision_function([features_img_STD])
                            score_tel_STDFIT[idx] = model_STDFIT.decision_function([features_img_STDFIT])
                        else:
                            gammaness_tel_STD[idx] = model_STD.predict_proba([features_img_STD])[
                                :, 1
                            ]
                            gammaness_tel_STDFIT[idx] = model_STDFIT.predict_proba([features_img_STDFIT])[
                                :, 1
                            ]
                        # Should test other weighting strategy (e.g. power of charge, impact, etc.)
                        # For now, weighting a la Mars
                        weight_tel[idx] = np.sqrt(moments.intensity)

                    # Weight the final decision/proba
                    if use_proba_for_classifier is True:
                        gammaness_STD = np.sum(weight_tel * gammaness_tel_STD) / sum(weight_tel)
                        gammaness_STDFIT = np.sum(weight_tel * gammaness_tel_STDFIT) / sum(weight_tel)
                    else:
                        score_STD = np.sum(weight_tel * score_tel_STD) / sum(weight_tel)
                        score_STDFIT = np.sum(weight_tel * score_telSTDFIT) / sum(weight_tel)
                else:
                    score_STD = np.nan
                    score_STDFIT = np.nan
                    
                    gammaness_STD = np.nan
                    gammaness_STDFIT = np.nan
                    

                # Regardless if energy or gammaness is estimated, if the user
                # wants to save the images of the run we do it here
                # (Probably not the most efficient way, but for one file is ok)
                if args.save_images is True:
                    for idx, tel_id in enumerate(hillas_dict.keys()):
                        cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                        if cam_id not in images_phe:
                            images_table[cam_id] = images_outfile.create_table(
                                "/", "_".join(["images", cam_id]), StoredImages
                            )
                            images_phe[cam_id] = images_table[cam_id].row

                shower = event.mc
                mc_core_x = shower.core_x
                mc_core_y = shower.core_y

                
                # Impact parameter STD
                reco_core_x_STD = reco_result_STD.core_x
                reco_core_y_STD = reco_result_STD.core_y

                # Impact parameter STDFIT
                reco_core_x_STDFIT = reco_result_STDFIT.core_x
                reco_core_y_STDFIT = reco_result_STDFIT.core_y

                alt_STD, az_STD = reco_result_STD.alt, reco_result_STD.az
                alt_STDFIT, az_STDFIT = reco_result_STDFIT.alt, reco_result_STDFIT.az
                

                # Fill table's attributes
                reco_event["NTels_trig"] = len(event.dl0.tels_with_data)
                reco_event["NTels_reco"] = len(hillas_dict)
                reco_event["NTels_reco_lst"] = n_tels["LST_LST_LSTCam"]
                reco_event["NTels_reco_mst"] = (
                    n_tels["MST_MST_NectarCam"]
                    + n_tels["MST_MST_FlashCam"]
                    + n_tels["MST_SCT_SCTCam"]
                )
                reco_event["NTels_reco_sst"] = (
                    n_tels["SST_1M_DigiCam"]
                    + n_tels["SST_ASTRI_ASTRICam"]
                    + n_tels["SST_GCT_CHEC"]
                )
                reco_event["NTels_reco"] = len(hillas_dict)
                                           
                reco_event["NTels_reco_truncated"] = n_truncated
                reco_event["NTels_reco_lst_truncated"] = n_tels_truncated["LST_LST_LSTCam"]
                reco_event["NTels_reco_mst_truncated"] = (
                    n_tels_truncated["MST_MST_NectarCam"]
                    + n_tels_truncated["MST_MST_FlashCam"]
                    + n_tels_truncated["MST_SCT_SCTCam"]
                )
                reco_event["NTels_reco_sst_truncated"] = (
                    n_tels_truncated["SST_1M_DigiCam"]
                    + n_tels_truncated["SST_ASTRI_ASTRICam"]
                    + n_tels_truncated["SST_GCT_CHEC"]
                )
                                        
                
                reco_event["reco_energy_STD"] = reco_energy_STD
                reco_event["reco_energy_STDFIT"] = reco_energy_STDFIT
                
                reco_event["reco_alt_STD"] = alt_STD.to("deg").value
                reco_event["reco_alt_STDFIT"] = alt_STDFIT.to("deg").value
                
                reco_event["reco_az_STD"] = az_STD.to("deg").value
                reco_event["reco_az_STDFIT"] = az_STDFIT.to("deg").value
                
                reco_event["offset_STD"] = offset_STD.to("deg").value
                reco_event["offset_STDFIT"] = offset_STDFIT.to("deg").value
                
                reco_event["xi_STD"] = xi_STD.to("deg").value
                reco_event["xi_STDFIT"] = xi_STDFIT.to("deg").value
                
                reco_event["h_max_STD"] = h_max_STD.to("m").value
                reco_event["h_max_STDFIT"] = h_max_STDFIT.to("m").value
                
                reco_event["reco_core_x_STD"] = reco_core_x_STD.to("m").value
                reco_event["reco_core_x_STDFIT"] = reco_core_x_STDFIT.to("m").value
                
                reco_event["reco_core_y_STD"] = reco_core_y_STD.to("m").value
                reco_event["reco_core_y_STDFIT"] = reco_core_y_STDFIT.to("m").value
                
                reco_event["mc_core_x"] = mc_core_x.to("m").value
                reco_event["mc_core_y"] = mc_core_y.to("m").value
                if use_proba_for_classifier is True:
                    reco_event["gammaness_STD"] = gammaness_STD
                    reco_event["gammaness_STDFIT"] = gammaness_STDFIT
                else:
                    reco_event["score_STD"] = score_STD
                    reco_event["score_STDFIT"] = score_STDFIT
                    
                reco_event["success"] = True
                
                reco_event["ErrEstPos"] = np.nan
                reco_event["ErrEstDir"] = np.nan
            else:
                reco_event["success"] = False
                

            # save basic event infos
            reco_event["mc_energy"] = event.mc.energy.to("TeV").value
            reco_event["event_id"] = event.r1.event_id
            reco_event["obs_id"] = event.r1.obs_id

            if args.save_images is True:
                images_phe[cam_id]["event_id"] = event.r0.event_id
                images_phe[cam_id]["tel_id"] = tel_id
                images_phe[cam_id]["dl1_phe_image"] = dl1_phe_image
                images_phe[cam_id]["mc_phe_image"] = mc_phe_image

                images_phe[cam_id].append()

            # Fill table
            reco_table.flush()
            reco_event.append()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break

    # make sure everything gets written out nicely
    reco_table.flush()

    if args.save_images is True:
        for table in images_table.values():
            table.flush()

    # Add in meta-data's table?
    try:
        print()
        evt_cutflow()
        print()
        img_cutflow()

        evt_table=evt_cutflow.get_table()
        evt_table.remove_column('Efficiency')
        evt_table.add_row(['Run',n_run])
        evt_table.write('EventCut_Table_run'+str(n_run)+'.csv')
        
        img_table=img_cutflow.get_table()
        img_table.remove_column('Efficiency')
        img_table.add_row(['Run',n_run]) 
        img_table.write('ImageCut_Table_run'+str(n_run)+'.csv')

    except ZeroDivisionError:
        pass

    print("Job done!")


if __name__ == "__main__":
    main()

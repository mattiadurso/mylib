"""
Code copied from run_gtsfm notebook.
"""



import os
# append parent pat
import glob
import pandas as pd
# turn off user warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

scene_path = '/media/mattia/HDD/Datasets/ETH3D_Stereo/DSLR_undistorted_gtsfm/botanical_garden'


os.system(f"python gtsfm/runner/run_scene_optimizer_colmaploader.py \
                --mvs_off \
                --config unified \
                --correspondence_generator_config_name sift \
                --share_intrinsics \
                --images_dir {scene_path}/images \
                --colmap_files_dirpath {scene_path}/dslr_calibration_undistorted \
                --num_matched 5 \
                --max_frame_lookahead 10 \
                --num_workers 1 \
                --worker_memory_limit 40GB \
                --output_root {scene_path}/gtsfm\
                --max_resolution 760 \
                ")


# scenes = glob.glob("/media/mattia/HDD/Datasets/COLMAP/*/undistorted_images")
scenes = glob.glob("/media/mattia/HDD/Datasets/ETH3D_Stereo/DSLR_undistorted_gtsfm/*/")

for scene_path in scenes:
    # then move files from {scene_path}/gtsfm/results/ba_output to {scene_path}/gtsfm/sparse/0

    path_from = f"{scene_path}/gtsfm/results/ba_output/*"
    path_to =   f"{scene_path}/gtsfm/sparse/0"

    os.makedirs(path_to, exist_ok=True)
    os.system(f"cp -r {path_from} {path_to}")




### TIme taken computation

    # read json file as dict
import json
# import PAth
from pathlib import Path


duration_gtsfm = {}

for scene_path in glob.glob("/media/mattia/HDD/Datasets/Sony/*/"):
    temp_duration = {}
    
    for i in range(1):
        model_name = f"gtsfm"

        # rotation averaging
        metrics_path = Path(scene_path, model_name, "result_metrics/rotation_averaging_metrics.json")
        with open(metrics_path) as f:
            data = json.load(f)
            rot_duration = data["rotation_averaging_metrics"]["total_duration_sec"] 
            

        # translation averaging
        metrics_path = Path(scene_path, model_name, "result_metrics/translation_averaging_metrics.json")
        with open(metrics_path) as f:
            data = json.load(f)
            tras_duration = data["translation_averaging_metrics"]["total_duration_sec"] + data["translation_averaging_metrics"]["outlier_rejection_duration_sec"] + data["translation_averaging_metrics"]["optimization_duration_sec"]

        # data association
        metrics_path = Path(scene_path, model_name, "result_metrics/data_association_metrics.json")
        with open(metrics_path) as f:
            data = json.load(f)
            data_ass_duration = data["data_association_metrics"]["total_duration_sec"]

        # BA
        metrics_path = Path(scene_path, model_name, "result_metrics/bundle_adjustment_metrics.json")
        with open(metrics_path) as f:
            data = json.load(f)
            ba_duration = data["bundle_adjustment_metrics"]["total_run_duration_sec"]

        # sum
        total_duration = rot_duration + tras_duration + data_ass_duration + ba_duration

        # storing
        temp_duration[i] = total_duration

    duration_gtsfm[scene_path.split("/")[-2]] = temp_duration


# to pandas
duration_gtsfm_df = pd.DataFrame(duration_gtsfm).T

duration_gtsfm_df["Mapping"] = duration_gtsfm_df.mean(axis=1) # mean
duration_gtsfm_df["std"] = duration_gtsfm_df.std(axis=1)

duration_gtsfm_df
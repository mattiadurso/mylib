from __future__ import annotations

import os
import sys

import os
import sys

COLMAP_PATH = os.environ.get(
    "COLMAP_PATH", os.path.expanduser("~/Desktop/Repos/colmap")
)
sys.path.append(COLMAP_PATH)
from scripts.python.read_write_model import read_model, write_model, qvec2rotmat
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from tqdm.auto import tqdm
from pathlib import Path
from itertools import combinations
from ..metrics import evaluate_R_t, compute_AUC_pxsfm


#  -----------------------------
#  ----------- Utils -----------
#  -----------------------------


def print_path_to_colmap():
    """
    Print the path to the COLMAP repository.
    """
    return os.path.expanduser("~") + "/Desktop/Repos/colmap"


def read_colmap_model(model_path):
    """
    Read a camera, images, and points3D files from a COLMAP model.
    Args:
        model_path: path to the folder conatining camera.bin, images.bin, and points3D.bin in COLMAP format.
    Returns:
        cameras:  dict with lines in camera.bin/txt file.
        images:   dict with lines in images.bin/txt file.
        points3D: dict with lines in points3D.bin/txt file.
    """
    # check if the paths exist with .txt
    if os.path.exists(Path(model_path, "cameras.txt")):
        ext = ".txt"
    else:
        ext = ".bin"

    cameras, images, points = read_model(path=model_path, ext=ext)
    return cameras, images, points


def convert_bin2text(scene_path, mapper="colmap", model="sparse/0"):
    """
    Function to essily convert the .bin files to txt files in a COLMAP model.
    Args:
        scene_path:  path to the scene, assuming the files are in path_to_scene/model/sparse/0.
        mapper:      name of the algorithm to evaluate. [colmap or glomap]
        model:       name of the model to evaluate. [e.g., sparse/0]
    """
    # read model
    model_path = Path(scene_path, mapper, model)
    cameras, images, points = read_colmap_model(model_path)
    # convert to text
    write_model(cameras, images, points, model_path, ext=".txt")


def colmap_images2dict(data):
    """
    Cast from images.bin/txt from COLMAP format to dict, e.g. {name: {qvec, tvec}}.
    Args:
        data: object read from images.bin/txt file with read_model from colmap/scripts/python/read_write_model.
    Returns:
        images: dict with images data {name: {qvec, tvec}}.
    """
    # images_dict = {data[i].name: {"qvec":data[i].qvec, "tvec":data[i].tvec} for i in data.keys()}
    images_dict = {
        data[i].name.split("/")[-1]: {"qvec": data[i].qvec, "tvec": data[i].tvec}
        for i in data.keys()
    }
    return images_dict


#  -----------------------------
#  ----------- Evals -----------
#  -----------------------------


def evaluate_scene(images_gt_dict, images_pred_dict, deg=True):
    """
    Given two dictionaries {"image_idx":{qvec, tvec}}, evaluate the relative pose between all the possible pairs of images.
    Args:
        images_gt_dict:   dictionary with the ground truth poses.
        images_pred_dict: dictionary with the predicted poses.
        deg: if True, the errors are returned in degrees else in radians. Default is True.
    Returns:
        df: dataframe with the relative pose errors with keys {image1, image2, q_error, t_error}.
    TODO:
        - One may read and pass quaternions since the metric in compute in quaternion space.
    """

    df = {
        "image1": [],
        "image2": [],
        "q_error": [],
        "t_error": [],
        "max_error": [],
    }

    # for each pair of images in the ground truth
    for image_1_path, image_2_path in combinations(images_gt_dict.keys(), 2):
        # set to inf if both images have not been registered (= in *_pred_dict)
        if not (
            image_1_path in images_pred_dict.keys()
            and image_2_path in images_pred_dict.keys()
        ):  # working?
            q_err, t_err, max_error = np.inf, np.inf, np.inf
        else:
            # get the rotation and translation for two images (GT)
            R1_gt, t1_gt = (
                qvec2rotmat(images_gt_dict[image_1_path]["qvec"]),
                images_gt_dict[image_1_path]["tvec"],
            )
            R2_gt, t2_gt = (
                qvec2rotmat(images_gt_dict[image_2_path]["qvec"]),
                images_gt_dict[image_2_path]["tvec"],
            )

            # get the rotation and translation for two images (predicted)
            R1_pred, t1_pred = (
                qvec2rotmat(images_pred_dict[image_1_path]["qvec"]),
                images_pred_dict[image_1_path]["tvec"],
            )
            R2_pred, t2_pred = (
                qvec2rotmat(images_pred_dict[image_2_path]["qvec"]),
                images_pred_dict[image_2_path]["tvec"],
            )

            # compute the relative pose between the two images (GT)
            R_gt = R2_gt @ R1_gt.T
            t_gt = t2_gt - R_gt @ t1_gt

            # compute the relative pose between the two images (predicted)
            R_pred = R2_pred @ R1_pred.T
            t_pred = t2_pred - R_pred @ t1_pred

            # compute the error
            q_err, t_err = evaluate_R_t(R_pred, t_pred, R_gt, t_gt, deg=deg)
            max_error = max(q_err, t_err)

        # append to the dataframe
        df["image1"].append(image_1_path)
        df["image2"].append(image_2_path)
        df["q_error"].append(q_err)
        df["t_error"].append(t_err)
        df["max_error"].append(max_error if max_error < 10 else 180)

    return pd.DataFrame(df)


def eval_colmap_model(
    model_path, gt_path, thrs=[1, 3, 5], return_df=False, AUC_col="max_error"
):
    """
    Given a scene path, evaluate the model in the given folder versus the ground truth in the gt_folder.
    Args:
        model_path: path to the model to evaluate.
        gt_path:    path to the ground truth model.
        thrs:       list of thresholds for the AUC computation.
        return_df:  if True, return the dataframe with the errors.
        AUC_col:    column to compute the AUC. Default is "max_error". Other options are "q_error" and "t_error".
    Returns:
        df_AUC: dataframe with the AUC values for the model.

    """

    if not os.path.exists(model_path):
        raise Exception(f"Path {model_path} does not exist.")

    if not os.path.exists(gt_path):
        raise Exception(f"Path {gt_path} does not exist.")

    # read models
    cameras_gt, images_gt, points_gt = read_colmap_model(model_path=gt_path)
    cameras_pred, images_pred, points_pred = read_colmap_model(
        model_path=model_path
    )  # e.g. /colmap/sparse/0
    # print(f"Found {len(images_gt)} images in ground truth model and {len(images_pred)} images in {model_path.parts[-4]}.")
    # print(f"Registered {100*len(images_pred)/len(images_gt):.2f}% images in {model_path.parts[-4]}.")

    # format them to {image_name: {qvec, tvec}}
    images_gt_dict = colmap_images2dict(images_gt)
    images_pred_dict = colmap_images2dict(images_pred)

    # evaluate scene (each pair of images) and compute the AUC
    df = evaluate_scene(images_gt_dict, images_pred_dict)
    AUC_score_max = np.array(compute_AUC_pxsfm(df[AUC_col], thrs))

    if return_df:
        return AUC_score_max, df

    return AUC_score_max


def eval_colmap_model_all_scenes_joblib(
    scene_path,
    mapper="colmap",
    gt_path="./sparse",
    gt_sparse_name="sparse",
    scene_sparse_name="sparse",
    thrs=[0.5, 1, 3, 5],
    AUC_col="max_error",
    n_jobs=-1,
) -> pd.DataFrame:
    """
    Evaluate the model on all the scenes in the data_path using parallel processing.
    These must be in COLMAP format. The model is evaluated at the specified thresholds.

    Args:
        scene_path (Path): Path to the directory containing the COLMAP models for each scene.
        gt_path (Path, optional): Path to the ground truth models for each scene. Defaults to "./sparse".
        thrs (List[int], optional): List of thresholds for AUC computation.
        AUC_col (str, optional): Column to compute the AUC from in the evaluation DataFrame.
                                 Defaults to "max_error". Other options are "q_error" and "t_error".
        n_jobs (int, optional): Number of parallel jobs to run. -1 means using all available CPU cores. Defaults to -1. # max=16

    Returns:
        pd.DataFrame: A DataFrame with the AUC values for each scene, indexed by scene name.
    """

    s = time.time()

    # Define a helper function to evaluate a single scene
    def evaluate_single_scene(scene_path, scene_name, mapper, gt_path, thrs, AUC_col):
        model_path = scene_path / scene_name / mapper / scene_sparse_name / "0"
        gt_path = gt_path / scene_name / gt_sparse_name
        print(f"Evaluating...")
        print(f"model_path: {model_path}")
        print(f"gt_path: {gt_path}")

        if not model_path.exists():
            print(f"Model does not exist in {model_path}. Skipping.\n")
            return scene_name, None  # Skip if the model does not exist
        try:
            auc_scores = eval_colmap_model(
                model_path=model_path, gt_path=gt_path, thrs=thrs, AUC_col=AUC_col
            )
            return scene_name, auc_scores
        except Exception as e:
            print(f"Error in {scene_name}: {e}\n")
            return scene_name, None  # Return None for failed scenes

    # Use joblib to parallelize the evaluation of each scene
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_scene)(
            scene_path, scene_name, mapper, gt_path, thrs, AUC_col
        )  # <-- CORRECTED LINE
        for scene_name in tqdm(sorted(os.listdir(gt_path)), desc="Evaluating scenes")
    )

    # Process results and create the DataFrame
    res = {}
    for scene_name, auc_scores in tqdm(results, desc="Processing results"):
        if auc_scores is not None:
            res[scene_name] = auc_scores

    # Creating the DataFrame and transposing it to have the scenes as rows
    df_res_colmap = pd.DataFrame(res, index=thrs).transpose()

    # Rename the columns as {model}@{thrs}
    df_res_colmap.columns = [f"{mapper}@{thr}" for thr in thrs]
    print(f"Evaluation completed in {time.time() - s:.2f} seconds.")
    return df_res_colmap.round(2)

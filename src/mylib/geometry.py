# direct importes
import torch
import kornia
import cv2
import numpy as np
import poselib
from .conversions import *  # Changed from 'mylib.src.conversions import *'
from .projections import *  # Changed from 'mylib.src.projections import *

from typing import Union
from torch import Tensor

TensorOrArray = Union[torch.Tensor, np.ndarray]


def compute_relative_camera_motion(
    R1: TensorOrArray = None,  # bx3x3
    t1: TensorOrArray = None,  # bx3x1
    R2: TensorOrArray = None,  # bx3x3
    t2: TensorOrArray = None,  # bx3x1
    P1: TensorOrArray = None,  # bx3x4
    P2: TensorOrArray = None,  # bx3x4
):  # -> tuple[Tensor, Tensor]: # bx3x3, bx3x1
    """
    Compute the relative camera motion between two poses.
    Args:
        R1: rotation matrix or quaternion 1
        R2: rotation matrix or quaternion 2
        t1: translation vector 1
        t2: translation vector 2
    Returns:
        R_rel: relative rotation matrix
        t_rel: relative translation vector
    """
    # assert either P1 and P2 are given or R1, R2, t1, t2
    assert (P1 is not None and P2 is not None) or (
        R1 is not None and R2 is not None and t1 is not None and t2 is not None
    ), "Either P1 and P2 or R1, R2, t1, t2 must be provided"

    if P1 is not None:
        R1, t1 = from_P_to_Rt(to_torch(P1))
    else:
        R1 = to_torch(R1, b=True)
        t1 = to_torch(t1, b=True)
    if P2 is not None:
        R2, t2 = from_P_to_Rt(to_torch(P2))
    else:
        R2 = to_torch(R2, b=True)
        t2 = to_torch(t2, b=True)

    if R1.shape[-1] == 4:
        # its a quaternion
        R1 = kornia.geometry.conversions.quaternion_to_rotation_matrix(R1)

    if R2.shape[-1] == 4:
        # its a quaternion
        R2 = kornia.geometry.conversions.quaternion_to_rotation_matrix(R2)

    t1 = t1.reshape(-1, 3, 1)
    t2 = t2.reshape(-1, 3, 1)

    # R_rel, t_rel = kornia.geometry.epipolar.relative_camera_motion(R0, t0, R1, t1)
    R_rel = R2 @ R1.transpose(-1, -2)
    t_rel = t2 - R_rel @ t1
    t_rel = t_rel / t_rel.norm()

    return R_rel, t_rel.reshape(-1, 3)


def compute_essential_from_poses(
    R1: TensorOrArray,  # bx3x3
    R2: TensorOrArray,  # bx3x1
    t1: TensorOrArray,  # bx3x3
    t2: TensorOrArray,  # bx3x1
):  # -> Tensor: # bx3x3, bx3x1
    """
    Function to compute the essential matrix from the pose matrices.
    Args:
        P0: pose matrix of image 0
        P1: pose matrix of image 1
    Returns:
        Em: essential matrix
    """

    R_rel, t_rel = compute_relative_camera_motion(
        R1, R2, t1, t2
    )  # the function takes care of the conversions

    return compute_essential_from_relative_motion(R_rel, t_rel)


def compute_essential_from_relative_motion(R, t):
    """
    Compute the essential matrix from the relative rotation and translation.
    Args:
        R: relative rotation matrix
        t: relative translation vector
    Returns:
        Em: essential matrix
    """
    R = to_torch(R)
    t = to_torch(t, b=False)

    if R.shape[-1] == 4:
        # its a quaternion
        R = kornia.geometry.conversions.quaternion_to_rotation_matrix(R)

    Tx = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(t)
    Em = Tx @ R

    return Em


def compute_essential_poselib(
    points1,
    points2,
    camera_dict1,
    camera_dict2=None,
    return_Rt=False,
    max_epipolar_error=1.0,
):
    """
    Compute realtive R and t from the poses and the points. Then compute the essential matrix.
    """
    bundle_opt = {
        "max_iterations": 100,
        "loss_type": "CAUCHY",
        "loss_scale": 1.0,
        "gradient_tol": 1e-10,
        "step_tol": 1e-08,
        "initial_lambda": 1e-3,
        "min_lambda": 1e-10,
        "max_lambda": 1e10,
        "verbose": False,
    }

    # Interesting: with max_it=0, sucks, with max_it=1, it's good
    ransac_opt = {
        "max_iterations": 50000,
        "min_iterations": 1000,
        "dyn_numtrials_mult": 3.0,
        "success_prob": 0.9999,
        "max_reproj_error": 12.0,
        "max_epipolar_error": max_epipolar_error,
        "seed": 0,
        "progessive_sampling": False,
        "max_prosac_iterations": 100000,
        "real_focal_check": False,
    }

    if camera_dict2 is None:
        camera_dict2 = camera_dict1

    points1 = to_torch(points1, b=False).double().reshape(-1, 2, 1).numpy()
    points2 = to_torch(points2, b=False).double().reshape(-1, 2, 1).numpy()

    pose, dict_pose = poselib.estimate_relative_pose(
        points2D_1=list(points1),
        points2D_2=list(points2),
        camera1_dict=camera_dict1,
        camera2_dict=camera_dict2,
        ransac_opt=ransac_opt,
        bundle_opt=bundle_opt,
    )
    R, t = pose.R, pose.t
    E_poselib = compute_essential_from_relative_motion(pose.q, pose.t)

    if return_Rt:
        return E_poselib, np.array(dict_pose["inliers"]).astype(int), R, t

    return E_poselib, np.array(dict_pose["inliers"]).astype(np.bool_)


def compute_fundamental_from_relative_motion(R, t, K0, K1):
    """
    Compute the fundamental matrix from the relative rotation and translation.
    Args:
        R: relative rotation matrix
        t: relative translation vector
        K0: intrinsics matrix of image 0
        K1: intrinsics matrix of image 1
    Returns:
        Fm: fundamental matrix
    """
    R, t = to_torch(R), to_torch(t, b=False)
    K0, K1 = to_torch(K0, b=True), to_torch(K1, b=True)
    Em = compute_essential_from_relative_motion(R, t)
    Fm = torch.bmm(K1.permute(0, 2, 1).inverse(), torch.bmm(Em, K0.inverse()))
    return Fm


def compute_fundamental_from_poses(
    R1: TensorOrArray,  # bx3x3
    R2: TensorOrArray,  # bx3x1
    t1: TensorOrArray,  # bx3x3
    t2: TensorOrArray,  # bx3x1
    K1: TensorOrArray,  # bx3x3
    K2: TensorOrArray,  # bx3x1
):  # -> Tensor: # bx3x3, bx3x1
    """
    Function to compute the fundamental matrix from the pose and intrisics matrices.
    Args:
        R1: rotation matrix or quaternion 1
        R2: rotation matrix or quaternion 2
        t1: translation vector 1
        t2: translation vector 2
        K0: intrinsics matrix of image 0
        K1: intrinsics matrix of image 1
    Returns:
        Fm: fundamental matrix
    """

    K1, K2 = to_torch(K1, b=True), to_torch(K2, b=True)

    Em = compute_essential_from_poses(R1, R2, t1, t2)
    # Fm_kornia = kornia.geometry.epipolar.fundamental_from_essential(Em_kornia, K0, K1)
    Fm = torch.bmm(K2.permute(0, 2, 1).inverse(), torch.bmm(Em, K1.inverse()))
    return Fm


def compute_fundamental_from_essential(
    E: TensorOrArray,  # bx3x3
    K1: TensorOrArray,  # bx3x3
    K2: TensorOrArray,  # bx3x1
):  # -> Tensor: # bx3x3, bx3x1
    """
    Compute the fundamental matrix from the essential matrix and the intrinsics matrices.
    Args:
        E: essential matrix
        K1: intrinsics matrix of image 0
        K2: intrinsics matrix of image 1
    Returns:
        Fm: fundamental matrix
    """
    E = to_torch(E, b=True)
    K1, K2 = to_torch(K1, b=True), to_torch(K2, b=True)

    Fm = K2.permute(0, 2, 1).inverse() @ E @ K1.inverse()
    return Fm


def compute_epipolar_lines_coeff(
    E: TensorOrArray,  # bx3x3
    points: TensorOrArray,  # bxNx2
    K=None,  # bx3x3
):  # :# -> tuple[Tensor, Tensor]: # bx3x3, bx3x1
    """
    Compute the epipolar lines coefficients from the essential/fundamental matrix and the points.
    It is needed to unproject points if using the Essetial matrix.
    Args:
        E: essential matrix
        points: points in the image
        K: intrinsics matrix. If not provide E is assumed to be the fundamental matrix.
    Returns:
        epi_lines: epipolar lines coefficients
    """
    points = to_torch(points, b=False)
    E = to_torch(E, b=False)[0]
    if K is not None:
        K = to_torch(K, b=False)

    if K is not None:
        points = unproject_points2d(points, K, remove_last=False)
    else:
        points = to_homogeneous(points)

    return (E @ points.T).T  # epipolar coefficients [a,b,c] for each point


def compute_epipolar_lines_to_plot_from_F(img, F, points, K=None):
    """
    plot output as plt.plot(line[:2], line[2:]), they are justy the first and last point. plt will make a segment.
    Args:
        img: tensor (H,W,C)
        F: tensor (1,3,3)
        points: tensor (N,2)
        K: tensor (1,3,3)

    """
    points = to_torch(points, b=False)
    F = to_torch(F, b=False)
    if K is not None:
        K = to_torch(K, b=False)

    lines = compute_epipolar_lines_coeff(F, points, K)

    line_range = []  # eventually use torch.empty()
    xs = torch.tensor([0, img.shape[1]])
    for line in lines:
        a, b, c = line.squeeze()
        ys = -(c + a * xs) / b
        line_range.append(torch.stack([xs, ys], dim=0))  # x1,x2,y1,y2

    return torch.stack(line_range)

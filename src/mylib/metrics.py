# direct importes
import torch
import kornia
import cv2
import numpy as np
from .conversions import *
from .projections import *

from typing import Union
from torch import Tensor

TensorOrArray = Union[torch.Tensor, np.ndarray]


def check_epipolar_constraint(
    F: TensorOrArray,  # bx3x3
    points1: TensorOrArray,  # bxNx2
    points2: TensorOrArray,  # bxNx2
    K1=None,  # bx3x3
    K2=None,  # bx3x3
    return_tensor=False,
) -> Tensor:  # bxN
    """
    Check the epipolar constraint for a set of points.
    Args:
        F: fundamental matrix [3,3]
        points1: points in image 1 [N,2]
        points2: points in image 2 [N,2]
        K1: intrinsics matrix of image 1. If K1 in not provided points1 are assumed to be in normalized coordinates.
        K2: intrinsics matrix of image 2. If K2 in not provided points2 are assumed to be in normalized coordinates.
    Returns:
        epipolar_constraint: mean epipolar constraint error
    """

    F = to_torch(F)
    if K1 is not None:
        K1 = to_torch(K1, b=False)
        points1 = unproject_points2d(points1, K1, remove_last=False)
    else:
        points1 = to_homogeneous(points1)
    if K2 is not None:
        K2 = to_torch(K2, b=False)
        points2 = unproject_points2d(points2, K2, remove_last=False)
    else:
        points2 = to_homogeneous(points2)

    errors12 = (
        points2.reshape(-1, 3, 1).permute(0, 2, 1) @ F @ points1.reshape(-1, 3, 1)
    )
    errors21 = (
        points1.reshape(-1, 3, 1).permute(0, 2, 1)
        @ F.permute(0, 2, 1)
        @ points2.reshape(-1, 3, 1)
    )

    if return_tensor:
        return errors12.flatten(), errors21.flatten()

    return {"1->2": errors12.mean().item(), "2->1": errors21.mean().item()}


def compute_sampson_error(
    F: TensorOrArray,  # bx3x3
    points1: TensorOrArray,  # bxNx2
    points2: TensorOrArray,  # bxNx2
    K1=None,  # bx3x3
    K2=None,  # bx3x3
    return_tensor=False,
) -> Tensor:  # bxN
    """
    Compute the Sampson error for a set of points.
    Args:
        F: fundamental matrix [3,3]
        points1: points in image 1 [N,2]
        points2: points in image 2 [N,2]
        K1: intrinsics matrix of image 1. If K1 in not provided points1 are assumed to be in normalized coordinates.
        K2: intrinsics matrix of image 2. If K2 in not provided points2 are assumed to be in normalized coordinates.
    Returns:
        sampson_error: Sampson error
    """

    F = to_torch(F)
    if K1 is not None:
        K1 = to_torch(K1, b=False)
        points1 = unproject_points2d(points1, K1, remove_last=False)
    else:
        points1 = to_homogeneous(points1)
    if K2 is not None:
        K2 = to_torch(K2, b=False)
        points2 = unproject_points2d(points2, K2, remove_last=False)
    else:
        points2 = to_homogeneous(points2)

    errors12 = kornia.geometry.epipolar.sampson_epipolar_distance(points1, points2, F)
    errors21 = kornia.geometry.epipolar.sampson_epipolar_distance(
        points2, points1, F.permute(0, 2, 1)
    )

    if return_tensor:
        return errors12.flatten(), errors21.flatten()

    return {
        "1->2": round(errors12.mean().item(), 6),
        "2->1": round(errors21.mean().item(), 6),
    }


def evaluate_R_err(R_gt, R, deg=True):
    eps = 1e-15

    # Make and normalize the quaternions.
    q = quaternion_from_matrix_colmap(R)
    q_gt = quaternion_from_matrix_colmap(R_gt)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    # Relative Rotation Angle in radians. Equivalant to acos(trace(R)*.5) with R = R_gt*R^T but more stable.
    loss_q = np.maximum(
        eps, (1.0 - np.inner(q, q_gt) ** 2)
    )  # Max to void NaNs, always > 0 due to **2.
    err_q = np.arccos(1 - 2 * loss_q)

    if deg:
        err_q = np.rad2deg(err_q)  # rad*180/np.pi

    if np.sum(np.isnan(err_q)):
        # This should never happen! Debug here
        import IPython

        IPython.embed()

    return err_q.item()


def evaluate_t_err(t_gt, t, deg=True):
    t_gt = to_numpy(t_gt)
    t = to_numpy(t)
    # Flatten
    t = t.flatten()
    t_gt = t_gt.flatten()
    eps = 1e-15

    # Equivalent to arccos(cosine_sim(t,t_gt))
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.inner(t, t_gt) ** 2))  # Max to void NaNs
    err_t = np.arccos(np.sqrt(1 - loss_t))
    # err_t = np.arccos(np.clip(np.inner(t,t_gt), -1.0, 1.0)) # Equivalent to above

    if np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython

        IPython.embed()

    if deg:
        err_t = np.rad2deg(err_t)  # rad*180/np.pi

    return err_t.item()


def evaluate_R_t(R_gt, t_gt, R, t, deg=True):
    """
    Evaluate the rotation and translation errors between two poses. From IMC2020.
    Args:
        R_gt: Ground truth relative rotation matrix.
        t_gt: Ground truth relative translation vector.
        R:    Predicted relative rotation matrix.
        t:    Predicted relative translation vector.
    Returns:
        err_q: Rotation error in radians.
        err_t: Translation error in radians.
    """
    err_q = evaluate_R_err(R_gt, R, deg=deg)
    err_t = evaluate_t_err(t_gt, t, deg=deg)

    return np.stack([err_q, err_t])


def compute_recall_pxsfm(errors):
    """
    Compute the recall for the errors. From Pixel-Perfect SfM.
    Args:
        errors: numpy array or errors.
    Returns:
        errors: sorted errors.
        recall: recall for each.

    """
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements  # cumsum accuracy?
    return errors, recall


def compute_AUC_pxsfm(errors, thresholds, min_error=None):
    """
    Compute the AUC for one array of errors. From Pixel-Perfect SfM.
    Args:
        errors: numpy array or errors.
        thresholds: list of thresholds for the AUC computation.
        min_error: minimum error to consider.
    Returns:
        aucs: list with the AUC values for each threshold.
    Note:
        - It is computed as the defined integral of the recall over the error.
    """
    l = len(errors)

    errors, recall = compute_recall_pxsfm(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / l
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:  # [1,3,5]
        last_index = np.searchsorted(
            errors, t, side="right"
        )  # index of the first element >= t
        r = np.r_[recall[:last_index], recall[last_index - 1]]  # error < t
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t  # ?
        aucs.append(auc * 100)
    return aucs

from __future__ import annotations


# direct importes
import torch
import kornia
import cv2

# imports as alias
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# imports from
from torch import Tensor
from .conversions import *
from .geometry import *

from typing import Union

TensorOrArray = Union[torch.Tensor, np.ndarray]


# not checked
def scale_intrisics(
    K: Tensor, scaling_factor_x: float, scaling_factor_y: float = None
) -> Tensor:
    """
    Function to scale the intrinsics matrix based on the scaling factor. Used for scaling the intrinsics matrix when scaling images.
    Args:
        K: intrinsics matrix of image
        scaling_factor_x: scaling factor for x-axis or both axes if scaling_factor_y is None
        scaling_factor_y: scaling factor for y-axis, if None, scaling_factor_x is used for both axes
    Returns:
        scaled_K: reduced intrinsics matrix of image
    """
    # not checked
    # assert considering batch size of B
    assert K.shape == (1, 3, 3), "Intrinsics matrix should be of shape 1x3x3"

    if scaling_factor_y is None:
        scaling_factor_y = scaling_factor_x

    scaling_factor_matrix = torch.tensor(
        [scaling_factor_x, 0, 0, 0, scaling_factor_y, 0, 0, 0, 1]
    ).view(1, 3, 3)
    scaled_K = scaling_factor_matrix @ K

    return scaled_K


def reduce_images_pt(
    image: Tensor, new_height: int, new_width: int, permute: bool = False
) -> Tensor:
    """
    Function to reduce the image-like tensors.
    Args:
        image0: image 0 of shape HxWxC
        scaling_factor_x: scaling factor for x-axis or both axes if scaling_factor_y is None
        scaling_factor_y: scaling factor for y-axis, if None, scaling_factor_x is used for both axes
    Returns:
        image_red: reduced image 0
    """
    assert len(image.shape) == 3, "Images should be of shape HxWxC"
    assert image.shape[-1] in [1, 3], "Channels should be 1 or 3"

    image_reduced = F.interpolate(
        image.permute(2, 0, 1)[None], size=(new_height, new_width), mode="nearest"
    )

    if permute:
        image_reduced = image_reduced.permute(0, 2, 3, 1)

    if len(image.shape) == 3:
        return image_reduced[0]

    return image_reduced


# not checked
def project_points_2D_to_2D(
    points, Z0, Rt0, Rt1, K0, K1, device="cpu", cut_out_of_image=True
):
    points = to_torch(points)
    Z0 = to_torch(Z0)
    Rt0 = to_torch(Rt0)
    Rt1 = to_torch(Rt1)
    K0 = to_torch(K0)
    K1 = to_torch(K1)
    p0 = to_homogeneous(points).view(-1, 3, 1).to(device)

    # filter out points out of depth shape
    mask = filter_points_outside_image(p0, Z0.shape)
    p0 = p0[mask.view(-1, 1, 1).expand(-1, 3, -1)].reshape(-1, 3, 1)

    z0_x, z0_y = p0[:, 0, :].long().cpu(), p0[:, 1, :].long().cpu()
    z = Z0[0, z0_y, z0_x].unsqueeze(-1).to(device)  # N,1,1

    Rt0_torch, Rt1_torch = Rt0.to(device), Rt1.to(device)
    K0_torch, K1_torch = K0.to(device), K1.to(device)

    R0_torch = Rt0_torch[:, :3, :3]
    R1_torch = Rt1_torch[:, :3, :3]
    t0_torch = Rt0_torch[:, :3, 3].view(-1, 3, 1)
    t1_torch = Rt1_torch[:, :3, 3].view(-1, 3, 1)

    R_rel = torch.bmm(R1_torch, R0_torch.transpose(1, 2))
    t_rel = t1_torch - torch.bmm(R_rel, t0_torch)

    p1_r = K1_torch @ R_rel @ K0_torch.inverse() @ p0
    p1_t = K1_torch @ t_rel / z
    p1 = p1_r + p1_t

    if cut_out_of_image:
        mask = filter_points_outside_image(p1, Z0.shape)
        print(p1.shape, Z0.shape, mask.shape)
        p0 = p0[mask.view(-1, 1, 1).expand(-1, 3, -1)].reshape(-1, 3, 1)
        p1 = p1[mask.view(-1, 1, 1).expand(-1, 3, -1)].reshape(-1, 3, 1)

    # dehomogenize
    p0 = dehomogenize(p0).cpu()
    p1 = dehomogenize(p1).cpu()
    return p0, p1


def filter_points_outside_image(points, image_shape):
    """
    Function to filter points outside the image.
    Args:
        points: points of shape (N,3,1 )
        image_shape: shape of the image (*,H,W)
    Returns:
        filtered_points: points inside the image
    """
    H, W = image_shape[-2], image_shape[-1]

    x, y = points[:, 0, :], points[:, 1, :]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return mask


def create_grid(image: Tensor) -> Tensor:
    """
    Function to create a grid of the same size as the image.
    Args:
        image: image of shape HxWxC
    Returns:
        grid: grid of the same size as the image HxWx2
    """

    assert len(image.shape) == 3, "Images should be of shape HxWxC"
    assert image.shape[-1] in [1, 3], "Channels should be 1 or 3"

    H, W, C = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing="ij"
    )  # add +.5 for center of pixel
    grid = torch.stack((grid_x, grid_y), dim=-1).float()  # Add batch dimension

    return grid


def project_grid_with_depth(
    grid: Tensor, P0: Tensor, P1: Tensor, K0: Tensor, K1: Tensor, Z0: Tensor
) -> Tensor:
    """
    Project image 0 grid to image 1 using depth and intrisisc information.
    Args:
        grid: grid of shape HxWx2
        P0: pose matrix of image 0
        P1: pose matrix of image 1
        K0: intrinsics matrix of image 0
        K1: intrinsics matrix of image 1
        Z0: depth of image 0
    Returns:
        grid_homogeneous: images 0's grid projected to image 1
    """

    assert K0.shape == (1, 3, 3), "Intrinsics matrix should be of shape 1x3x3"
    assert K1.shape == (1, 3, 3), "Intrinsics matrix should be of shape 1x3x3"
    assert grid.shape[-1] == 2, "Grid last dimension be 2"
    # assert len(grid.shape) == 3, "Grid should be of shape HxWx2"
    assert len(Z0.shape) == 3, "Depth should be of shape 1xHxW"

    # swap poses to send image 0 to image 1, the grid will result in the inverse transformation
    # K0, K1 = K1, K0
    # P0, P1 = P1, P0

    # transform grid in homogeneous coordinates
    grid = torch.cat((grid, torch.ones_like(grid[..., :1])), dim=-1).float()
    # grid = to_homogeneous(grid).float()
    H, W = grid.shape[0], grid.shape[1]

    R_rel, t_rel = compute_relative_camera_motion(P0, P1)
    z0_x, z0_y = grid[:, :, 0].long().cpu(), grid[:, :, 1].long().cpu()
    Z = Z0[:, z0_y, z0_x].unsqueeze(-1)

    # rotation (homography)
    grid_rotated = K1 @ R_rel @ K0.inverse() @ grid.view(-1, 3, 1)
    # translation
    grid_translated = grid_rotated + K1 @ t_rel / Z.view(-1, 1, 1)
    # normalization
    grid_homogeneous = grid_translated / grid_translated[:, 2].unsqueeze(-1)

    return grid_homogeneous[:, :2].view(H, W, 2)


def project_image(image: Tensor, new_coords: Tensor) -> Tensor:
    """
    Function to wrap the image to new coordinates (that are)
    Args:
        image: image fo shape HxWxC
        new_coords: new coordinates to project the image
    Returns:
        projected_image: image projected to new coordinates
    """
    assert len(image.shape) == 3, "Images should be of shape HxWxC"

    H, W, C = image.shape
    image = image.permute(2, 0, 1)[None].float()  # BxCxHxW

    # normalize such that [-1,-1] is top left and [1,1] is bottom right
    normalized_new_coords = 2.0 * new_coords / torch.tensor([W, H]) - 1.0
    normalized_new_coords = normalized_new_coords[None]

    projected_image = F.grid_sample(
        image,
        normalized_new_coords,
        mode="nearest",
        align_corners=False,
        padding_mode="zeros",
    )

    return projected_image.permute(0, 2, 3, 1)[0]


def project_image_with_depth(
    image: Tensor, P0: Tensor, P1: Tensor, K0: Tensor, K1: Tensor, Z0: Tensor
) -> Tensor:
    """
    Function to project the image to new coordinates.
    Args:
        see above functions
    Returns:
        projected_image: image projected to new coordinates
    """

    grid = create_grid(image)
    project_grid = project_grid_with_depth(
        grid, P1, P0, K1, K0, Z0
    )  # Poses need to be inverted to send the grid from image 0 to image 1
    projected_image = project_image(image, project_grid)

    return projected_image


def unproject_points2d(points, K, remove_last=True):
    """
    Unproject 2D points to 3D points.
    """
    points = to_torch(points, b=False)
    K = to_torch(K, b=False)

    points = to_homogeneous(points)
    points_unprojected = (K.inverse() @ points.permute(-1, -2)).permute(
        -1, -2
    )  # K^-1 != K.T

    if remove_last:
        points_unprojected = points_unprojected[:, :2] / points_unprojected[:, 2:]
        return points_unprojected.reshape(-1, 2)

    points_unprojected = points_unprojected / points_unprojected[:, 2:]
    return points_unprojected.reshape(-1, 3)


def compute_epipolar_lines_to_plot_from_line(img, line):

    xs = torch.tensor([0, img.shape[1]])
    a, b, c = line.squeeze()
    ys = -(c + a * xs) / b

    return torch.stack([xs, ys], dim=0)


def distance_line_points_parallel(line, points):
    """
    line: tensor [1,3], [3], [3,1]
    points: tensor [N,2]
    """
    a, b, c = line.flatten()
    x, y = points[:, 0], points[:, 1]
    return torch.abs(a * x + b * y + c) / (a**2 + b**2) ** 0.5

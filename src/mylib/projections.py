from __future__ import annotations


# direct importes
import torch

# imports as alias
import numpy as np
import torch.nn.functional as F

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


def grid_sample_nan(xy: Tensor, img: Tensor, mode="nearest") -> tuple[Tensor, Tensor]:
    """pytorch grid_sample with embedded coordinate normalization and grid nan handling (if a nan is present in xy,
    the output will be nan). Works both with input with shape B,n,2 and B,n0,n1,2
    xy point that fall outside the image are treated as nan (those which are really close are interpolated using
    border padding mode)
    Args:
        xy: input coordinates (with the convention top-left pixel center at (0.5, 0.5))
            B,n,2 or B,n0,n1,2
        img: the image where the sampling is done
            BxCxHxW or BxHxW
        mode: the interpolation mode
    Returns:
        sampled: the sampled values
            BxCxN or BxCxN0xN1 (if no C dimension in input BxN or BxN0xN1)
        mask_img_nan: mask of the points that had a nan in the img. The points xy that were nan appear as false in the
            mask in the same way as point that had a valid img value. This is done to discriminate between invalid
            sampling position and valid sampling position with a nan value in the image
            BxN or BxN0xN1
    """
    assert img.dim() in {3, 4}
    if img.dim() == 3:
        # ? remove the channel dimension from the result at the end of the function
        squeeze_result = True
        img.unsqueeze_(1)
    else:
        squeeze_result = False

    assert xy.shape[-1] == 2
    assert xy.dim() == 3 or xy.dim() == 4
    B, C, H, W = img.shape

    xy_norm = normalize_pixel_coordinates(xy, img.shape[-2:])  # BxNx2 or BxN0xN1x2
    # ? set to nan the point that fall out of the second image
    xy_norm[(xy_norm < -1) + (xy_norm > 1)] = float("nan")
    if xy.ndim == 3:
        sampled = F.grid_sample(
            img,
            xy_norm[:, :, None, ...],
            align_corners=False,
            mode=mode,
            padding_mode="border",
        ).view(
            B, C, xy.shape[1]
        )  # BxCxN
    else:
        sampled = F.grid_sample(
            img, xy_norm, align_corners=False, mode=mode, padding_mode="border"
        )  # BxCxN0xN1
    # ? points xy that are not nan and have nan img. The sum is just to squash the channel dimension
    mask_img_nan = torch.isnan(sampled.sum(1))  # BxN or BxN0xN1
    # ? set to nan the sampled values for points xy that were nan (grid_sample consider those as (-1, -1))
    xy_invalid = xy_norm.isnan().any(-1)  # BxN or BxN0xN1
    if xy.ndim == 3:
        sampled[xy_invalid[:, None, :].repeat(1, C, 1)] = float("nan")
    else:
        sampled[xy_invalid[:, None, :, :].repeat(1, C, 1, 1)] = float("nan")

    if squeeze_result:
        img.squeeze_(1)
        sampled.squeeze_(1)

    return sampled, mask_img_nan


def normalize_pixel_coordinates(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray
) -> Tensor:
    """normalize pixel coordinates from -1 to +1. Being (-1,-1) the exact top left corner of the image
    the coordinates must be given in a way that the center of pixel is at half coordinates (0.5,0.5)
    xy ordered as (x, y) and shape ordered as (H, W)
    Args:
        xy: input coordinates in order (x,y) with the convention top-left pixel center is at coordinates (0.5, 0.5)
            ...x2
        shape: shape of the image in the order (H, W)
    Returns:
        xy_norm: normalized coordinates between [-1, 1]
    """
    xy_norm = xy.clone()
    # ? the shape index are flipped because the coordinates are given as x,y but shape is H,W
    xy_norm[..., 0] = 2 * xy_norm[..., 0] / shape[1]
    xy_norm[..., 1] = 2 * xy_norm[..., 1] / shape[0]
    xy_norm -= 1
    return xy_norm


def to_homogeneous(xy: Tensor) -> Tensor:
    return torch.cat((xy, torch.ones_like(xy[..., 0:1])), dim=-1)


def from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    z_vec: Tensor = points[..., -1:]
    # set the results of division by zero/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))
    output = scale * points[..., :-1]
    return output


def unproject_to_virtual_plane(
    xy: Tensor, K: Tensor, cast_to_double: bool = True
) -> Tensor:
    """unproject points to the camera virtual plane at depth 1
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns:
        xyz: 3D points laying on the virtual plane
            B,n,3
    """
    xy_hom = to_homogeneous(xy)  # B,n,3
    if False:  # cast_to_double:
        original_type = xy.dtype
        # Bx3x3 * Bx3xn = Bx3xn  -> B,n,3 after permute
        xyz = (
            (
                torch.inverse(K.to(torch.double))
                @ (xy_hom.permute(0, 2, 1).to(torch.double))
            )
            .permute(0, 2, 1)
            .to(original_type)
        )
    else:
        # Bx3x3 * Bx3xn = Bx3xn  -> B,n,3 after permute
        xyz = (torch.inverse(K) @ (xy_hom.permute(0, 2, 1))).permute(0, 2, 1)

    return xyz


def unproject_to_3D(xy: Tensor, K: Tensor, depths: Tensor) -> Tensor:
    """unproject points to 3D in the camera ref system
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        depths: the points depth
            B,n
    Returns:
        xyz: unprojected 3D points in the camera reference system
            B,n,3
    """
    assert xy.shape[0] == K.shape[0] and xy.shape[0] == depths.shape[0]
    assert xy.shape[1] == depths.shape[1]
    assert xy.shape[2] == 2

    xyz = unproject_to_virtual_plane(xy, K)  # B,n,3
    xyz *= depths[:, :, None]  # B,n,3

    return xyz


def invert_P(P: Tensor) -> Tensor:
    """invert the extrinsics P matrix in a more stable way with respect to np.linalg.inv()
    Args:
        P: input extrinsics P matrix
            Bx4x4
    Return:
        P_inv: the inverse of the P matrix
            Bx4x4
    Raises:
        None
    """
    B = P.shape[0]
    R = P[:, 0:3, 0:3]
    t = P[:, 0:3, 3:4]
    P_inv = torch.cat((R.permute(0, 2, 1), -R.permute(0, 2, 1) @ t), dim=2)
    P_inv = torch.cat(
        (P_inv, P.new_tensor([[0.0, 0.0, 0.0, 1.0]])[None, ...].repeat(B, 1, 1)), dim=1
    )
    return P_inv


def change_reference_3D_points(
    xyz0: Tensor, P0: Tensor, P1: Tensor, cast_to_double: bool = True
) -> Tensor:
    """move 3D points from P0 to P1 reference systems
    Args:
        xyz0: the 3D points in the P0 coordinate system
            B,n,3
        P0: the source coordinate system
            B,4,4
        P1: the destination coordinate system
            B,4,4
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns
        xyz1: the 3D points in the P1 coordinate system
            B,n,3
    """
    assert (
        xyz0.shape[0] == P0.shape[0] and xyz0.shape[0] == P1.shape[0]
    ), f"Expected xyz0 and P0 to have the same batch size, got {xyz0.shape[0]} and {P0.shape[0]}"
    assert xyz0.shape[2] == 3, f"Expected xyz0 to have 3 channels, got {xyz0.shape[2]}"
    assert (
        P0.shape[1] == 4 and P0.shape[2] == 4
    ), f"Expected P0 to have shape Bx4x4, got {P0.shape}"
    assert (
        P1.shape[1] == 4 and P1.shape[2] == 4
    ), f"Expected P1 to have shape Bx4x4, got {P1.shape}"

    xyz0_hom = to_homogeneous(xyz0)  # B,n,4
    if False:  # cast_to_double:
        original_dtype = xyz0.dtype
        P0_inv = invert_P(P0.float())
        xyz1_hom = P1.float() @ P0_inv @ xyz0_hom.permute(0, 2, 1).float()  # B,4,n
        xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1)).to(original_dtype)  # B,n,3
    else:
        P0_inv = invert_P(P0)
        xyz1_hom = P1 @ P0_inv @ xyz0_hom.permute(0, 2, 1)  # B,4,n
        xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1))  # B,n,3

    return xyz1


def filter_outside(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray, border: int = 0
) -> Tensor:
    """set as nan all the points that are not inside rectangle defined with shape HxW
    Args:
        xy: keypoints with coordinate (x, y)
            (B)xnx2
        shape: shape where the keypoints should be contained (H, W)
            2
        border: the minimum border to apply when masking
    Returns:
        Tensor: input keypoints with 'nan' where one of the two coordinates was not contained inside shape
        xy_filtered     (B)xnx2
    """
    assert xy.shape[-1] == 2, f"xy must have last dimension of size 2, got {xy.shape}"
    assert len(shape) == 2, f"shape must be a tuple of 2 elements, got {shape}"
    assert border < max(
        shape
    ), f"border must be smaller than the smallest shape dimension, got {border} and {shape}"

    xy = xy.clone()
    outside_mask = (
        (xy[..., 0] < border)
        + (xy[..., 0] >= shape[1] - border)
        + (xy[..., 1] < border)
        + (xy[..., 1] >= shape[0] - border)
    )  # (B)xn
    xy[outside_mask] = float("nan")
    return xy


def project_to_2D(
    xyz: Tensor,
    K: Tensor,
    img_shape: tuple[int, int] | None = None,
    border: int = 0,
) -> Tensor | tuple[Tensor, Tensor]:
    """project 3D points to 2D using the provided intrinsics matrix K. If img_shape is provided, set to nan the points
    that project out of the img and additionally return mask_outside boolean tensor
    Args:
        xyz: the 3D points
            B,n,3
        K: the camera intrinsics matrix
            B,3,3
        img_shape: if provided, set to nan the points that map out of the image and additionally return mask_outside
        border: if img_shape is provided, set to nan the points that map out of the image border
    Returns
        xy_proj: the 2D projection of the 3D points
            B,n,2
        mask_outside: optional (if img_shape is provided). True where the point map outside img_shape
            B,n bool
    """
    original_dtype = xyz.dtype
    # B,3,3 * B,3,n =  B,3,n  -> B,n,3 after permutation
    xy_proj_hom = (K.float() @ xyz.permute(0, 2, 1).float()).permute(0, 2, 1)
    xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # B,n,2

    if img_shape is not None:
        # ? filter points that fall outside the second image but have depth valid
        # ? as the comparison of a 'nan' values with something else is always false, only the points that had valid
        # ? depth will appear in mask_outside
        mask_outside = (
            (xy_proj[..., 0] < border)
            + (xy_proj[..., 0] >= img_shape[1] - border)
            + (xy_proj[..., 1] < border)
            + (xy_proj[..., 1] >= img_shape[0] - border)
        )
        xy_proj = filter_outside(xy_proj, img_shape, border)
        return xy_proj, mask_outside
    else:
        return xy_proj


def reproject_2D_2D(
    xy0: Tensor,
    depthmap0: Tensor,
    P0: Tensor,
    P1: Tensor,
    K0: Tensor,
    K1: Tensor,
    img1_shape: tuple[int, int] | None = None,
    border: int = 0,
    mode: str = "nearest",
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """projects xy0 points from img0 to img1 using depth0. Points that have an invalid depth='nan' are
        set to 'nan' (if bilinear sampling is used, all the 4 closest depth values must be valid to get a valid projection).
        If img1_shape is provided, also the points that project out of the second image are set to Nan
    Args:
        xy0: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        depthmap0: depthmap of img0
            B,H,W or B,n
        P0: camera0 extrinsics matrix
            B,4,4
        P1: camera1 extrinsics matrix
            B,4,4
        K0: camera0 intrinsics matrix
            B,3,3
        K1: camera1 intrinsics matrix
            B,3,3
        img1_shape: shape of img1 (H, W)
        border: if > 0, the points that project closer to the image borders are set to nan
        mode: depthmap interpolation mode, can be 'nearest' or 'bilinear'
    Returns:
        xy0_proj: the projected keypoints in img1
            B,n,2
        mask_invalid_depth: mask of points that had invalid depth
            B,n  bool
        mask_outside: optional (if img1_shape is provided) mask of points that had valid depth but project out of the
            second image
            B,n  bool
    """
    # ? interpolate depths
    if depthmap0.dim() == 3:
        selected_depths0, mask_invalid_depth0 = grid_sample_nan(
            xy0, depthmap0, mode=mode
        )  # Bxn, Bxn
    else:
        # pre-sampled depths
        assert (
            depthmap0.shape == xy0.shape[:2]
        ), f"If depthmap0 is not BxHxW, it must be Bxn, got {depthmap0.shape} and {xy0.shape}"
        selected_depths0 = depthmap0

    # ? use the depth to define the 3D coordinates of points in the ref system of camera0
    xyz0 = unproject_to_3D(xy0, K0, selected_depths0)  # B,n,3

    # ? change the ref system of the 3d point to camera1
    xyz0_proj = change_reference_3D_points(xyz0, P0, P1)  # B,n,3

    # ? project the point in the destination image
    if img1_shape is not None:
        xy0_proj, mask_outside0 = project_to_2D(
            xyz0_proj, K1, img1_shape, border
        )  # B,n,2, B,n,2
        return xy0_proj
    else:
        assert border == 0, "border must be 0 if img1_shape is not provided"
        xy0_proj = project_to_2D(xyz0_proj, K1)  # B,n,2, B,n,2
        return xy0_proj


def filter_points_outside_image(points, image_shape, border=5):
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
    mask = (x >= border) & (x < W - border) & (y >= border) & (y < H - border)
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

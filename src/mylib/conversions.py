import math
import torch
import kornia
import numpy as np
from copy import deepcopy


def is_torch(vector):
    """
    Check if a vector is a torch tensor.
    """
    if isinstance(vector, torch.Tensor):
        return True
    else:
        return False


def to_torch(vector_, b=True, grad=False):
    """
    Convert a numpy array or list to a torch tensor.
    Optionally add batch dim (b) and requires_grad (grad).
    Will NOT deepcopy tensors (for autograd safety).
    """
    # If already tensor, do not deepcopy (this breaks autograd!)
    if isinstance(vector_, torch.Tensor):
        vector = vector_
    else:
        vector = deepcopy(vector_)
        vector = torch.tensor(vector)
    # Add batch dimension if needed
    if b and len(vector.shape) < 3:
        vector = vector.unsqueeze(0)
    # Only set requires_grad if not already a tensor with requires_grad set
    if grad and not vector.requires_grad:
        vector = vector.clone().detach().requires_grad_()
    return vector.float()


def to_numpy(vector):
    """
    Convert a torch tensor to a numpy array.
    """
    if is_torch(vector):
        return vector.detach().cpu().numpy()
    return vector


def to_homogeneous(vector):
    """
    Convert a 2D vector to homogeneous coordinates.
    """
    vector = to_torch(vector, b=False)
    if vector.shape[1] == 2:
        vector = torch.hstack([vector, torch.ones_like(vector)[..., :1]])
    return vector.float()


def dehomogenize(points):
    """
    Function to dehomogenize points.
    Args:
        points: points of shape (N,3) or (N,2)
    Returns:
        dehomogenized_points: points of shape (N,2)
    """
    points = points.view(-1, 3)
    points = points / points[:, 2].unsqueeze(-1)
    points = points[:, :2]

    return points


def normalize_quat(quaternion):
    """
    Normalize a quaternion.
    """
    return kornia.geometry.conversions.normalize_quaternion(quaternion)


def quat_to_rotmat(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    """
    quaternion = to_torch(quaternion, b=False)
    # Normalize quaternion
    quaternion = normalize_quat(quaternion)
    return kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)


def rotmat_to_quat(rotmat, glomap_format=False):
    """
    Convert a rotation matrix to a quaternion.
    """
    rotmat = to_torch(rotmat, b=False)
    quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotmat)
    if glomap_format:
        quaternion = quaternion[[1, 2, 3, 0]]  # w,x,y,z -> x,y,z,w
    return normalize_quat(quaternion)


def rotmat_to_axis_angles(rotmat, degrees=True):
    """
    Convert a rotation matrix to euler angles.
    """
    rotmat = to_torch(rotmat, b=False)
    euler = kornia.geometry.conversions.rotation_matrix_to_axis_angle(rotmat)
    if degrees:
        euler = kornia.geometry.conversions.rad2deg(euler)
    return euler


def axis_angles_to_rotmat(axis_angles, degrees=True):
    """
    Convert euler angles to a rotation matrix.
    """
    axis_angles = to_torch(axis_angles, b=True)
    if degrees:
        axis_angles = kornia.geometry.conversions.deg2rad(axis_angles)
    return kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angles)


def concat_pose_no_batch(R, t):
    """
    Concatenate a rotation matrix and a translation vector to a pose matrix.
    """
    R = to_torch(R, b=False)
    t = to_torch(t, b=False)

    if R.shape[-1] == 4:
        # its a quaternion
        R = kornia.geometry.conversions.quaternion_to_rotation_matrix(R)

    t = t.flatten().reshape(3, 1)
    return torch.hstack([R, t])


def quaternion_from_matrix_colmap(matrix, isprecise=False):
    """Return quaternion from rotation matrix. From IMC2020.

    matrix: 3x3
    """

    matrix = np.asarray(to_numpy(matrix), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def rotmat_from_quaternion_colmap(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


def find_position(tensor, element):
    """
    Finds the position of an element in a 2D or 3D tensor.

    Parameters:
    - tensor (torch.Tensor): A tensor of shape (B x N x 2) or (N x 2).
    - element (torch.Tensor or list or tuple): The element (coordinate pair) to find in the tensor.

    Returns:
    - torch.Tensor or None: A tensor containing the indices of the element in the form
                            (batch_index, position_index) for B x N x 2,
                            or (position_index) for N x 2.
                            Returns None if the element is not found.
    """
    # Convert element to tensor if it isn't already
    tensor = to_torch(tensor)
    element = to_torch(element)

    # Check if tensor is 2D (N x 2) or 3D (B x N x 2)
    if tensor.dim() == 2:
        # For N x 2 tensors, directly compare to find matching rows
        positions = torch.nonzero(
            (tensor == element).all(dim=1), as_tuple=False
        ).squeeze()
    elif tensor.dim() == 3:
        # For B x N x 2 tensors, compare along the last dimension
        positions = torch.nonzero((tensor == element).all(dim=2), as_tuple=False)
    else:
        raise ValueError("Tensor must be of shape B x N x 2 or N x 2.")

    return positions if positions.numel() > 0 else None


def from_P_to_Rt(P):
    """
    Decompose a projection matrix into a rotation matrix and a translation vector.
    """
    P = to_torch(P, b=False)
    R = P[:, :3, :3]
    t = P[:, :3, 3:]
    return R, t

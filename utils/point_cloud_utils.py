import torch

from utils.common_utils import check_numpy_to_torch


def rotate_points_along_z(points, angle):
    """

    Args:
        points: ndarray, [B, N, 3 + C]
        angle: ndarray, [B], angle along z-axis, angle increases x ==> y

    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)

    return points_rot.numpy() if is_numpy else points_rot

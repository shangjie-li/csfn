import numpy as np
import torch

from utils.common_utils import check_numpy_to_torch
from utils.point_cloud_utils import rotate_points_along_z


def boxes3d_to_corners3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        boxes3d: ndarray, [N, 7], (x, y, z, dx, dy, dz, heading), where (x, y, z) is the box center

    Returns:
        corners3d: ndarray, [N, 8, 3]

    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def boxes3d_lidar_to_camera(boxes3d_lidar, calib):
    """

    Args:
        boxes3d_lidar: ndarray, [N, 7], (x, y, z, dx, dy, dz, heading), where (x, y, z) is the box center
        calib: kitti_calibration_utils.Calibration

    Returns:
        boxes3d_camera: ndarray, [N, 7], (x, y, z, h, w, l, r), where (x, y, z) is the box center

    """
    xyz_lidar = boxes3d_lidar[:, 0:3]
    l, w, h = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6]
    r = boxes3d_lidar[:, 6:7]

    xyz_camera = calib.lidar_to_rect(xyz_lidar)
    r = -r - np.pi / 2
    boxes3d_camera = np.concatenate([xyz_camera, h, w, l, r], axis=-1)

    return boxes3d_camera


def boxes3d_lidar_to_image(boxes3d_lidar, calib, image_shape=None):
    """

    Args:
        boxes3d_lidar: ndarray, [N, 7], (x, y, z, dx, dy, dz, heading), where (x, y, z) is the box center
        calib: kitti_calibration_utils.Calibration
        image_shape: [2], H and W

    Returns:
        boxes2d: ndarray, [N, 4], (u1, v1, u2, v2)

    """
    corners = boxes3d_to_corners3d(boxes3d_lidar[:, 0:7])  # [N, 8, 3]
    boxes2d = []
    for i in range(corners.shape[0]):
        pts_img, _ = calib.lidar_to_img(corners[i])
        min_u, min_v = pts_img[:, 0].min(), pts_img[:, 1].min()
        max_u, max_v = pts_img[:, 0].max(), pts_img[:, 1].max()
        box2d = [min_u, min_v, max_u, max_v]
        boxes2d.append(box2d)

    boxes2d = np.array(boxes2d).reshape(-1, 4)

    if image_shape is not None:
        boxes2d[:, 0] = np.clip(boxes2d[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d[:, 1] = np.clip(boxes2d[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d[:, 2] = np.clip(boxes2d[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d[:, 3] = np.clip(boxes2d[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d


def boxes_iou_normal(boxes_a, boxes_b):
    """

    Args:
        boxes_a: tensor, [N1, 4], (x1, y1, x2, y2) in lidar coordinates
        boxes_b: tensor, [N2, 4], (x1, y1, x2, y2) in lidar coordinates

    Returns:
        iou: tensor, [N1, N2]

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])

    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)

    return iou

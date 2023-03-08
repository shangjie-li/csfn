import cv2
import numpy as np

from utils.box_utils import boxes3d_to_corners3d

box_colormap = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 255, 0),
    'Cyclist': (0, 255, 255),
}  # BGR


def normalize_img(img):
    """
    Normalize the image data to np.uint8 and image shape to [H, W, 3].

    Args:
        img: ndarray, [H, W] or [H, W, 1] or [H, W, 3]

    Returns:
        img: ndarray of uint8, [H, W, 3]

    """
    img = img.copy()
    img += -img.min() if img.min() < 0 else 0
    img = np.clip(img / img.max(), a_min=0., a_max=1.) * 255.
    img = img.astype(np.uint8)
    img = img[:, :, None] if len(img.shape) == 2 else img

    assert len(img.shape) == 3
    if img.shape[-1] == 1:
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif img.shape[-1] == 3:
        return img
    else:
        raise NotImplementedError


def draw_boxes3d(img, calib, boxes3d, names=None, scores=None, color=(0, 255, 0), thickness=2):
    """
    Draw 3D boxes in the image.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        calib: kitti_calibration_utils.Calibration
        boxes3d: ndarray of float32, [N, 7], (x, y, z, l, w, h, heading] in lidar coordinates
        names: list of str, name of each object
        scores: list of float, score of each object
        color: tuple
        thickness: int

    Returns:
        img: ndarray of uint8, [H, W, 3], BGR image

    """
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.6
    f_thickness = 2
    img_h, img_w, _ = img.shape

    corners = boxes3d_to_corners3d(boxes3d)  # [N, 8, 3]
    for m in range(boxes3d.shape[0]):
        pts = corners[m]
        pts_img, pts_depth = calib.lidar_to_img(pts)

        if (pts_depth > 0).sum() < 8:
            continue

        if names is not None:
            color = box_colormap[names[m]]

        pts_img = pts_img.astype(np.int)
        cv2.line(img, (pts_img[0, 0], pts_img[0, 1]), (pts_img[5, 0], pts_img[5, 1]), color, thickness)
        cv2.line(img, (pts_img[1, 0], pts_img[1, 1]), (pts_img[4, 0], pts_img[4, 1]), color, thickness)
        
        for k in range(4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)

        if scores is not None and scores[m] > 0:
            text = '%.2f' % scores[m]
            text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            text_h += 6

            p = 4
            while p < 8:
                u, v = pts_img[p, 0], pts_img[p, 1]
                if u >= 0 and u + text_w < img_w and v - text_h >= 0 and v < img_h: break
                p += 1
            cv2.rectangle(img, (u, v), (u + text_w, v - text_h), color, -1)
            cv2.putText(img, text, (u, v - 4), f_face, f_scale, (0, 0, 0), f_thickness, cv2.LINE_AA)

    return img

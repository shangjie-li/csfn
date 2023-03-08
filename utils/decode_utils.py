import numpy as np
import torch

from utils.box_utils import boxes3d_lidar_to_camera
from utils.box_utils import boxes3d_lidar_to_image
from utils.nms_utils import nms


def decode_detections(data_dict, dataset, score_thresh, nms_thresh):
    det = {}
    batch_size = data_dict['batch_size']
    max_objs = dataset.max_objs

    for i in range(batch_size):
        frame_id = data_dict['frame_id'][i]
        img_id = int(frame_id)
        calib = dataset.get_calib(img_id)
        image_shape = dataset.get_image_shape(img_id)

        b1 = data_dict['det_boxes1'][i]
        b2 = data_dict['det_boxes2'][i]
        if b1.shape[0] > max_objs: b1 = b1[:max_objs]
        if b2.shape[0] > max_objs: b2 = b2[:max_objs]

        b1[:, -1] = data_dict['pred_scores1'][i].cpu().numpy()
        b2[:, -1] = data_dict['pred_scores2'][i].cpu().numpy()
        boxes = np.concatenate([b1, b2], axis=0)
        boxes3d_lidar, cls_ids, scores = boxes[:, :7], boxes[:, 7], boxes[:, 8]

        selected, selected_scores = nms(
            torch.from_numpy(scores).float().cuda(), torch.from_numpy(boxes3d_lidar).float().cuda(),
            score_thresh=score_thresh, nms_thresh=nms_thresh
        )
        selected = selected.cpu().numpy()
        scores = selected_scores.cpu().numpy()
        boxes3d_lidar, cls_ids = boxes3d_lidar[selected], cls_ids[selected]

        boxes3d_camera = boxes3d_lidar_to_camera(boxes3d_lidar, calib)
        boxes2d = boxes3d_lidar_to_image(boxes3d_lidar, calib, image_shape)

        locs3d = boxes3d_camera[:, 0:3]
        sizes3d = boxes3d_camera[:, 3:6]
        rys = boxes3d_camera[:, 6:7]
        alphas = -np.arctan2(locs3d[:, 0:1], locs3d[:, 2:3]) + rys
        locs3d[:, 1] += sizes3d[:, 0] / 2

        det[frame_id] = np.concatenate(
            [cls_ids.reshape(-1, 1), alphas, boxes2d, sizes3d, locs3d, rys, scores.reshape(-1, 1)],
            axis=-1
        )

    return det

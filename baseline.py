import os
import argparse
import datetime
import yaml
import tqdm
import numpy as np
import torch

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from helpers.logger_helper import create_logger
from utils.box_utils import boxes3d_lidar_to_camera
from utils.box_utils import boxes3d_lidar_to_image
from utils.nms_utils import nms


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/csfn.yaml',
                        help='path to the config file')
    parser.add_argument('--result_dir', type=str, default='outputs/data',
                        help='path to save detection results')
    parser.add_argument('--mode', type=int, default=1,
                        help='fusion mode for decoding detections')
    args = parser.parse_args()
    return args


def decode_detections(dataset, data_dict, mode):
    frame_id = data_dict['frame_id']
    img_id = int(frame_id)
    calib = dataset.get_calib(img_id)
    image_shape = dataset.get_image_shape(img_id)

    if mode == 1:
        b1 = data_dict['det_boxes1']
        b2 = data_dict['det_boxes2']
        b1 = b1[np.sqrt(b1[:, 0] ** 2 + b1[:, 1] ** 2) < 60]
        b2 = b2[np.sqrt(b2[:, 0] ** 2 + b2[:, 1] ** 2) >= 60]
        boxes = np.concatenate([b1, b2], axis=0)
        boxes3d_lidar, cls_ids, scores = boxes[:, :7], boxes[:, 7], boxes[:, 8]

    elif mode == 2:
        boxes = np.concatenate([data_dict['det_boxes1'], data_dict['det_boxes2']], axis=0)
        boxes3d_lidar, cls_ids, scores = boxes[:, :7], boxes[:, 7], boxes[:, 8]
        selected, selected_scores = nms(
            torch.from_numpy(scores).float().cuda(), torch.from_numpy(boxes3d_lidar).float().cuda(),
            score_thresh=0.1, nms_thresh=0.1
        )
        selected = selected.cpu().numpy()
        scores = selected_scores.cpu().numpy()
        boxes3d_lidar, cls_ids = boxes3d_lidar[selected], cls_ids[selected]

    else:
        raise NotImplementedError

    boxes3d_camera = boxes3d_lidar_to_camera(boxes3d_lidar, calib)
    boxes2d = boxes3d_lidar_to_image(boxes3d_lidar, calib, image_shape)

    locs3d = boxes3d_camera[:, 0:3]
    sizes3d = boxes3d_camera[:, 3:6]
    rys = boxes3d_camera[:, 6:7]
    alphas = -np.arctan2(locs3d[:, 0:1], locs3d[:, 2:3]) + rys
    locs3d[:, 1] += sizes3d[:, 0] / 2

    det = {}
    det[frame_id] = np.concatenate(
        [cls_ids.reshape(-1, 1), alphas, boxes2d, sizes3d, locs3d, rys, scores.reshape(-1, 1)],
        axis=-1
    )
    return det


def save_results(dataset, all_det, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    for img_id in all_det.keys():
        output_path = os.path.join(result_dir, '{:06d}.txt'.format(int(img_id)))
        f = open(output_path, 'w')
        objs = all_det[img_id]

        for i in range(len(objs)):
            class_name = dataset.class_names[int(objs[i][0])]
            f.write('{} 0.0 0'.format(class_name))
            for j in range(1, 14):
                f.write(' {:.2f}'.format(objs[i][j]))
            f.write('\n')

        f.close()


def main():
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=cfg['tester']['split'], is_training=False)
    else:
        raise NotImplementedError

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log_baseline_%s.txt' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    all_det = {}
    progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
    for idx, data_dict in enumerate(dataset):
        det = decode_detections(dataset, data_dict, mode=args.mode)
        all_det.update(det)

        progress_bar.update()
    progress_bar.close()

    save_results(dataset, all_det, args.result_dir)
    dataset.eval(result_dir=args.result_dir, logger=logger)


if __name__ == '__main__':
    main()

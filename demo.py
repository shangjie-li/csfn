import os
import argparse
import yaml
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from csfn import build_model
from helpers.checkpoint_helper import load_checkpoint
from dataset_player import visualize
from utils.nms_utils import nms


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/csfn.yaml',
                        help='path to the config file')
    parser.add_argument('--split', type=str, default=None,
                        help='must be chosen from ["train", "val", "trainval", "test"]')
    parser.add_argument('--score_thresh_b1', type=float, default=None,
                        help='score threshold for filtering detections from input boxes1')
    parser.add_argument('--score_thresh_b2', type=float, default=None,
                        help='score threshold for filtering detections from input boxes2')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='score threshold for filtering detections')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='NMS threshold for filtering detections')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the checkpoint')
    parser.add_argument('--sample_idx', type=str, default=None,
                        help='index of the sample')
    args = parser.parse_args()
    return args


def run(model, dataset, args, cfg, data_dict, device):
    if args.score_thresh_b1 or args.score_thresh_b2:
        img_id = int(data_dict['frame_id'])
        calib = dataset.get_calib(img_id)

        b1 = dataset.get_objects(img_id, dataset.det_dir1, calib)  # [M1, 9]
        b2 = dataset.get_objects(img_id, dataset.det_dir2, calib)  # [M2, 9]

        b1 = b1[b1[:, -1] >= args.score_thresh_b1] if args.score_thresh_b1 else b1
        b2 = b2[b2[:, -1] >= args.score_thresh_b2] if args.score_thresh_b2 else b2

        x1, mask1, _ = dataset.get_inputs(b1, b2)
        x2, mask2, _ = dataset.get_inputs(b2, b1)
        data_dict.update({
            'det_boxes1': b1,
            'det_boxes2': b2,
            'x1': x1[None, :, :].transpose(2, 0, 1),  # [num_features, 1, max_objs]
            'x2': x2[None, :, :].transpose(2, 0, 1),  # [num_features, 1, max_objs]
            'mask1': mask1,  # [max_objs]
            'mask2': mask2,  # [max_objs]
        })

    frame_id = data_dict['frame_id']
    max_objs = dataset.max_objs

    det_boxes1 = data_dict['det_boxes1']  # [M1, 9]
    det_boxes2 = data_dict['det_boxes2']  # [M1, 9]

    b1 = det_boxes1.copy()
    b2 = det_boxes2.copy()
    if b1.shape[0] > max_objs: b1 = b1[:max_objs]
    if b2.shape[0] > max_objs: b2 = b2[:max_objs]

    batch_dict = dataset.collate_batch([data_dict])
    batch_dict = dataset.load_data_to_gpu(batch_dict, device)

    batch_dict = model(batch_dict)

    b1[:, -1] = batch_dict['pred_scores1'][0].cpu().numpy()
    b2[:, -1] = batch_dict['pred_scores2'][0].cpu().numpy()
    boxes = np.concatenate([b1, b2], axis=0)
    boxes3d_lidar, cls_ids, scores = boxes[:, :7], boxes[:, 7], boxes[:, 8]

    selected, selected_scores = nms(
        torch.from_numpy(scores).float().cuda(), torch.from_numpy(boxes3d_lidar).float().cuda(),
        score_thresh=cfg['tester']['score_thresh'], nms_thresh=cfg['tester']['nms_thresh']
    )
    selected = selected.cpu().numpy()
    scores = selected_scores.cpu().numpy()
    scores = np.clip(scores, a_min=0.0, a_max=1.0)
    boxes3d_lidar, cls_ids = boxes3d_lidar[selected], cls_ids[selected]
    pred_boxes = np.concatenate([boxes3d_lidar, cls_ids.reshape(-1, 1), scores.reshape(-1, 1)], axis=-1)

    visualize(dataset, frame_id, pred_boxes, det_boxes1, det_boxes2)


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.split is not None:
        cfg['tester']['split'] = args.split
    if args.score_thresh is not None:
        cfg['tester']['score_thresh'] = args.score_thresh
    if args.nms_thresh is not None:
        cfg['tester']['nms_thresh'] = args.nms_thresh
    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=cfg['tester']['split'], is_training=False)
    else:
        raise NotImplementedError

    model = build_model(cfg['model'], num_features=dataset.num_features)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    assert os.path.exists(cfg['tester']['checkpoint'])
    load_checkpoint(
        file_name=cfg['tester']['checkpoint'],
        model=model,
        optimizer=None,
        map_location=device,
        logger=None,
    )

    torch.set_grad_enabled(False)
    model.eval()

    if args.sample_idx is not None:
        assert args.sample_idx in dataset.id_list
        i = dataset.id_list.index(args.sample_idx)
        run(model, dataset, args, cfg, dataset[i], device)
    else:
        progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
        for i in range(len(dataset)):
            run(model, dataset, args, cfg, dataset[i], device)
            progress_bar.update()
        progress_bar.close()

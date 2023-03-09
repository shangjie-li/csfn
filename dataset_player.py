import os
import argparse
import yaml
import tqdm
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from utils.opencv_vis_utils import normalize_img
from utils.opencv_vis_utils import draw_boxes3d


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/csfn.yaml',
                        help='path to the config file')
    parser.add_argument('--split', type=str, default='train',
                        help='must be chosen from ["train", "val", "trainval", "test"]')
    parser.add_argument('--sample_idx', type=str, default=None,
                        help='index of the sample')
    args = parser.parse_args()
    return args


def visualize(dataset, frame_id, gt_boxes, det_boxes1, det_boxes2):
    img_id = int(frame_id)
    calib = dataset.get_calib(img_id)
    image = dataset.get_image(img_id)[:, :, ::-1]  # BGR image
    image = normalize_img(image)

    n = 0
    img_size = (792, 240)
    box_dict = {frame_id: gt_boxes, 'det_boxes1: %s' % frame_id: det_boxes1, 'det_boxes2: %s' % frame_id: det_boxes2}
    data = {}
    for k, v in box_dict.items():
        x = image.copy()
        boxes, cls_ids, scores = v[:, :7], v[:, 7], v[:, 8]
        names = [dataset.class_names[int(idx)] for idx in cls_ids]
        x = draw_boxes3d(x, calib, boxes, names, scores)
        x = cv2.resize(x, img_size)
        data.update({k: x})

        cv2.namedWindow(k, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(k, img_size[0] + 100, img_size[1] + 100)
        cv2.imshow(k, x)
        cv2.moveWindow(k, 150, n * (img_size[1] + 100) + 10)
        n += 1

    key = cv2.waitKey(0)
    while key:
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            break
        elif key == 13:  # Enter
            for k, v in data.items():
                cv2.imwrite(k + '.png', v)
            cv2.destroyAllWindows()
            break
        else:
            key = cv2.waitKey(0)


def run(dataset, data_dict):
    frame_id = data_dict['frame_id']

    b1 = data_dict['det_boxes1']  # [M1, 9]
    b2 = data_dict['det_boxes2']  # [M2, 9]
    gt = data_dict['gt_boxes'] if data_dict.get('gt_boxes') is not None else None

    visualize(dataset, frame_id, gt, b1, b2)


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=args.split, is_training=False)
    else:
        raise NotImplementedError

    if args.sample_idx is not None:
        assert args.sample_idx in dataset.id_list
        i = dataset.id_list.index(args.sample_idx)
        run(dataset, dataset[i])
    else:
        progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
        for i in range(len(dataset)):
            run(dataset, dataset[i])
            progress_bar.update()
        progress_bar.close()

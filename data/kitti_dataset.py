import os
from skimage import io
from collections import defaultdict
import numpy as np
import torch

from data.kitti_object_eval_python.kitti_common import get_label_annos
from data.kitti_object_eval_python.eval import get_official_eval_result
from utils.kitti_calibration_utils import parse_calib
from utils.kitti_object3d_utils import parse_objects
from ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu


class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, is_training=True):
        self.root_dir = 'data/kitti'
        self.split = split
        self.is_training = is_training

        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.id_list = [x.strip() for x in open(self.split_file).readlines()]

        self.data_dir = os.path.join(self.root_dir, 'testing' if self.split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.velodyne_dir = os.path.join(self.data_dir, 'velodyne')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.det_dir1 = os.path.join('data/detected_objects', cfg['det_dir1'], self.split)
        self.det_dir2 = os.path.join('data/detected_objects', cfg['det_dir2'], self.split)

        self.class_names = cfg['class_names']
        self.cls_to_id = {}
        for i, name in enumerate(self.class_names):
            self.cls_to_id[name] = i
        self.write_list = cfg['write_list']
        self.max_objs = 30
        self.num_features = 4  # [bev_iou, ego_score, ref_score, distance]
        self.bev_iou_thresh = 0.25

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return io.imread(img_file).astype(np.float32) / 255.0  # ndarray of float32, [H, W, 3], RGB image

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)  # ndarray of int, [2], H and W

    def get_points(self, idx):
        pts_file = os.path.join(self.velodyne_dir, '%06d.bin' % idx)
        assert os.path.exists(pts_file)
        return np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)  # ndarray of float32, [N, 4], (x, y, z, i)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return parse_calib(calib_file)  # kitti_calibration_utils.Calibration

    def get_objects(self, idx, root_path, calib):
        label_file = os.path.join(root_path, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        obj_list = parse_objects(label_file)
        if len(obj_list) == 0: return np.array([]).reshape(-1, 9)

        names = np.array([obj.cls_type for obj in obj_list])
        sizes = np.array([[obj.h, obj.w, obj.l] for obj in obj_list])
        locs = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        rys = np.array([obj.ry for obj in obj_list])
        scores = np.array([obj.score for obj in obj_list])

        keep_indices = []  # make sure that objects of the same category are grouped together
        for cur_name in self.class_names:
            for i, x in enumerate(names):
                if x == cur_name:
                    keep_indices.append(i)

        names = names[keep_indices]
        sizes = sizes[keep_indices]
        locs = locs[keep_indices]
        rys = rys[keep_indices]
        scores = scores[keep_indices]

        locs_lidar = calib.rect_to_lidar(locs)
        hs, ws, ls = sizes[:, 0:1], sizes[:, 1:2], sizes[:, 2:3]
        locs_lidar[:, 2] += hs[:, 0] / 2
        boxes_lidar = np.concatenate(
            [locs_lidar, ls, ws, hs, -(np.pi / 2 + rys[..., np.newaxis])], axis=1
        )  # [M, 7], (x, y, z, l, w, h, heading) in lidar coordinates

        cls_ids = np.array([self.cls_to_id[i] for i in names]).astype(np.float32)
        boxes_lidar = np.concatenate(
            [boxes_lidar, cls_ids.reshape(-1, 1), scores.reshape(-1, 1)], axis=1
        )  # [M, 9]

        return boxes_lidar

    def eval(self, result_dir, logger):
        logger.info('==> Loading detections and ground truths...')
        img_ids = [int(idx) for idx in self.id_list]
        dt_annos = get_label_annos(result_dir)
        gt_annos = get_label_annos(self.label_dir, img_ids)
        logger.info('==> Done.')

        logger.info('==> Evaluating...')
        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        for category in self.write_list:
            result_str = get_official_eval_result(
                gt_annos, dt_annos, test_id[category], use_ldf_eval=True, print_info=False)
            logger.info(result_str)

    @staticmethod
    def get_matching_matrix(iou, iou_thresh):
        num_e, num_r = iou.shape[0], iou.shape[1]

        best_iou_per_e = np.max(iou, axis=1)  # [num_e]
        best_r_ind_per_e = np.argmax(iou, axis=1)  # [num_e]
        best_iou_per_r = np.max(iou, axis=0)  # [num_r]
        best_e_ind_per_r = np.argmax(iou, axis=0)  # [num_r]

        m1 = np.zeros_like(iou)
        for i in range(num_e):
            if best_iou_per_e[i] > iou_thresh:
                m1[i, best_r_ind_per_e[i]] = 1

        m2 = np.zeros_like(iou)
        for j in range(num_r):
            if best_iou_per_r[j] > iou_thresh:
                m2[best_e_ind_per_r[j], j] = 1

        return m1 * m2  # [num_e, num_r]

    def get_inputs(self, ego_boxes, ref_boxes):
        ego_boxes, ego_cls_ids, ego_scores = ego_boxes[:, :7], ego_boxes[:, 7], ego_boxes[:, 8]
        ref_boxes, ref_cls_ids, ref_scores = ref_boxes[:, :7], ref_boxes[:, 7], ref_boxes[:, 8]

        all_inputs = []
        all_assigned_ref_boxes = []
        for name in self.class_names:
            cls_id = self.cls_to_id[name]
            mask = ego_cls_ids == cls_id
            e_boxes, e_scores = ego_boxes[mask], ego_scores[mask]
            mask = ref_cls_ids == cls_id
            r_boxes, r_scores = ref_boxes[mask], ref_scores[mask]

            num_e, num_r = e_boxes.shape[0], r_boxes.shape[0]
            inputs = np.zeros((num_e, self.num_features), dtype=np.float32)
            assigned_ref_boxes = np.zeros((num_e, 7), dtype=np.float32)
            inputs[:, 1] = e_scores
            inputs[:, 3] = np.sqrt(e_boxes[:, 0] ** 2 + e_boxes[:, 1] ** 2) * 0.01
            if num_e == 0 or num_r == 0:
                all_inputs.append(inputs)
                all_assigned_ref_boxes.append(assigned_ref_boxes)
                continue

            bev_iou = boxes_bev_iou_cpu(e_boxes, r_boxes)  # [num_e, num_r]
            m = self.get_matching_matrix(bev_iou, self.bev_iou_thresh)
            for i in range(num_e):
                if m[i].sum() > 0:
                    assigned_r_ind = np.where(m[i] > 0)[0][0]
                    inputs[i][0] = bev_iou[i][assigned_r_ind]
                    inputs[i][2] = r_scores[assigned_r_ind]
                    assigned_ref_boxes[i] = r_boxes[assigned_r_ind]
            all_inputs.append(inputs)
            all_assigned_ref_boxes.append(assigned_ref_boxes)

        all_inputs = np.concatenate(all_inputs, axis=0)  # [num_objs, num_features]
        all_assigned_ref_boxes = np.concatenate(all_assigned_ref_boxes, axis=0)  # [num_objs, 7]
        if all_inputs.shape[0] > self.max_objs:
            x = all_inputs[:self.max_objs]
            mask = np.ones((self.max_objs,), dtype=np.float32)
        else:
            x = np.zeros((self.max_objs, self.num_features), dtype=np.float32)
            x[:all_inputs.__len__()] = all_inputs
            mask = np.zeros((self.max_objs,), dtype=np.float32)
            mask[:all_inputs.__len__()] = 1.0

        return x, mask, all_assigned_ref_boxes

    def get_targets(self, ego_boxes, gt_boxes, all_assigned_ref_boxes):
        ego_boxes, ego_cls_ids, ego_scores = ego_boxes[:, :7], ego_boxes[:, 7], ego_boxes[:, 8]
        gt_boxes, gt_cls_ids = gt_boxes[:, :7], gt_boxes[:, 7]

        all_targets = []
        for name in self.class_names:
            cls_id = self.cls_to_id[name]
            mask = ego_cls_ids == cls_id
            e_boxes, e_scores, r_boxes = ego_boxes[mask], ego_scores[mask], all_assigned_ref_boxes[mask]
            mask = gt_cls_ids == cls_id
            g_boxes = gt_boxes[mask]

            num_e, num_g = e_boxes.shape[0], g_boxes.shape[0]
            targets = e_scores.copy()
            if num_e == 0 or num_g == 0:
                all_targets.append(targets)
                continue

            bev_iou = boxes_bev_iou_cpu(e_boxes, g_boxes)  # [num_e, num_g]
            m = self.get_matching_matrix(bev_iou, self.bev_iou_thresh)
            for i in range(num_e):
                if m[i].sum() > 0:
                    assigned_g_ind = np.where(m[i] > 0)[0][0]
                    x = boxes_bev_iou_cpu(r_boxes[i].reshape(-1, 7), g_boxes[assigned_g_ind].reshape(-1, 7))[0][0]
                    if x >= self.bev_iou_thresh:
                        if x > bev_iou[i][assigned_g_ind]:
                            targets[i] = 0  # if the ref model has higher iou, suppress the ego model's result
                        else:
                            targets[i] += 0.2  # otherwise, improve the confidence of the ego model's result
            all_targets.append(targets)

        all_targets = np.concatenate(all_targets, axis=0)  # [num_objs]
        if all_targets.shape[0] > self.max_objs:
            y = all_targets[:self.max_objs]
        else:
            y = np.zeros((self.max_objs,), dtype=np.float32)
            y[:all_targets.__len__()] = all_targets

        return y

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_id = int(self.id_list[idx])
        calib = self.get_calib(img_id)

        b1 = self.get_objects(img_id, self.det_dir1, calib)  # [M1, 9]
        b2 = self.get_objects(img_id, self.det_dir2, calib)  # [M2, 9]
        x1, mask1, assigned_ref_boxes1 = self.get_inputs(b1, b2)
        x2, mask2, assigned_ref_boxes2 = self.get_inputs(b2, b1)
        data_dict = {
            'frame_id': self.id_list[idx],
            'det_boxes1': b1,
            'det_boxes2': b2,
            'x1': x1[None, :, :].transpose(2, 0, 1),  # [num_features, 1, max_objs]
            'x2': x2[None, :, :].transpose(2, 0, 1),  # [num_features, 1, max_objs]
            'mask1': mask1,  # [max_objs]
            'mask2': mask2,  # [max_objs]
        }

        if self.is_training:
            if mask1.sum() + mask2.sum() == 0:
                return self.__getitem__(np.random.randint(self.__len__()))

        if self.split != 'test':
            gt = self.get_objects(img_id, self.label_dir, calib)  # [M3, 9]
            y1 = self.get_targets(b1, gt, assigned_ref_boxes1)
            y2 = self.get_targets(b2, gt, assigned_ref_boxes2)
            data_dict.update({
                'gt_boxes': gt,
                'y1': y1,  # [max_objs]
                'y2': y2,  # [max_objs]
            })

        return data_dict

    @staticmethod
    def collate_batch(batch_list):
        temp_dict = defaultdict(list)
        for data_dict in batch_list:
            for key, val in data_dict.items():
                temp_dict[key].append(val)

        batch_size = len(batch_list)
        batch_dict = {}
        batch_dict['batch_size'] = batch_size

        for key, val in temp_dict.items():
            if key in ['x1', 'x2', 'mask1', 'mask2', 'y1', 'y2']:
                batch_dict[key] = np.stack(val, axis=0)
            else:
                batch_dict[key] = val

        return batch_dict

    @staticmethod
    def load_data_to_gpu(batch_dict, device):
        for key, val in batch_dict.items():
            if key in ['x1', 'x2', 'mask1', 'mask2', 'y1', 'y2']:
                batch_dict[key] = torch.from_numpy(val).float().to(device)

        return batch_dict

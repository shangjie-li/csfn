import os
import argparse
import yaml
from pathlib import Path
import time
import math
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from csfn import build_model
from helpers.checkpoint_helper import load_checkpoint
from utils.kitti_calibration_utils import parse_calib
from utils.opencv_vis_utils import box_colormap
from utils.opencv_vis_utils import normalize_img
from utils.opencv_vis_utils import draw_boxes3d
from utils.nms_utils import nms


image_lock = threading.Lock()
marker_lock1 = threading.Lock()
marker_lock2 = threading.Lock()


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/csfn.yaml',
                        help='path to the config file')
    parser.add_argument('--display', action='store_true', default=False,
                        help='whether to show the RGB image')
    parser.add_argument('--print', action='store_true', default=False,
                        help='whether to print results in the txt file')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='score threshold for filtering detections')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='NMS threshold for filtering detections')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the checkpoint')
    parser.add_argument('--current_classes', type=str, default='0,1,2',
                        help='a filter for desired classes, e.g., 0,1,2 (split by a comma)')
    parser.add_argument('--sub_image', type=str, default='/kitti/camera_color_left/image_raw',
                        help='image topic to subscribe')
    parser.add_argument('--sub_marker1', type=str, default='/det_boxes1',
                        help='marker topic to subscribe')
    parser.add_argument('--sub_marker2', type=str, default='/det_boxes2',
                        help='marker topic to subscribe')
    parser.add_argument('--pub_marker', type=str, default='/result',
                        help='marker topic to publish')
    parser.add_argument('--frame_rate', type=int, default=10,
                        help='working frequency')
    parser.add_argument('--frame_id', type=str, default='velo_link',
                        help='frame id for ROS message')
    parser.add_argument('--calib_file', type=str, default='data/kitti/testing/calib/000000.txt',
                        help='path to the calibration file')
    args = parser.parse_args()
    return args


def publish_marker_msg(pub, boxes, labels, scores, header, frame_rate, color_map):
    marker_array = MarkerArray()
    for i in range(boxes.shape[0]):
        marker = Marker()
        marker.header = header
        marker.ns = labels[i]
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = boxes[i][0]
        marker.pose.position.y = boxes[i][1]
        marker.pose.position.z = boxes[i][2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(0.5 * boxes[i][6])
        marker.pose.orientation.w = math.cos(0.5 * boxes[i][6])
        
        marker.scale.x = boxes[i][3]
        marker.scale.y = boxes[i][4]
        marker.scale.z = boxes[i][5]
    
        marker.color.r = color_map[labels[i]][2] / 255.0
        marker.color.g = color_map[labels[i]][1] / 255.0
        marker.color.b = color_map[labels[i]][0] / 255.0
        marker.color.a = scores[i]  # 0 for invisible
        
        marker.lifetime = rospy.Duration(1 / frame_rate)
        marker_array.markers.append(marker)
        
    pub.publish(marker_array)


def display(img, v_writer, win_name='result'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True


def print_info(frame, stamp, delay, boxes, labels, scores, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        box = boxes[i]
        info_str = 'box:%.2f %.2f %.2f %.2f %.2f %.2f %.2f  score:%.2f  label:%s' % (
            box[0], box[1], box[2], box[3], box[4], box[5], box[6], scores[i], labels[i]
        )
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')


def create_input_data(dataset, b1, b2, frame_id):
    x1, mask1, assigned_ref_boxes1 = dataset.get_inputs(b1, b2)
    x2, mask2, assigned_ref_boxes2 = dataset.get_inputs(b2, b1)
    data_dict = {
        'frame_id': frame_id,
        'det_boxes1': b1,
        'det_boxes2': b2,
        'x1': x1[None, :, :].transpose(2, 0, 1),  # [num_features, 1, max_objs]
        'x2': x2[None, :, :].transpose(2, 0, 1),  # [num_features, 1, max_objs]
        'mask1': mask1,  # [max_objs]
        'mask2': mask2,  # [max_objs]
    }

    return data_dict


def get_boxes_from_marker_array(marker_array):
    num_objs = len(marker_array.markers)
    boxes = []
    for i in range(num_objs):
        marker = marker_array.markers[i]
        label = marker.ns
        if label not in dataset.class_names: continue

        x = marker.pose.position.x
        y = marker.pose.position.y
        z = marker.pose.position.z
        l = marker.scale.x
        w = marker.scale.y
        h = marker.scale.z
        phi = np.arctan2(marker.pose.orientation.z, marker.pose.orientation.w) * 2

        cls_id = dataset.cls_to_id[label]
        score = marker.color.a
        boxes.append([x, y, z, l, w, h, phi, cls_id, score])

    return np.array(boxes, dtype=np.float32).reshape(-1, 9)


def image_callback(image):
    global image_header_list, image_frame_list
    image_lock.acquire()
    cur_header = image.header
    cur_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)  # ndarray of uint8, [H, W, 3], BGR image

    if len(image_header_list) == keep_topics:
        image_header_list.pop(0)
        image_frame_list.pop(0)
    image_header_list.append(cur_header)
    image_frame_list.append(cur_frame)

    assert len(image_header_list) <= keep_topics
    assert len(image_header_list) == len(image_frame_list)
    image_lock.release()


def marker_callback1(marker_array):
    global header1_list, det1_list
    marker_lock1.acquire()
    cur_frame = get_boxes_from_marker_array(marker_array)

    if len(marker_array.markers) > 0:
        cur_header = marker_array.markers[0].header
    else:
        cur_header = Header()
        cur_header.frame_id = args.frame_id
        cur_header.stamp = rospy.Time.now()

    if len(header1_list) == keep_topics:
        header1_list.pop(0)
        det1_list.pop(0)
    header1_list.append(cur_header)
    det1_list.append(cur_frame)

    assert len(header1_list) <= keep_topics
    assert len(header1_list) == len(det1_list)
    marker_lock1.release()


def marker_callback2(marker_array):
    global header2_list, det2_list
    marker_lock2.acquire()
    cur_frame = get_boxes_from_marker_array(marker_array)

    if len(marker_array.markers) > 0:
        cur_header = marker_array.markers[0].header
    else:
        cur_header = Header()
        cur_header.frame_id = args.frame_id
        cur_header.stamp = rospy.Time.now()

    if len(header2_list) == keep_topics:
        header2_list.pop(0)
        det2_list.pop(0)
    header2_list.append(cur_header)
    det2_list.append(cur_frame)

    assert len(header2_list) <= keep_topics
    assert len(header2_list) == len(det2_list)
    marker_lock2.release()


def get_normalized_time(stamp):
    return stamp.secs + 0.000000001 * stamp.nsecs


def find_min_error(ego_time, ref_times):
    x = np.abs(np.array(ref_times) - ego_time)
    return x.min(), x.argmin()


def select_topic_idx(ego_headers, headers1, headers2):
    times1 = np.array([get_normalized_time(h.stamp) for h in headers1])
    times2 = np.array([get_normalized_time(h.stamp) for h in headers2])

    for i, h in enumerate(ego_headers):
        cur_time = get_normalized_time(h.stamp)
        m1, argm1 = find_min_error(cur_time, times1)
        m2, argm2 = find_min_error(cur_time, times2)
        if m1 == 0 and m2 == 0:
            return i, argm1, argm2

    cur_time = get_normalized_time(ego_headers[0].stamp)
    m1, argm1 = find_min_error(cur_time, times1)
    m2, argm2 = find_min_error(cur_time, times2)
    return 0, argm1, argm2


def timer_callback(event):
    cur_header = Header()

    image_lock.acquire()
    marker_lock1.acquire()
    marker_lock2.acquire()

    image_idx, idx1, idx2 = select_topic_idx(image_header_list, header1_list, header2_list)
    cur_header.frame_id = header1_list[idx1].frame_id
    cur_header.stamp = header1_list[idx1].stamp
    cur_image = image_frame_list[image_idx].copy()
    b1 = det1_list[idx1].copy()
    b2 = det2_list[idx2].copy()

    image_lock.release()
    marker_lock1.release()
    marker_lock2.release()

    global frame
    frame += 1
    start = time.time()

    data_dict = create_input_data(dataset, b1, b2, args.frame_id)

    batch_dict = dataset.collate_batch([data_dict])
    batch_dict = dataset.load_data_to_gpu(batch_dict, device)

    with torch.no_grad():
        batch_dict = model(batch_dict)

    max_objs = dataset.max_objs
    b1 = batch_dict['det_boxes1'][0]
    b2 = batch_dict['det_boxes2'][0]
    if b1.shape[0] > max_objs: b1 = b1[:max_objs]
    if b2.shape[0] > max_objs: b2 = b2[:max_objs]

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

    indices = [i for i, cls_id in enumerate(cls_ids) if cls_id in args.current_classes]
    boxes3d_lidar, cls_ids, scores = boxes3d_lidar[indices], cls_ids[indices], scores[indices]
    names = [dataset.class_names[int(k)] for k in cls_ids]

    publish_marker_msg(pub_marker, boxes3d_lidar, names, scores, cur_header, args.frame_rate, box_colormap)

    if args.display:
        image = normalize_img(cur_image)
        image = draw_boxes3d(image, calib, boxes3d_lidar, names, scores)
        if not display(image, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")

    if args.print:
        stamp = rospy.Time.now()
        cur_stamp = get_normalized_time(stamp)
        delay = round(time.time() - start, 3)
        print_info(frame, cur_stamp, delay, boxes3d_lidar, names, scores, result_file)


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.score_thresh is not None:
        cfg['tester']['score_thresh'] = args.score_thresh
    if args.nms_thresh is not None:
        cfg['tester']['nms_thresh'] = args.nms_thresh
    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    args.current_classes = list(map(int, args.current_classes.split(',')))

    rospy.init_node("csfn", anonymous=True, disable_signals=True)
    frame = 0

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

    calib_file = Path(args.calib_file)
    assert os.path.exists(calib_file)
    calib = parse_calib(calib_file)

    keep_topics = 10

    image_header_list, image_frame_list = [], []
    rospy.Subscriber(args.sub_image, Image, image_callback, queue_size=1, buff_size=52428800)

    header1_list, det1_list = [], []
    header2_list, det2_list = [], []
    rospy.Subscriber(args.sub_marker1, MarkerArray, marker_callback1, queue_size=1, buff_size=52428800)
    rospy.Subscriber(args.sub_marker2, MarkerArray, marker_callback2, queue_size=1, buff_size=52428800)
    print('==> Waiting for topic %s, %s and %s...' % (args.sub_image, args.sub_marker1, args.sub_marker2))
    while len(image_header_list) < keep_topics or len(header1_list) < keep_topics or len(header2_list) < keep_topics:
        time.sleep(0.1)
    print('==> Done.')

    if args.display:
        win_h, win_w = image_frame_list[0].shape[0], image_frame_list[0].shape[1]
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)

    if args.print:
        result_file = 'result.txt'
        with open(result_file, 'w') as fob:
            fob.seek(0)
            fob.truncate()

    pub_marker = rospy.Publisher(args.pub_marker, MarkerArray, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)

    rospy.spin()

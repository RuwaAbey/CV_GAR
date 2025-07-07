import pickle
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmengine.registry import init_default_scope
from mmpose.evaluation.functional import nms
from collections import Counter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collective_detection_keypoints_ordered.log'),
        logging.StreamHandler()
    ]
)

# Dataset constants
FRAMES_NUM = {1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302,
              11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342,
              21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356,
              31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401,
              41: 707, 42: 420, 43: 410, 44: 356}

ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking']
ACTIONS_ID = {a: i for i, a in enumerate(ACTIONS)}
MAX_PEOPLE = 13  # Maximum number of people per frame
NUM_FRAMES = 10  # Number of frames per bunch (1 to 10, 11 to 20, etc.)
KEYPOINT_SCORE_THRESHOLD = 0.3  # Threshold for keypoint confidence scores
NUM_KEYPOINTS = 17  # Number of COCO keypoints
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Configuration
DET_CONFIG = '/home/akila17/e19-group-activity/Group_Activity/Datasets/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
DET_CHECKPOINT = '/home/akila17/e19-group-activity/Group_Activity/Datasets/Skeleton_Data/faster_rcnn_r50_fpn_1x_coco.pth'
POSE_CONFIG = '/home/akila17/e19-group-activity/Group_Activity/Datasets/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
POSE_CHECKPOINT = '/home/akila17/e19-group-activity/Group_Activity/Datasets/Skeleton_Data/hrnet_w32_coco_256x192.pth'
DATASET_PATH = '../Group_Activity/Datasets/collective_dataset'  # Update to your dataset path
SAVE_PATH = '/home/akila17/e19-group-activity/CAD'


def read_annotations(path, sid):
    """
    Read annotations for a given sequence and return a dictionary with actions and group activity.
    """
    annotations = {}
    path = f"{path}/seq{sid:02d}/annotations.txt"
    
    with open(path, mode='r') as f:
        frame_id = None
        group_activity = None
        actions = []
        
        for line in f.readlines():
            values = line.strip().split('\t')
            current_frame = int(values[0])
            
            if current_frame != frame_id:
                if frame_id is not None and frame_id % 10 == 1 and frame_id + 9 <= FRAMES_NUM[sid]:
                    counter = Counter(actions).most_common(2)
                    group_activity = counter[0][0] - 1 if counter[0][0] != 0 else counter[1][0] - 1
                    annotations[frame_id] = {
                        'actions': actions,
                        'group_activity': group_activity
                    }
                
                frame_id = current_frame
                group_activity = None
                actions = []
            
            # Action ID (Class ID) is in values[5], zero-based indexing
            actions.append(int(values[5]) - 1)
        
        # Process the last frame
        if frame_id is not None and frame_id % 10 == 1 and frame_id + 9 <= FRAMES_NUM[sid]:
            counter = Counter(actions).most_common(2)
            group_activity = counter[0][0] - 1 if counter[0][0] != 0 else counter[1][0] - 1
            annotations[frame_id] = {
                'actions': actions,
                'group_activity': group_activity
            }
    
    return annotations

def process_frame(img_path, detector, pose_estimator):
    """
    Process a single frame to detect bounding boxes and extract keypoints.
    Returns bounding boxes in (x1, y1, x2, y2) format, keypoints, and scores.
    """
    if not os.path.exists(img_path):
        logging.warning(f"Image not found at {img_path}")
        return {
            'boxes': np.zeros((MAX_PEOPLE, 4)),
            'keypoints': np.zeros((MAX_PEOPLE, NUM_KEYPOINTS, 2)),
            'keypoint_scores': np.zeros((MAX_PEOPLE, NUM_KEYPOINTS))
        }
    
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Object detection
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    
    detect_result = inference_detector(detector, image_rgb)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    det_bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    det_bboxes = det_bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
    det_bboxes = det_bboxes[nms(det_bboxes, 0.3)][:, :4]  # (x1, y1, x2, y2)
    
    # Pose estimation
    pose_results = inference_topdown(pose_estimator, image_rgb, det_bboxes)
    all_keypoints = []
    all_scores = []
    for pose_result in pose_results:
        keypoints = pose_result.pred_instances.keypoints.squeeze(0) if pose_result.pred_instances.keypoints.shape[0] == 1 else pose_result.pred_instances.keypoints
        scores = pose_result.pred_instances.keypoint_scores.squeeze(0) if pose_result.pred_instances.keypoint_scores.shape[0] == 1 else pose_result.pred_instances.keypoint_scores
        mask = scores >= KEYPOINT_SCORE_THRESHOLD
        keypoints[~mask] = 0
        scores[~mask] = 0
        all_keypoints.append(keypoints)
        all_scores.append(scores)
    
    if len(all_keypoints) > 0:
        all_keypoints = np.stack(all_keypoints, axis=0)
        all_scores = np.stack(all_scores, axis=0)
    else:
        all_keypoints = np.zeros((0, NUM_KEYPOINTS, 2))
        all_scores = np.zeros((0, NUM_KEYPOINTS))
    
    # Pad or truncate to MAX_PEOPLE
    M = len(det_bboxes)
    if M > MAX_PEOPLE:
        logging.warning(f"Truncating {M} people to {MAX_PEOPLE} in {img_path}")
        det_bboxes = det_bboxes[:MAX_PEOPLE]
        all_keypoints = all_keypoints[:MAX_PEOPLE]
        all_scores = all_scores[:MAX_PEOPLE]
        M = MAX_PEOPLE
    
    if M < MAX_PEOPLE:
        pad_size = MAX_PEOPLE - M
        boxes_pad = np.zeros((pad_size, 4))
        det_bboxes = np.concatenate([det_bboxes, boxes_pad], axis=0) if det_bboxes.size > 0 else boxes_pad
        keypoints_pad = np.zeros((pad_size, NUM_KEYPOINTS, 2))
        all_keypoints = np.concatenate([all_keypoints, keypoints_pad], axis=0) if M > 0 else keypoints_pad
        scores_pad = np.zeros((pad_size, NUM_KEYPOINTS))
        all_scores = np.concatenate([all_scores, scores_pad], axis=0) if M > 0 else scores_pad
    
    return {
        'boxes': det_bboxes,
        'keypoints': all_keypoints,
        'keypoint_scores': all_scores
    }

def main():
    # Initialize models
    detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)
    pose_estimator = init_pose_estimator(
        POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    )
    logging.info("Detector and pose estimator initialized successfully.")

    # Load annotations for all sequences in ascending order
    seqs = sorted(range(1, 45))  # Ensure ascending order: seq01, seq02, ..., seq44
    annotations = {f'seq{sid:02d}': read_annotations(DATASET_PATH, sid) for sid in seqs}

    # Track missing frames
    missing_frames_summary = {}
    total_frames_expected = 0
    total_frames_processed = 0
    total_frames_missing = 0

    # Process each sequence in ascending order
    final_annotations = {}
    for seq_id_str in tqdm(sorted(annotations.keys()), desc="Processing sequences"):  # Sort sequence keys
        sid = int(seq_id_str[3:])  # Extract sequence ID (e.g., 'seq01' -> 1)
        missing_frames_summary[seq_id_str] = {'missing_frames': [], 'total_frames': 0, 'processed_frames': 0}
        final_annotations[seq_id_str] = {}
        
        # Process frames in ascending order
        for frame_id in sorted(annotations[seq_id_str].keys()):  # Sort frame IDs
            # Define the 10-frame bunch (e.g., frame_id to frame_id + 9)
            expected_frame_ids = list(range(frame_id, frame_id + NUM_FRAMES))
            
            # Check which frames exist
            frame_ids = []
            for fid in expected_frame_ids:
                frame_path = os.path.join(DATASET_PATH, f'seq{sid:02d}', f'frame{fid:04d}.jpg')
                if os.path.exists(frame_path):
                    frame_ids.append(fid)
                else:
                    logging.warning(f"Frame {fid} not found for sequence {seq_id_str}, annotated frame {frame_id}")
            
            missing_frames = [fid for fid in expected_frame_ids if fid not in frame_ids]
            if missing_frames:
                logging.warning(f"Sequence {seq_id_str}, Frame {frame_id}: Missing frames {missing_frames}")
                missing_frames_summary[seq_id_str]['missing_frames'].append((frame_id, missing_frames))
            
            missing_frames_summary[seq_id_str]['total_frames'] += len(expected_frame_ids)
            missing_frames_summary[seq_id_str]['processed_frames'] += len(frame_ids)
            total_frames_expected += len(expected_frame_ids)
            total_frames_processed += len(frame_ids)
            total_frames_missing += len(missing_frames)
            
            if not frame_ids:
                logging.warning(f"No valid frames found for sequence {seq_id_str} around frame {frame_id}")
                continue
            
            # Process each frame in the bunch in ascending order
            frame_bboxes = []
            frame_keypoints = []
            frame_keypoint_scores = []
            for fid in expected_frame_ids:  # Frames are in ascending order
                frame_path = os.path.join(DATASET_PATH, f'seq{sid:02d}', f'frame{fid:04d}.jpg')
                result = process_frame(frame_path, detector, pose_estimator)
                frame_bboxes.append(result['boxes'].tolist())
                frame_keypoints.append(result['keypoints'].tolist())
                frame_keypoint_scores.append(result['keypoint_scores'].tolist())
            
            # Store annotations
            final_annotations[seq_id_str][frame_id] = {
                'bounding_boxes': frame_bboxes,
                'keypoints_coordinates': frame_keypoints,
                'keypoint_scores': frame_keypoint_scores,
                'actions': annotations[seq_id_str][frame_id]['actions'],
                'group_activity': annotations[seq_id_str][frame_id]['group_activity']
            }
        
        # Summarize missing frames per sequence
        total_missing = sum(len(fids) for _, fids in missing_frames_summary[seq_id_str]['missing_frames'])
        logging.info(f"Sequence {seq_id_str}: {total_missing} missing frames out of {missing_frames_summary[seq_id_str]['total_frames']}, processed {missing_frames_summary[seq_id_str]['processed_frames']}")
    
    # Final summary
    logging.info(f"\nSummary of Missing Frames:")
    logging.info(f"Total frames expected: {total_frames_expected}")
    logging.info(f"Total frames processed: {total_frames_processed}")
    logging.info(f"Total frames missing: {total_frames_missing}")

    # Save to .pkl file
    with open(os.path.join(SAVE_PATH, 'collective_detected_bboxes_keypoints_ordered.pkl'), 'wb') as f:
        pickle.dump(final_annotations, f)
    logging.info("collective_detected_bboxes_keypoints_ordered.pkl file has been created successfully.")

if __name__ == "__main__":
    main()

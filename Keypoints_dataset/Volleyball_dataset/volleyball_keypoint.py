import os
import pickle
import numpy as np
import torch
from mmcv import imread
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from typing import List, Optional

# Setup MMPose models
def setup_models(pose_config, pose_checkpoint, device='cuda:0'):
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    pose_estimator = init_pose_estimator(
        pose_config, pose_checkpoint, device=device, cfg_options=cfg_options
    )

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    return pose_estimator, visualizer

# Get 5 frames before, 1 middle, 4 after
def get_frame_window(video_dir: str, target_frame: int) -> Optional[List[str]]:
    frame_ids = list(range(target_frame - 5, target_frame + 5))
    frame_paths = []

    for frame_id in frame_ids:
        frame_path = os.path.join(video_dir, f"{frame_id}.jpg")
        if not os.path.exists(frame_path):
            print(f"Missing frame: {frame_path}")
            return None
        frame_paths.append(frame_path)

    return frame_paths

# Load bounding boxes from pkl for a frame window
def get_bounding_boxes_per_frame(pkl_data: dict, video_id: int, target_frame: int) -> Optional[dict]:
    if video_id not in pkl_data:
        print(f"Video ID {video_id} not in PKL data.")
        return None

    frame_range = list(range(target_frame - 5, target_frame + 5))
    bboxes_per_frame = {fid: np.empty((0, 4), dtype=np.int32) for fid in frame_range}

    for middle_frame in pkl_data[video_id]:
        for fid in frame_range:
            if fid in pkl_data[video_id][middle_frame]:
                current_boxes = pkl_data[video_id][middle_frame][fid]
                if bboxes_per_frame[fid].size == 0:
                    bboxes_per_frame[fid] = current_boxes
                else:
                    bboxes_per_frame[fid] = np.vstack((bboxes_per_frame[fid], current_boxes))

    return bboxes_per_frame

# Extract and pad keypoints for a frame
def extract_keypoints(pose_results, num_people=12) -> np.ndarray:
    all_keypoints = []

    for result in pose_results:
        keypoints = result.pred_instances.keypoints  # (N, num_kpts, 2)
        scores = result.pred_instances.keypoint_scores  # (N, num_kpts)

        if keypoints.ndim == 2:
            keypoints = keypoints[None]
            scores = scores[None]

        keypoints_with_scores = np.concatenate([keypoints, scores[..., None]], axis=-1)  # (N, num_kpts, 3)
        all_keypoints.append(keypoints_with_scores)

    if all_keypoints:
        keypoints = np.vstack(all_keypoints)  # (total_people, num_kpts, 3)
    else:
        keypoints = np.zeros((0, 17, 3), dtype=np.float32)

    # Limit and pad to num_people
    keypoints = keypoints[:num_people]
    if keypoints.shape[0] < num_people:
        padding = np.zeros((num_people - keypoints.shape[0], keypoints.shape[1], 3), dtype=np.float32)
        keypoints = np.concatenate([keypoints, padding], axis=0)

    return keypoints

# Main function
def main():
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset_path = "../../Group_Activity/Datasets/volleyball_dataset"
    pkl_path = "volleyball_bboxes.pkl"
    output_pkl_path = "volleyball_keypoints.pkl"

    pose_config  = '/home/akila17/e19-group-activity/Group_Activity/Datasets/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    pose_checkpoint = '/home/akila17/e19-group-activity/Group_Activity/Datasets/Skeleton_Data/hrnet_w32_coco_256x192.pth'

    # Load bounding box annotations
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    # Initialize model
    pose_estimator, _ = setup_models(pose_config, pose_checkpoint, device=DEVICE)

    # Final keypoint storage dict
    keypoint_dict = {}

    for video_id in range(1):
        video_base_path = os.path.join(dataset_path, str(video_id))
        if not os.path.isdir(video_base_path):
            continue

        keypoint_dict[video_id] = {}

        for target_frame_str in sorted(os.listdir(video_base_path), key=lambda x: int(x) if x.isdigit() else float('inf')):
            if not target_frame_str.isdigit():
                continue

            target_frame = int(target_frame_str)
            video_dir = os.path.join(video_base_path, target_frame_str)

            frame_paths = get_frame_window(video_dir, target_frame)
            if not frame_paths:
                continue

            bboxes_per_frame = get_bounding_boxes_per_frame(pkl_data, video_id, target_frame)
            if not bboxes_per_frame:
                continue

            keypoint_dict[video_id][target_frame] = {}

            for path in frame_paths:
                frame_id = int(os.path.splitext(os.path.basename(path))[0])
                bounding_boxes = bboxes_per_frame.get(frame_id, np.empty((0, 4), dtype=np.int32))

                pose_results = inference_topdown(pose_estimator, path, bounding_boxes)
                keypoints = extract_keypoints(pose_results, num_people=12)

                # Print the shape of the keypoint array
                print(f"Keypoints shape for video {video_id}, middle frame {target_frame}, frame {frame_id}: {keypoints.shape}")


                keypoint_dict[video_id][target_frame][frame_id] = keypoints

            print(f"Processed video {video_id}, middle frame {target_frame}")

    # Save the final output
    with open(output_pkl_path, "wb") as f:
        pickle.dump(keypoint_dict, f)

    print(f"Keypoint data saved to {output_pkl_path}")

if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
import torch
import cv2
from typing import List, Optional
from mmcv import imread
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import tempfile
from IPython.display import Image, display

local_runtime = True  # Set to False if running in a non-notebook environment

# Setup MMPose models
def setup_models(pose_config, pose_checkpoint, device='cuda:0'):
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    pose_estimator = init_pose_estimator(
        pose_config, pose_checkpoint, device=device, cfg_options=cfg_options
    )

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 1

    return pose_estimator, visualizer

# Get 5 frames before, 1 middle, 4 after
def get_frame_window(dataset_path: str, video_id: int, target_frame: int) -> Optional[List[str]]:
    video_dir = os.path.join(dataset_path, str(video_id), str(target_frame))
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return None

    frame_ids = list(range(target_frame - 5, target_frame + 5))
    frame_paths = []

    for frame_id in frame_ids:
        frame_path = os.path.join(video_dir, f"{frame_id}.jpg")
        if not os.path.exists(frame_path):
            print(f"Missing frame: {frame_path}")
            return None
        frame_paths.append(frame_path)

    return frame_paths

# Load bounding boxes from pkl
def get_bounding_boxes_per_frame(pkl_path: str, video_id: int, target_frame: int) -> Optional[dict]:
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if video_id not in data:
            print(f"Video ID {video_id} not in PKL data.")
            return None

        frame_range = list(range(target_frame - 5, target_frame + 5))
        bboxes_per_frame = {fid: np.empty((0, 4), dtype=np.int32) for fid in frame_range}

        for middle_frame in data[video_id].keys():
            for fid in frame_range:
                if fid in data[video_id][middle_frame]:
                    bboxes_per_frame[fid] = data[video_id][middle_frame][fid]

        return bboxes_per_frame

    except Exception as e:
        print(f"Error loading pkl: {e}")
        return None

# Visualize image with pose annotations
def visualize_img(img_path, pose_estimator, visualizer, bounding_boxes, out_file=None):
    img = imread(img_path, channel_order='rgb')
    if isinstance(bounding_boxes, torch.Tensor):
        bounding_boxes = bounding_boxes.cpu().numpy()

    pose_results = inference_topdown(pose_estimator, img_path, bounding_boxes)
    data_samples = merge_data_samples(pose_results)

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=True,
        draw_bbox=True,
        show=False,
        wait_time=0,
        out_file=out_file,
        kpt_thr=0.3
    )

    vis_result = visualizer.get_image()

    if local_runtime:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = os.path.join(tmpdir, os.path.basename(img_path).replace('.jpg', '_pose.png'))
            cv2.imwrite(file_name, vis_result[:, :, ::-1])
            display(Image(file_name))
    else:
        output_name = os.path.basename(img_path).replace('.jpg', '_pose.png')
        cv2.imwrite(output_name, vis_result[:, :, ::-1])

# Main function
def main():
    dataset_path = "../../Group_Activity/Datasets/volleyball_dataset"
    pkl_path = "volleyball_bboxes.pkl"
    video_id = 39
    target_frame = 29885

    pose_config  = '/home/akila17/e19-group-activity/Group_Activity/Datasets/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    pose_checkpoint = '/home/akila17/e19-group-activity/Group_Activity/Datasets/Skeleton_Data/hrnet_w32_coco_256x192.pth'

    # Step 1: Get image paths
    frame_paths = get_frame_window(dataset_path, video_id, target_frame)
    if not frame_paths:
        return

    # Step 2: Get bounding boxes
    bboxes_per_frame = get_bounding_boxes_per_frame(pkl_path, video_id, target_frame)
    if not bboxes_per_frame:
        return

    # Step 3: Load MMPose models
    pose_estimator, visualizer = setup_models(pose_config, pose_checkpoint)

    # Step 4: Generate and visualize keypoint annotated images
    for path in frame_paths:
        frame_id = int(os.path.splitext(os.path.basename(path))[0])
        bounding_boxes = bboxes_per_frame.get(frame_id, np.empty((0, 4), dtype=np.int32))
        visualize_img(path, pose_estimator, visualizer, bounding_boxes)

if __name__ == "__main__":
    main()

# Volleyball Pose Estimation Visualizer

This project is a Python script that uses [MMPose](https://github.com/open-mmlab/mmpose) to visualize human poses on frames from the volleyball dataset. It performs 2D top-down pose estimation using pretrained models and displays the detected keypoints with bounding boxes and heatmaps.

## 📌 Features

- Loads image frames centered around a target frame from a video
- Reads bounding box annotations from a `.pkl` file
- Uses MMPose's top-down keypoint estimation pipeline
- Visualizes and saves pose-annotated images
- Supports both notebook (`local_runtime = True`) and non-notebook environments

## 🧩 Requirements

- Python 3.7+
- PyTorch
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMEngine](https://github.com/open-mmlab/mmengine)
- OpenCV
- NumPy
- Pickle

## 📂 Dataset Structure

Your dataset should follow this folder structure:

<dataset_path>/
└── <video_id>/
└── <target_frame>/
├── 29880.jpg
├── 29881.jpg
└── ...

css
Copy
Edit

Bounding boxes are expected in a pickle file (`.pkl`) with structure:
```python
{
    <video_id>: {
        <middle_frame>: {
            <frame_id>: np.ndarray of shape (N, 4)
        }
    }
}
```

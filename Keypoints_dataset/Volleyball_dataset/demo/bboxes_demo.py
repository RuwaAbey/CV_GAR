import os
import pickle
import numpy as np
import cv2
from typing import List, Tuple, Optional

def get_frame_window(dataset_path: str, video_id: int, target_frame: int) -> Optional[List[str]]:
    """
    Retrieves paths to 10 frames (5 before, 1 middle, 4 after) for a given video and target frame.
    
    Args:
        dataset_path (str): Root path to the volleyball dataset.
        video_id (int): ID of the video (0 to 54).
        target_frame (int): The middle frame ID (e.g., 29885).
    
    Returns:
        Optional[List[str]]: List of frame file paths in order (or None if frames are invalid).
    """
    # Validate video ID
    if not (0 <= video_id <= 54):
        print(f"Invalid video ID: {video_id}. Must be between 0 and 54.")
        return None
    
    # Construct video directory path (e.g., volleyball/39/29885)
    video_dir = os.path.join(dataset_path, str(video_id), str(target_frame))
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return None
    
    # Calculate frame range: 5 before, middle, 4 after (total 10 frames)
    start_frame = target_frame - 5
    end_frame = target_frame + 4
    frame_ids = list(range(start_frame, end_frame + 1))  # Inclusive range
    
    # Validate frame range (ensure all frames exist)
    frame_paths = []
    for frame_id in frame_ids:
        frame_path = os.path.join(video_dir, f"{frame_id}.jpg")
        if not os.path.exists(frame_path):
            print(f"Frame {frame_id} not found in {video_dir}")
            return None
        frame_paths.append(frame_path)
    
    return frame_paths

def get_bounding_boxes_per_frame(pkl_path: str, video_id: int, target_frame: int) -> Optional[dict]:
    """
    Extract bounding box data as a dictionary of NumPy arrays for each frame in the range:
    5 frames above, the target frame, and 4 frames below.
    
    Args:
        pkl_path (str): Path to the volleyball_bboxes.pkl file.
        video_id (int): Sequence ID (e.g., 39).
        target_frame (int): Target frame number (e.g., 29885).
    
    Returns:
        dict: A dictionary with frame IDs as keys and NumPy arrays as values.
              Each array has shape (n, 4) with columns [x_min, y_min, x_max, y_max].
              Returns None if data is not found or pkl file is invalid.
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if video_id not in data:
            print(f"Error: Sequence ID {video_id} not found in the dataset.")
            return None
        
        frame_range = list(range(target_frame - 5, target_frame + 5))
        bboxes_per_frame = {}
        
        for frame_id in frame_range:
            bboxes_per_frame[frame_id] = np.array([], dtype=np.int32).reshape(0, 4)
        
        for middle_frame in data[video_id].keys():
            if any(frame in data[video_id][middle_frame] for frame in frame_range):
                for frame_id in frame_range:
                    if frame_id in data[video_id][middle_frame]:
                        bboxes_per_frame[frame_id] = data[video_id][middle_frame][frame_id]
        
        if all(len(bboxes) == 0 for bboxes in bboxes_per_frame.values()):
            print(f"Warning: No bounding box data found for frames {frame_range} in sequence {video_id}.")
        
        return bboxes_per_frame
    
    except FileNotFoundError:
        print(f"Error: File '{pkl_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def draw_bounding_boxes_on_frames(frame_paths: List[str], bboxes_per_frame: dict, output_dir: str) -> None:
    """
    Draw bounding boxes on each frame and save the output images.
    
    Args:
        frame_paths (List[str]): List of paths to the frame images.
        bboxes_per_frame (dict): Dictionary mapping frame IDs to bounding box arrays.
        output_dir (str): Directory to save the output images.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_path in frame_paths:
        # Extract frame ID from the file name (e.g., 29885 from 29885.jpg)
        frame_id = int(os.path.splitext(os.path.basename(frame_path))[0])
        
        # Load the image
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Error: Could not load image {frame_path}")
            continue
        
        # Get bounding boxes for this frame
        bboxes = bboxes_per_frame.get(frame_id, np.array([], dtype=np.int32).reshape(0, 4))
        
        # Draw each bounding box on the image
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            # Draw rectangle (color: green, thickness: 2)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Add frame ID label at the top-left corner of the bounding box
            label = f"Frame {frame_id}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the output image
        output_path = os.path.join(output_dir, f"frame_{frame_id}_with_bboxes.jpg")
        cv2.imwrite(output_path, image)
        print(f"Saved image with bounding boxes: {output_path}")

def main():
    dataset_path = "../../Group_Activity/Datasets/volleyball_dataset"  # Replace with actual dataset path
    pkl_path = "volleyball_bboxes.pkl"  # Path to the generated .pkl file
    video_id = 39  # Example video ID
    target_frame = 29885  # Example target frame
    output_dir = "output_frames"  # Directory to save output images
    
    # Step 1: Get frame paths
    frame_paths = get_frame_window(dataset_path, video_id, target_frame)
    if not frame_paths:
        print("Failed to retrieve frame paths.")
        return
    
    print(f"Retrieved {len(frame_paths)} frames for video {video_id}, target frame {target_frame}:")
    for i, path in enumerate(frame_paths):
        print(f"Frame {i - 5}: {path}")
    
    # Step 2: Get bounding boxes
    bboxes_per_frame = get_bounding_boxes_per_frame(pkl_path, video_id, target_frame)
    if bboxes_per_frame is None:
        print("Failed to retrieve bounding box data.")
        return
    
    print("\nBounding box data:")
    for frame_id in range(target_frame - 5, target_frame + 5):
        offset = frame_id - target_frame
        bboxes = bboxes_per_frame[frame_id]
        print(f"Frame {offset} ({frame_id}): {bboxes.shape[0]} boxes\n{bboxes}\n")
    
    # Step 3: Draw bounding boxes on frames and save
    draw_bounding_boxes_on_frames(frame_paths, bboxes_per_frame, output_dir)

if __name__ == "__main__":
    main()

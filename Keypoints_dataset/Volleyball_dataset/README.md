# Volleyball Dataset Repository

This repository contains code and resources for working with the **Volleyball Dataset** and the **Volleyball Tracking Annotation Dataset**, designed for tasks such as action recognition and player tracking in volleyball videos.

## Volleyball Dataset Summary

### Dataset Structure
- **Total Videos**: 55 videos, each with a unique ID (0 to 54).
- **Train Videos** (24): IDs 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
- **Validation Videos** (15): IDs 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
- **Test Videos** (16): IDs 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

### Frame Annotations
- Each video has directories (e.g., `volleyball/39/29885`) containing 41 frames per annotated frame:
  - 20 frames before, 1 target frame, 20 frames after (e.g., for frame 29885, window = 29865 to 29905).
  - **Note**: Scenes change rapidly in volleyball, so frames outside this window are typically not representative. A smaller window of 5 frames before and 4 frames after the target frame is often used for analysis.

### Annotations File
- Each video directory contains an `annotations.txt` file with lines formatted as:

{Frame ID} {Frame Activity Class} {Player Annotation} {Player Annotation} ...

- **Player Annotation Format**: `{Action Class} X Y W H` (bounding box coordinates for each player).

### Video Resolutions
- **1920x1080**: Videos 2, 37, 38, 39, 40, 41, 44, 45 (8 videos).
- **1280x720**: All other videos (47 videos).

## Volleyball Tracking Annotation Dataset Summary

### Structure
- Organized into directories named `seq01`, `seq02`, etc., each corresponding to a sequence.
- Each sequence directory contains a subdirectory named after the **middle frame** (e.g., `seq01/3595`).
- Inside this subdirectory, a `.txt` file (e.g., `3595.txt`) contains bounding box annotations for a 21-frame window:
- 10 frames before, the middle frame, and 10 frames after.
- Example: For middle frame 29885, annotations cover frames 29875 to 29895.

### Annotation Format
Each line in the `.txt` file follows this format:

{Player ID} {X_min} {Y_min} {X_max} {Y_max} {Frame ID} {Flag 1} {Flag 2} {Flag 3} {Action Class}

- **Player ID**: Unique identifier for each player (e.g., 0, 1, ..., 10).
- **X_min, Y_min, X_max, Y_max**: Bounding box coordinates (top-left and bottom-right corners).
- **Frame ID**: Frame number (e.g., 3586 to 3605).
- **Unknown Flags (3)**: Three binary flags (0 or 1). Possible meanings include:
  - Visibility, occlusion, or tracking confidence.
  - Team affiliation or role (e.g., attacker vs. defender).
  - Consult dataset documentation or analyze patterns to confirm their purpose.
- **Action Class**: Player’s action (e.g., `digging`, `standing`).

## Usage Notes
- **Preprocessing**:
  - Parse `annotations.txt` for frame-level activity classes and player bounding boxes.
  - Parse tracking `.txt` files for player tracking across 21-frame windows.
  - Normalize bounding box coordinates for videos with different resolutions (1920x1080 or 1280x720).
- **Tasks**:
  - **Action Recognition**: Use frame-level and player-level action classes for team or individual action classification.
  - **Player Tracking**: Use bounding box data for multi-object tracking tasks.
  - **Temporal Analysis**: Leverage the 41-frame (main dataset) or 21-frame (tracking dataset) windows for sequence modeling.
- **Tools**:
  - Use OpenCV for video frame extraction, Pandas for parsing annotations, and PyTorch/TensorFlow for model training.
- **Resolution Handling**:
  - Check video IDs to confirm resolution and adjust coordinates accordingly.
  - Normalize coordinates (e.g., divide by width/height) for consistent processing.

## Repository Structure
- `data/`: Placeholder for dataset files (not included due to size; download from the original source).
- `scripts/`: Python scripts for parsing annotations, extracting frames, and analysis.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and visualization.
- `models/`: Placeholder for trained models (to be added based on your experiments).

## Getting Started
1. **Download the Dataset**: Obtain the volleyball dataset and tracking annotations from their respective sources.
2. **Parse Annotations**: Use scripts in the `scripts/` directory to process `annotations.txt` and tracking `.txt` files.
3. **Run Analysis**: Explore the data using notebooks in the `notebooks/` directory.
4. **Contribute**: Add your own scripts, models, or visualizations to the repository.

## License
This repository is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The volleyball dataset and tracking annotations are sourced from [original dataset source, if available].
- This repository is intended for research purposes, focusing on action recognition and player tracking in volleyball videos.

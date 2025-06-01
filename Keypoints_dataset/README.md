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

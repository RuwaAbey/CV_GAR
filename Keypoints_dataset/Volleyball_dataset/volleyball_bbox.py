import os
import numpy as np
import pickle

def read_volleyball_annotations(dataset_path):
    """
    Read Volleyball Tracking Annotation Dataset with structure XX/middle_frame/middle_frame.txt
    and create a nested dictionary of bounding boxes.
    
    Args:
        dataset_path (str): Path to the root directory containing 01, 02, etc.
    
    Returns:
        dict: Nested dictionary {seq_id: {middle_frame: {frame_id: bbox_array}}}
    """
    data = {}
    
    # Check if dataset_path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return data
    
    # Iterate through sequence directories (01, 02, etc.) and sort by seq_id
    seq_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not seq_dirs:
        print(f"Warning: No sequence directories found in '{dataset_path}'.")
        return data
    
    # Sort sequence directories by converting to integers
    seq_dirs.sort(key=int)
    
    for seq_dir in seq_dirs:
        seq_path = os.path.join(dataset_path, seq_dir)
        try:
            seq_id = int(seq_dir)  # Extract sequence number (e.g., 1 from 01)
        except ValueError:
            print(f"Warning: Skipping invalid sequence directory '{seq_dir}'.")
            continue
        data[seq_id] = {}
        print(f"Processing sequence: {seq_dir}")
        
        # Iterate through middle frame subdirectories and sort by middle_frame
        middle_frame_dirs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]
        if not middle_frame_dirs:
            print(f"Warning: No middle frame directories found in '{seq_path}'.")
            continue
        
        # Sort middle frame directories by converting to integers
        middle_frame_dirs.sort(key=int)
        
        for middle_frame_dir in middle_frame_dirs:
            middle_frame_path = os.path.join(seq_path, middle_frame_dir)
            try:
                middle_frame = int(middle_frame_dir)  # Middle frame ID (e.g., 3595)
            except ValueError:
                print(f"Warning: Skipping invalid middle frame directory '{middle_frame_dir}' in {seq_dir}.")
                continue
            data[seq_id][middle_frame] = {}
            
            # Read the annotation .txt file (e.g., 3595.txt)
            txt_file = os.path.join(middle_frame_path, f"{middle_frame}.txt")
            if not os.path.exists(txt_file):
                print(f"Warning: Annotation file '{txt_file}' not found.")
                continue
            
            # Initialize dictionary to store bounding boxes by frame ID
            frame_bboxes = {}
            line_count = 0
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 10:
                        print(f"Warning: Skipping malformed line in '{txt_file}' (expected 10 fields, got {len(parts)}): {line.strip()}")
                        continue
                    try:
                        # Extract bounding box and frame ID
                        x_min, y_min, x_max, y_max = map(int, parts[1:5])
                        frame_id = int(parts[5])
                        bbox = [x_min, y_min, x_max, y_max]
                        
                        # Initialize frame_id entry if not present
                        if frame_id not in frame_bboxes:
                            frame_bboxes[frame_id] = []
                        frame_bboxes[frame_id].append(bbox)
                        line_count += 1
                    except ValueError:
                        print(f"Warning: Skipping invalid line in '{txt_file}': {line.strip()}")
                        continue
            
            if not frame_bboxes:
                print(f"Warning: No valid bounding boxes found in '{txt_file}'.")
                continue
            
            print(f"Processed {line_count} valid lines in '{txt_file}' for {len(frame_bboxes)} frames.")
            
            # Convert lists to NumPy arrays and store in the nested dictionary
            for frame_id, bboxes in frame_bboxes.items():
                data[seq_id][middle_frame][frame_id] = np.array(bboxes, dtype=np.int32)
    
    if not data:
        print("Error: No data was processed. The output .pkl file will be empty.")
    else:
        print(f"Processed {len(data)} sequences with bounding box data.")
    
    return data

def save_to_pkl(data, output_path):
    """
    Save the data dictionary to a .pkl file.
    
    Args:
        data (dict): Nested dictionary of bounding boxes
        output_path (str): Path to save the .pkl file
    """
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved bounding box data to '{output_path}'")

def main(dataset_path, output_pkl_path):
    """
    Main function to process dataset and save to .pkl file.
    
    Args:
        dataset_path (str): Path to the dataset root directory
        output_pkl_path (str): Path for the output .pkl file
    """
    # Read annotations and create the nested dictionary
    bbox_data = read_volleyball_annotations(dataset_path)
    
    # Save to .pkl file
    save_to_pkl(bbox_data, output_pkl_path)

if __name__ == "__main__":
    # Example usage
    dataset_path = "../../Group_Activity/Datasets/volleyball_tracking_annotation/_"  # Replace with actual dataset path
    output_pkl_path = "volleyball_bboxes.pkl"   # Output file name
    main(dataset_path, output_pkl_path)

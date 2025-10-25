# visualize_test_results.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

from configs.config import (
    ORIGINAL_POSES_DIR,
    PREDICTED_POSES_DIR,
    OUTPUT_GIFS_DIR,
    MAX_SAMPLES_VISUALIZATION,
    GIF_FPS,
    SHOW_AXES
)

# Define connections for body and hands
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (12, 14), (14, 16), (16, 22), (11, 13), (13, 15), (15, 21),
    (23, 24), (24, 26), (26, 28), (28, 32), (23, 25), (25, 27), (27, 29), (29, 31)
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),   # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Paths (modify these paths according to your environment)
original_dir = "/data/zmo/Swisspose/swissubase_2569_1_0/data/test_data_2d"
predicted_dir = "/data/zmo/Swisspose/Swisspose_project/test_result/"
output_dir = "/data/zmo/Swisspose/Swisspose_project//test_result/output_gifs"
max_samples = 20  # Set to None to process all files
fps = 25
show_axes = True  # True to display axes

def load_pose_sequence(npy_path):
    """
    Loads the pose sequence from a .npy file.

    Args:
        npy_path (str): Path to the .npy file.

    Returns:
        np.ndarray or None: Pose sequence array with shape (num_frames, 75, 2) or None if failed.
    """
    try:
        pose_sequence = np.load(npy_path, allow_pickle=True)  # Enable loading object arrays
        print(f"Loaded {npy_path} with dtype {pose_sequence.dtype} and shape {pose_sequence.shape}")

        if not isinstance(pose_sequence, np.ndarray):
            print(f"Warning: {npy_path} is not an ndarray. Actual type: {type(pose_sequence)}. Skipping this file.")
            return None

        if pose_sequence.ndim != 3 or pose_sequence.shape[1] != 75 or pose_sequence.shape[2] != 2:
            print(f"Warning: {npy_path} has shape {pose_sequence.shape}, expected (num_frames, 75, 2). Skipping this file.")
            return None

        # Ensure data is float type
        if not np.issubdtype(pose_sequence.dtype, np.floating):
            print(f"Converting {npy_path} data type from {pose_sequence.dtype} to float.")
            pose_sequence = pose_sequence.astype(float)

        return pose_sequence
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None

def plot_pose(pose, title="Pose Frame", show_axes=False):
    """
    Plots a single pose frame with keypoints and connections.

    Args:
        pose (np.ndarray): Pose data with shape (75, 2).
        title (str, optional): Title of the plot. Defaults to "Pose Frame".
        show_axes (bool, optional): Whether to display axes. Defaults to False.

    Returns:
        np.ndarray: RGB image array of the plot.
    """
    plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.gca()
    ax.set_title(title)
    if not show_axes:
        plt.axis('off')
    else:
        plt.xlabel("X")
        plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    
    # Invert y-axis to align coordinate system
    plt.gca().invert_yaxis()
    
    # Plot body keypoints
    plt.scatter(pose[:, 0], pose[:, 1], c='blue', s=20, zorder=2)
    
    # Plot body connections
    for connection in BODY_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(pose) and end_idx < len(pose):
            start_point = pose[start_idx]
            end_point = pose[end_idx]
            try:
                if not (np.isnan(start_point).any() or np.isnan(end_point).any()):
                    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'green', linewidth=2, zorder=1)
            except TypeError as te:
                print(f"TypeError with connection {connection}: {te}")
                continue
    
    # Plot left hand connections (keypoints 33 to 53)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        adjusted_start = 33 + start_idx
        adjusted_end = 33 + end_idx
        if adjusted_start < len(pose) and adjusted_end < len(pose):
            start_point = pose[adjusted_start]
            end_point = pose[adjusted_end]
            try:
                if not (np.isnan(start_point).any() or np.isnan(end_point).any()):
                    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'red', linewidth=2, zorder=1)
            except TypeError as te:
                print(f"TypeError with left hand connection {connection}: {te}")
                continue
    
    # Plot right hand connections (keypoints 54 to 74)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        adjusted_start = 54 + start_idx
        adjusted_end = 54 + end_idx
        if adjusted_start < len(pose) and adjusted_end < len(pose):
            start_point = pose[adjusted_start]
            end_point = pose[adjusted_end]
            try:
                if not (np.isnan(start_point).any() or np.isnan(end_point).any()):
                    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'orange', linewidth=2, zorder=1)
            except TypeError as te:
                print(f"TypeError with right hand connection {connection}: {te}")
                continue
    
    # Save plot to memory
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image

def create_combined_gif(original_pose_sequence, predicted_pose_sequence, gif_path, fps=15, show_axes=False):
    """
    Creates a side-by-side GIF comparing original and predicted pose sequences.

    Args:
        original_pose_sequence (np.ndarray): Original pose sequence with shape (num_frames, 75, 2).
        predicted_pose_sequence (np.ndarray): Predicted pose sequence with shape (num_frames, 75, 2).
        gif_path (str): Path to save the output GIF.
        fps (int, optional): Frames per second for the GIF. Defaults to 15.
        show_axes (bool, optional): Whether to display axes in the plots. Defaults to False.
    """
    frames = []
    num_frames_original = original_pose_sequence.shape[0]
    num_frames_predicted = predicted_pose_sequence.shape[0]
    num_frames = max(num_frames_original, num_frames_predicted)
    
    for frame_idx in range(num_frames):
        # Get original pose frame
        if frame_idx < num_frames_original:
            pose_original = original_pose_sequence[frame_idx]
        else:
            # Pad with zeros if original pose has fewer frames
            pose_original = np.zeros((75, 2))
        
        # Get predicted pose frame
        if frame_idx < num_frames_predicted:
            pose_predicted = predicted_pose_sequence[frame_idx]
        else:
            # Pad with zeros if predicted pose has fewer frames
            pose_predicted = np.zeros((75, 2))
        
        # Plot original pose
        title_original = f"Original Frame {frame_idx + 1}/{num_frames_original}"
        image_original = plot_pose(pose_original, title=title_original, show_axes=show_axes)
        
        # Plot predicted pose
        title_predicted = f"Predicted Frame {frame_idx + 1}/{num_frames_predicted}"
        image_predicted = plot_pose(pose_predicted, title=title_predicted, show_axes=show_axes)
        
        # Concatenate predicted and original images horizontally
        combined_image = np.hstack((image_original, image_predicted))  # Swapped order
        frames.append(combined_image)
    
    try:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF: {gif_path}")
    except Exception as e:
        print(f"Error saving GIF {gif_path}: {e}")


def main():
    # Define paths
    original_dir_local = original_dir
    predicted_dir_local = predicted_dir
    output_dir_local = output_dir
    max_samples_local = max_samples
    fps_local = fps
    show_axes_local = show_axes

    # Check if original and predicted directories exist
    if not os.path.exists(original_dir_local):
        print(f"Error: Original poses directory {original_dir_local} does not exist.")
        return
    
    if not os.path.exists(predicted_dir_local):
        print(f"Error: Predicted poses directory {predicted_dir_local} does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_local, exist_ok=True)
    print(f"Output GIFs will be saved to {output_dir_local}")
    
    # Get all original .npy files
    all_original_npy_files = [f for f in os.listdir(original_dir_local) if f.endswith('.npy')]
    total_files = len(all_original_npy_files)
    
    if total_files == 0:
        print(f"No .npy files found in {original_dir_local}.")
        return
    
    # If max_samples is set, limit the number of files to process
    if max_samples_local is not None:
        all_original_npy_files = all_original_npy_files[:max_samples_local]
    
    print(f"Processing {len(all_original_npy_files)} out of {total_files} .npy files...")
    
    for npy_file in tqdm(all_original_npy_files, desc="Generating Combined GIFs"):
        original_npy_path = os.path.join(original_dir_local, npy_file)
        # Assume predicted pose files are named as {original_filename}_prediction.npy
        base_filename = os.path.splitext(npy_file)[0]
        predicted_npy_file = f"{base_filename}_prediction.npy"
        predicted_npy_path = os.path.join(predicted_dir_local, predicted_npy_file)
        
        # Check if predicted pose file exists
        if not os.path.exists(predicted_npy_path):
            print(f"Warning: Predicted pose file {predicted_npy_file} does not exist. Skipping this file.")
            continue
        
        # Load original pose sequence
        original_pose_sequence = load_pose_sequence(original_npy_path)
        if original_pose_sequence is None:
            continue  # Skip if loading failed
        
        # Load predicted pose sequence
        predicted_pose_sequence = load_pose_sequence(predicted_npy_path)
        if predicted_pose_sequence is None:
            continue  # Skip if loading failed
        
        # Define output GIF filename
        combined_gif_filename = f"{base_filename}_comparison.gif"
        combined_gif_path = os.path.join(output_dir_local, combined_gif_filename)
        
        # Create and save the combined GIF
        create_combined_gif(original_pose_sequence, predicted_pose_sequence, combined_gif_path, fps=fps_local, show_axes=show_axes_local)
    
    print(f"All combined GIFs have been saved to {output_dir_local}.")

if __name__ == "__main__":
    main()

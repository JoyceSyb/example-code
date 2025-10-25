# demo_inference.py

import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

from models.model import IterativeTextGuidedPoseGenerationModel
from configs.config import (
    START_LEARNING_RATE_TEXT_ENCODER,
    START_LEARNING_RATE_GENERATOR,
    TOTAL_STEPS,
    BETA_START,
    BETA_END,
    ETA,
    MAX_SEQ_LEN,
    MAX_LENGTH_PREDICTION,
    BATCH_SIZE,
    NUM_WORKERS
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

class TextPoseDataset(Dataset):
    """
    Custom Dataset for inference. Each item is a text input.
    """
    def __init__(self, input_file, tokenizer, max_seq_len=300):
        """
        Initializes the dataset.
        
        Args:
            input_file (str): Path to the input text file.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text processing.
            max_seq_len (int, optional): Maximum sequence length. Defaults to 300.
        """
        self.texts = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    self.texts.append(text)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text
        text_inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len
        )
        text_input_ids = text_inputs['input_ids'].squeeze(0)  # (seq_len,)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)  # (seq_len,)
        return {
            'text_input_ids': text_input_ids,
            'text_attention_masks': text_attention_mask,
            'text': text
        }

def plot_pose(pose, title="Pose Frame", show_axes=False):
    """
    Plots a single pose frame with keypoints and connections.
    
    Args:
        pose (np.ndarray): Pose data with shape (75, 2) or (150,).
        title (str, optional): Title of the plot. Defaults to "Pose Frame".
        show_axes (bool, optional): Whether to display axes. Defaults to False.
    
    Returns:
        np.ndarray: RGB image array of the plot.
    """
    # Convert to NumPy array if not already
    pose = np.asarray(pose)
    
    # Reshape if necessary
    if pose.ndim == 1 and pose.size == 150:
        pose = pose.reshape(75, 2)
    elif pose.ndim != 2 or pose.shape[1] != 2:
        raise ValueError(f"Pose has an unexpected shape: {pose.shape}")
    
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


def create_pose_gif(pose_sequence, gif_path, fps=10, show_axes=False):
    """
    Creates a GIF from a pose sequence.
    
    Args:
        pose_sequence (np.ndarray): Pose sequence with shape (num_frames, 75, 2) or (num_frames, 150).
        gif_path (str): Path to save the GIF.
        fps (int, optional): Frames per second for the GIF. Defaults to 10.
        show_axes (bool, optional): Whether to display axes in the plots. Defaults to False.
    """
    frames = []
    num_frames = pose_sequence.shape[0]
    
    for frame_idx in range(num_frames):
        pose = pose_sequence[frame_idx]
        
        # Ensure pose is 2D
        if pose.ndim == 1 and pose.size == 150:
            pose = pose.reshape(75, 2)
        elif pose.ndim != 2 or pose.shape[1] != 2:
            raise ValueError(f"Pose at frame {frame_idx} has an unexpected shape: {pose.shape}")
        
        title = f"Frame {frame_idx + 1}/{num_frames}"
        image = plot_pose(pose, title=title, show_axes=show_axes)
        frames.append(image)
    
    try:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF: {gif_path}")
    except Exception as e:
        print(f"Error saving GIF {gif_path}: {e}")


def main():
    # Define paths
    input_txt_path = "demo/input/input.txt"
    output_npy_dir = "demo/output"
    output_gif_dir = os.path.join(output_npy_dir, "gif")
    checkpoint_path = "/data/zmo/Swisspose/Swisspose_project/best_model/best-checkpoint.ckpt"  # Updated checkpoint path
    
    # Ensure output directories exist
    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(output_gif_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    
    # Initialize dataset and dataloader
    inference_dataset = TextPoseDataset(
        input_file=input_txt_path,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN  
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=BATCH_SIZE,  
        shuffle=False,
        num_workers=NUM_WORKERS,  
        pin_memory=True,
        drop_last=False
    )
    
    # Load the trained model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        test_results_dir=output_npy_dir,
        text_model_name="xlm-roberta-base",  
        pose_dims=(75, 2),
        hidden_dim=512,           
        num_layers=4,            
        num_heads=8,             
        num_steps=20,             
        lr_text_encoder=START_LEARNING_RATE_TEXT_ENCODER,
        lr_generator=START_LEARNING_RATE_GENERATOR,
        beta_start=BETA_START,
        beta_end=BETA_END,
        eta=ETA,
        total_steps=TOTAL_STEPS,  
        pose_mean=0.0,             
        pose_std=1.0              
    )
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices=1,  
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',  
        precision=16 if torch.cuda.is_available() else 32,
        enable_progress_bar=True,  
    )
    
    # Move model to appropriate device
    model.to(trainer.strategy.root_device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradients for inference
    torch.no_grad().__enter__()
    
    # Iterate over the DataLoader
    for batch in tqdm(inference_loader, desc="Generating Poses"):
        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        texts = batch['text']
        
        # Generate pose sequences
        poses, num_steps_pred = model.generate_pose(
            text_input_ids=text_input_ids.to(model.device),
            text_attention_masks=text_attention_masks.to(model.device),
            num_steps=None  # Use default number of steps (20)
        )  # poses: (batch_size, num_frames, 75, 2)
        
        # Move poses to CPU and convert to NumPy
        poses = poses.cpu().numpy()
        print(f"Generated poses shape: {poses.shape}")
        # Iterate over each sample in the batch
        for i, pose_sequence in enumerate(poses):
            text = texts[i]
            # Create a safe filename from text
            safe_text = "".join([c if c.isalnum() or c in (' ', '_') else "_" for c in text]).strip().replace(" ", "_")
            if not safe_text:
                safe_text = f"sample_{i}"
            npy_filename = f"{safe_text}.npy"
            npy_path = os.path.join(output_npy_dir, npy_filename)
            
            # Save pose sequence as .npy file
            try:
                np.save(npy_path, pose_sequence)
                print(f"Saved pose sequence to {npy_path}")
            except Exception as e:
                print(f"Error saving {npy_path}: {e}")
                continue
            
            # Create and save visualization GIF
            gif_filename = f"{safe_text}.gif"
            gif_path = os.path.join(output_gif_dir, gif_filename)
            create_pose_gif(
                pose_sequence=pose_sequence,
                gif_path=gif_path,
                fps=10,
                show_axes=True
            )
    
    print("Inference and visualization completed.")

if __name__ == "__main__":
    main()

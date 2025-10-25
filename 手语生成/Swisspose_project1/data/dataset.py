# data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer
from utils.interpolation import interpolate_nans
from configs.config import MAX_SEQ_LEN, MAX_LENGTH_PREDICTION

class SignLanguageDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        """
        Initializes the dataset.
        """
        self.annotations = []
        self.transform = transform
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")  # Using XLM-RoBERTa Tokenizer

        # Iterate through all files in the data folder
        for file in os.listdir(data_folder):
            if file.endswith('.npy'):
                base_filename = os.path.splitext(file)[0]
                pose_path = os.path.join(data_folder, file)
                text_path = os.path.join(data_folder, f"{base_filename}.txt")
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    pose_sequence = np.load(pose_path, allow_pickle=True).astype(np.float32)  # (num_frames, 75, 2)
                    pose_sequence = interpolate_nans(pose_sequence)
                    
                    # Truncate pose_sequence to MAX_LENGTH_PREDICTION if necessary
                    if pose_sequence.shape[0] > MAX_LENGTH_PREDICTION:
                        pose_sequence = pose_sequence[:MAX_LENGTH_PREDICTION, :, :]  # (MAX_LENGTH_PREDICTION, 75, 2)
                    
                    if pose_sequence.shape[0] >= 1:
                        self.annotations.append({
                            'pose_path': pose_path,
                            'text': text,
                            'filename': base_filename  
                        })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        text = sample['text']
        pose_path = sample['pose_path']
        filename = sample['filename']

        # Process text using XLMRobertaTokenizer
        text_inputs = self.tokenizer(
            text=text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LEN
        )
        text_input_ids = text_inputs['input_ids'].squeeze(0)          # (seq_len,)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)  # (seq_len,)

        # Load pose data
        pose_sequence = np.load(pose_path, allow_pickle=True).astype(np.float32)  # (num_frames, 75, 2)

        # Interpolate NaN values
        pose_sequence = interpolate_nans(pose_sequence)  # (num_frames, 75, 2)

        # Truncate pose_sequence to MAX_LENGTH_PREDICTION if not already done
        if pose_sequence.shape[0] > MAX_LENGTH_PREDICTION:
            pose_sequence = pose_sequence[:MAX_LENGTH_PREDICTION, :, :]  # (MAX_LENGTH_PREDICTION, 75, 2)

        # Convert to torch.Tensor
        pose_sequence = torch.from_numpy(pose_sequence)  # (num_frames, 75, 2)

        # Apply transformation (e.g., normalization)
        if self.transform:
            pose_sequence = self.transform(pose_sequence)

        return text_input_ids, text_attention_mask, pose_sequence, filename

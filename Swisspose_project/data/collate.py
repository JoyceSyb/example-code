# data/collate.py

import torch
from configs.config import MAX_LENGTH_PREDICTION

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length pose sequences and text inputs.
    Ensures that pose sequences do not exceed MAX_LENGTH_PREDICTION.
    Dynamically computes and includes sequence lengths in the batch.
    """
    # Unzip the batch
    text_input_ids, text_attention_masks, pose_sequences, filenames = zip(*batch)

    # Stack text input IDs and attention masks
    text_input_ids = torch.stack(text_input_ids, dim=0)  # (batch_size, seq_len)
    text_attention_masks = torch.stack(text_attention_masks, dim=0)  # (batch_size, seq_len)

    # Initialize lists to store padded poses, masks, and lengths
    padded_pose_sequences = []
    masks = []
    lengths = []

    for pose_seq in pose_sequences:
        num_frames = pose_seq.size(0)
        lengths.append(min(num_frames, MAX_LENGTH_PREDICTION))  # Compute length capped at MAX_LENGTH_PREDICTION

        if num_frames > MAX_LENGTH_PREDICTION:
            # Truncate to MAX_LENGTH_PREDICTION
            padded_pose = pose_seq[:MAX_LENGTH_PREDICTION, :, :]  # (MAX_LENGTH_PREDICTION, 75, 2)
            mask = torch.ones(MAX_LENGTH_PREDICTION, dtype=torch.bool)
        elif num_frames < MAX_LENGTH_PREDICTION:
            padding_frames = MAX_LENGTH_PREDICTION - num_frames
            padding = torch.zeros((padding_frames, pose_seq.size(1), pose_seq.size(2)), dtype=pose_seq.dtype)
            padded_pose = torch.cat([pose_seq, padding], dim=0)  # (MAX_LENGTH_PREDICTION, 75, 2)
            mask = torch.cat([torch.ones(num_frames, dtype=torch.bool), torch.zeros(padding_frames, dtype=torch.bool)], dim=0)
        else:
            padded_pose = pose_seq  # (MAX_LENGTH_PREDICTION, 75, 2)
            mask = torch.ones(MAX_LENGTH_PREDICTION, dtype=torch.bool)

        padded_pose_sequences.append(padded_pose)
        masks.append(mask)

    # Stack all pose sequences and masks
    padded_pose_sequences = torch.stack(padded_pose_sequences, dim=0)  # (batch_size, MAX_LENGTH_PREDICTION, 75, 2)
    masks = torch.stack(masks, dim=0)  # (batch_size, MAX_LENGTH_PREDICTION)
    lengths = torch.tensor(lengths, dtype=torch.long)  # (batch_size,)

    # Optional: Print the shapes for debugging (ensure correct referencing)
    print(f"Batch text_input_ids shape: {text_input_ids.shape}")             # (batch_size, seq_len)
    print(f"Batch text_attention_masks shape: {text_attention_masks.shape}") # (batch_size, seq_len)
    print(f"Batch pose_sequences shape: {padded_pose_sequences.shape}")      # (batch_size, MAX_LENGTH_PREDICTION, 75, 2)
    print(f"Batch masks shape: {masks.shape}")                              # (batch_size, MAX_LENGTH_PREDICTION)
    print(f"Batch lengths shape: {lengths.shape}")                          # (batch_size,)

    return {
        'text_input_ids': text_input_ids,               # (batch_size, seq_len)
        'text_attention_masks': text_attention_masks,   # (batch_size, seq_len)
        'pose_sequences': padded_pose_sequences,        # (batch_size, MAX_LENGTH_PREDICTION, 75, 2)
        'masks': masks,                                 # (batch_size, MAX_LENGTH_PREDICTION)
        'filenames': filenames,                         # List of filenames
        'lengths': lengths                              # (batch_size,)
    }

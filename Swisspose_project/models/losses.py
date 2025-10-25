# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(text_embeddings, sign_embeddings, temperature=0.07):
    """
    InfoNCE loss function to align text embeddings with sign embeddings.
    Cross-modal alignment with normalization.
    
    Args:
        text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, embed_dim).
        sign_embeddings (torch.Tensor): Sign embeddings of shape (batch_size, embed_dim).
        temperature (float): Temperature parameter for scaling.
    
    Returns:
        torch.Tensor: Scalar InfoNCE loss.
    """
    batch_size = text_embeddings.size(0)
    # Normalize embeddings
    text_norm = F.normalize(text_embeddings, dim=1)
    sign_norm = F.normalize(sign_embeddings, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(text_norm, sign_norm.T)  # (batch_size, batch_size)
    similarity_matrix = similarity_matrix / temperature
    
    # Create labels
    labels = torch.arange(batch_size).to(text_embeddings.device)
    
    # Cross-entropy loss for both directions
    loss_fn = nn.CrossEntropyLoss()
    loss_text_to_sign = loss_fn(similarity_matrix, labels)
    loss_sign_to_text = loss_fn(similarity_matrix.T, labels)
    
    # Average the losses
    loss = (loss_text_to_sign + loss_sign_to_text) / 2
    return loss

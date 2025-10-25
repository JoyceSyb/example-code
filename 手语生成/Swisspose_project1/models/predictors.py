# models/predictors.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x)  # (batch_size, seq_len, embed_dim)
        x = self.layer_norm1(x + self.dropout1(attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout2(ff_output))
        return x

class SignPredictor(nn.Module):
    def __init__(self, embed_dim=3072, num_attention_blocks=8, num_heads=8, hidden_dim=3072, dropout=0.1, pose_dim=150):
        """
        Sign Predictor with eight Attention Blocks and an MLP to predict poses.
        
        Args:
            embed_dim (int): Dimension of the concatenated embeddings (4 * 768 = 3072).
            num_attention_blocks (int): Number of Attention Blocks.
            num_heads (int): Number of attention heads in each Attention Block.
            hidden_dim (int): Dimension of the hidden layers in MLP.
            dropout (float): Dropout rate for regularization.
            pose_dim (int): Dimension of the pose output (75 keypoints * 2 coordinates).
        """
        super(SignPredictor, self).__init__()
        self.attention_blocks = nn.Sequential(
            *[AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_attention_blocks)]
        )
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, pose_dim)
    
    def forward(self, fused_embeddings):
        """
        Forward pass for predicting poses.
        
        Args:
            fused_embeddings (torch.Tensor): Concatenated embeddings of shape (batch_size * max_num_frames, 3072).
        
        Returns:
            torch.Tensor: Predicted poses of shape (batch_size * max_num_frames, pose_dim).
        """
        # Pass through Attention Blocks
        x = self.attention_blocks(fused_embeddings)  # (batch_size * max_num_frames, 3072)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch_size * max_num_frames, hidden_dim)
        x = self.relu(x)
        pose_pred = self.fc2(x)  # (batch_size * max_num_frames, pose_dim)
        
        return pose_pred

class LengthPredictor(nn.Module):
    def __init__(self, input_dim=768, embed_dim=512, num_heads=8, num_transformer_layers=2, hidden_dims=[1024, 512], output_dim=1, dropout=0.2):
        """
        Enhanced Length Predictor using Transformer and MLP for predicting sequence length.
    
        Args:
            input_dim (int): Input dimension (text embedding dimension).
            embed_dim (int): Embedding dimension for Transformer.
            num_heads (int): Number of attention heads in Transformer.
            num_transformer_layers (int): Number of Transformer encoder layers.
            hidden_dims (list of int): Hidden layer dimensions for MLP.
            output_dim (int): Output dimension (predicted length).
            dropout (float): Dropout probability for regularization.
        """
        super(LengthPredictor, self).__init__()
        self.input_linear = nn.Linear(input_dim, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation='gelu'),
            num_layers=num_transformer_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass for length prediction.
    
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
    
        Returns:
            torch.Tensor: Predicted length of shape (batch_size, 1).
        """
        # Project input to Transformer embedding
        x = self.input_linear(x)  # (batch_size, embed_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, embed_dim)
        x = x.squeeze(1)  # (batch_size, embed_dim)
        
        # Pass through MLP to predict length
        length_pred = self.mlp(x)  # (batch_size, 1)
        
        # Clamp output to [0, 100]
        length_pred = torch.sigmoid(length_pred) * 100  # Ensure output is within [0, 100]
        
        return length_pred  # (batch_size, 1)

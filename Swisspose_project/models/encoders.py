# models/encoders.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel

class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(AdvancedMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.max_len = max_len

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim) with positional encodings added
        """
        batch_size, seq_len, embed_dim = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)  # (batch_size, seq_len)
        pos_embeddings = self.position_embedding(positions)  # (batch_size, seq_len, embed_dim)
        return x + pos_embeddings

class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(RelativePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.relative_positions = nn.Parameter(torch.randn(max_len, embed_dim))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.relative_positions[:seq_len, :].unsqueeze(0).expand(batch_size, seq_len, embed_dim)
        return x + pos_embeddings

class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)  # Self-attention
        x = self.layer_norm1(x + self.dropout1(attn_output))  # Residual connection and LayerNorm
        # Typically, a feed-forward network would follow here, but it's omitted as per current requirements
        return x

class TextEncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", embed_dim=768, num_heads=8, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.base_model = XLMRobertaModel.from_pretrained(model_name)

        self.attention1 = AttentionModule(embed_dim, num_heads, dropout)
        self.attention2 = AttentionModule(embed_dim, num_heads, dropout)
        self.mlp = AdvancedMLP(input_dim=embed_dim, hidden_dims=[3072, 1536], output_dim=embed_dim, dropout=dropout)
    
    def forward(self, input_ids, attention_mask):
        # Base model output
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        
        # Attention layers
        hidden_states = self.attention1(hidden_states)
        hidden_states = self.attention2(hidden_states)
        
        # Aggregate [CLS] token (assuming XLM-RoBERTa uses the first token as [CLS])
        cls_embedding = hidden_states[:, 0, :]  # (batch_size, 768)
        
        # MLP
        cls_embedding = self.mlp(cls_embedding)  # (batch_size, 768)
        
        return cls_embedding  # (batch_size, 768)

class SignEncoder(nn.Module):
    def __init__(self, pose_dim, embed_dim=768, num_attention_blocks=2, num_heads=8, dropout=0.1, max_len=100, use_relative_pos=True):
        super(SignEncoder, self).__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.input_linear = nn.Linear(pose_dim, embed_dim)
        self.use_relative_pos = use_relative_pos
        if self.use_relative_pos:
            self.positional_encoding = RelativePositionalEncoding(embed_dim, max_len)
        else:
            self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        
        self.attention_blocks = nn.ModuleList([
            AttentionModule(embed_dim, num_heads, dropout) for _ in range(num_attention_blocks)
        ])
        self.mlp = AdvancedMLP(input_dim=embed_dim, hidden_dims=[3072, 1536], output_dim=embed_dim, dropout=dropout)
    
    def forward(self, pose_sequences, masks):
        """
        Args:
            pose_sequences: (batch_size, max_num_frames, pose_dim)
            masks: (batch_size, max_num_frames)
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        x = self.input_linear(pose_sequences)  # (batch_size, max_num_frames, embed_dim)
        x = self.positional_encoding(x)  # (batch_size, max_num_frames, embed_dim)
        
        for attn in self.attention_blocks:
            x = attn(x)  # (batch_size, max_num_frames, embed_dim)
        
        # Masked mean pooling
        masks = masks.unsqueeze(-1)  # (batch_size, max_num_frames, 1)
        x = x * masks  # Apply mask
        sum_x = x.sum(dim=1)  # (batch_size, embed_dim)
        lengths = masks.sum(dim=1).clamp(min=1)  # (batch_size, 1)
        pooled = sum_x / lengths  # (batch_size, embed_dim)
        
        # MLP
        embeddings = self.mlp(pooled)  # (batch_size, embed_dim)
        
        return embeddings  # (batch_size, embed_dim)

class NoiseEncoder(nn.Module):
    def __init__(self, pose_dim, embed_dim=768, num_attention_blocks=2, num_heads=8, dropout=0.1, max_len=100, use_relative_pos=True):
        super(NoiseEncoder, self).__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.input_linear = nn.Linear(pose_dim, embed_dim)
        self.use_relative_pos = use_relative_pos
        if self.use_relative_pos:
            self.positional_encoding = RelativePositionalEncoding(embed_dim, max_len)
        else:
            self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        self.attention_blocks = nn.ModuleList([
            AttentionModule(embed_dim, num_heads, dropout) for _ in range(num_attention_blocks)
        ])
        self.mlp = AdvancedMLP(input_dim=embed_dim, hidden_dims=[3072, 1536], output_dim=embed_dim, dropout=dropout)
    
    def forward(self, noise):
        """
        Args:
            noise: (batch_size, pose_dim)
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        # Assuming noise is (batch_size, pose_dim), we add a time dimension of 1
        x = noise.unsqueeze(1)  # (batch_size, 1, pose_dim)
        x = self.input_linear(x)  # (batch_size, 1, embed_dim)
        x = self.positional_encoding(x)  # (batch_size, 1, embed_dim)
        
        for attn in self.attention_blocks:
            x = attn(x)  # (batch_size, 1, embed_dim)
        
        # Squeeze the time dimension
        x = x.squeeze(1)  # (batch_size, embed_dim)
        
        # MLP
        embeddings = self.mlp(x)  # (batch_size, embed_dim)
        
        return embeddings  # (batch_size, embed_dim)

class StepEncoder(nn.Module):
    def __init__(self, num_steps, embed_dim=768, num_attention_blocks=2, num_heads=8, dropout=0.1):
        super(StepEncoder, self).__init__()
        self.embedding = nn.Embedding(num_steps, embed_dim)
        self.attention_blocks = nn.ModuleList([
            AttentionModule(embed_dim, num_heads, dropout) for _ in range(num_attention_blocks)
        ])
        self.mlp = AdvancedMLP(input_dim=embed_dim, hidden_dims=[3072, 1536], output_dim=embed_dim, dropout=dropout)
    
    def forward(self, step_numbers):
        """
        Args:
            step_numbers: (batch_size, max_num_steps)
        Returns:
            embeddings: (batch_size, max_num_steps, embed_dim)
        """
        x = self.embedding(step_numbers)  # (batch_size, max_num_steps, embed_dim)
        
        for attn in self.attention_blocks:
            x = attn(x)  # (batch_size, max_num_steps, embed_dim)
        
        # Apply MLP to each step embedding
        batch_size, max_num_steps, embed_dim = x.size()
        x = x.view(batch_size * max_num_steps, embed_dim)
        embeddings = self.mlp(x)  # (batch_size * max_num_steps, embed_dim)
        embeddings = embeddings.view(batch_size, max_num_steps, embed_dim)  # (batch_size, max_num_steps, embed_dim)
        
        return embeddings  # (batch_size, max_num_steps, embed_dim)

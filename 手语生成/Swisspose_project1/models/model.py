import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from models.encoders import SignEncoder, StepEncoder, NoiseEncoder, TextEncoder
from models.predictors import SignPredictor, LengthPredictor  # Import LengthPredictor
from models.diffusion import DDIMSampler
from models.losses import info_nce_loss
from configs.config import (
    EPSILON,
    START_LEARNING_RATE_TEXT_ENCODER,
    START_LEARNING_RATE_GENERATOR,
    MAX_SEQ_LEN,
    FREEZE_LAYERS,
    MAX_LENGTH_PREDICTION,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_EPOCHS,
    TOTAL_STEPS,
    LR_TEXT_ENCODER,
    LR_GENERATOR,
    BETA_START,
    BETA_END,
    ETA,
    CHECKPOINT_DIR_BEST,
    CHECKPOINT_DIR_LAST,
    LOG_DIR,
    LOG_NAME,
    TEST_RESULTS_DIR
)

class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
    def __init__(
        self,
        text_model_name="xlm-roberta-base",
        pose_dims=(75, 2),
        hidden_dim=768,         
        num_layers=4,            
        num_heads=8,            
        num_steps=20,             # number of steps for denoising
        max_num_frames=80,           
        lr_text_encoder=1e-5,     # learning rate for text encoder
        lr_generator=1e-4,        # learning rate for generator
        lr_length_predictor=1e-4, # learning rate for length predictor
        beta_start=0.0001,       
        beta_end=0.02,            
        eta=0.0,                  # Deterministic sampling
        test_results_dir=TEST_RESULTS_DIR,
        total_steps=30000,       
        pose_mean=0.0,            # Mean for standardization
        pose_std=1.0,              # Standard deviation for standardization
        lambda_length=0.5         # Weight for length prediction loss
    ):
        super(IterativeTextGuidedPoseGenerationModel, self).__init__()
        self.save_hyperparameters()
        
        # Register buffers for mean and std
        self.register_buffer('pose_mean', torch.tensor(pose_mean).view(1, -1))
        self.register_buffer('pose_std', torch.tensor(pose_std).view(1, -1))

        self.test_results_dir = test_results_dir
        self.test_filenames = []
        self.test_predictions = []
        
        # Text Encoder
        self.text_encoder = TextEncoder(model_name=text_model_name, embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1)
        text_embedding_dim = hidden_dim  
        
        # Freeze the first few layers
        for i, layer in enumerate(self.text_encoder.base_model.encoder.layer):
            if i < FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False

        # Sign Encoder
        self.sign_encoder = SignEncoder(
            pose_dim=self.hparams.pose_dims[0] * self.hparams.pose_dims[1],
            embed_dim=hidden_dim,
            num_attention_blocks=2,  
            num_heads=num_heads,      
            dropout=0.1,
            max_len=80,
            use_relative_pos=True
        )

        # Step Encoder
        self.step_encoder = StepEncoder(
            num_steps=max_num_frames,
            embed_dim=hidden_dim,
            num_attention_blocks=2,  
            num_heads=num_heads,      
            dropout=0.1
        )

        # Noise Encoder
        self.noise_encoder = NoiseEncoder(
            pose_dim=self.hparams.pose_dims[0] * self.hparams.pose_dims[1],
            embed_dim=hidden_dim,
            num_attention_blocks=2,  
            num_heads=num_heads,      
            dropout=0.1,
            max_len=80,
            use_relative_pos=True
        )

        # Fusion Multihead Attention
        fusion_embed_dim = text_embedding_dim + hidden_dim + hidden_dim + hidden_dim  # Text + Sign + Step + Noise = 768*4=3072
        self.fusion_attention = nn.MultiheadAttention(embed_dim=fusion_embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.fusion_layer_norm = nn.LayerNorm(fusion_embed_dim)
        
        # Sign Predictor with eight Attention Blocks and an MLP to predict poses
        self.sign_predictor = SignPredictor(
            embed_dim=fusion_embed_dim,  
            num_attention_blocks=8,     
            num_heads=num_heads,  
            hidden_dim=3072,            
            dropout=0.1,
            pose_dim=self.hparams.pose_dims[0] * self.hparams.pose_dims[1]
        )
        
        # Length Predictor
        self.length_predictor = LengthPredictor(
            input_dim=text_embedding_dim,
            embed_dim=512,  # Transformer embedding dimension
            num_heads=8,
            num_transformer_layers=2,
            hidden_dims=[1024, 512],  # Increased hidden dimensions for better learning
            output_dim=1,
            dropout=0.2
        )

        # Diffusion Parameters
        self.num_steps = num_steps
        self.eta = eta
        self.betas = torch.linspace(beta_start, beta_end, self.num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # Register buffers for diffusion
        self.register_buffer('diffusion_beta', self.betas.clone())
        self.register_buffer('diffusion_alpha', self.alphas.clone())
        self.register_buffer('diffusion_alpha_cumprod', self.alpha_cumprod.clone())
        self.register_buffer('diffusion_sqrt_alpha_cumprod', self.sqrt_alpha_cumprod.clone())
        self.register_buffer('diffusion_sqrt_one_minus_alpha_cumprod', self.sqrt_one_minus_alpha_cumprod.clone())

        # Define loss functions
        self.loss_fn = nn.MSELoss()
        self.info_nce_fn = info_nce_loss  
        self.length_loss_fn = nn.MSELoss()  # Using MSELoss for length prediction

        # Initialize DDIM Sampler
        self.ddim_sampler = DDIMSampler(
            model=self,
            betas=self.betas.clone(),
            alphas=self.alphas.clone(),
            alpha_cumprod=self.alpha_cumprod.clone(),
            sqrt_alpha_cumprod=self.sqrt_alpha_cumprod.clone(),
            sqrt_one_minus_alpha_cumprod=self.sqrt_one_minus_alpha_cumprod.clone(),
            num_steps=20,  # Set denoising steps to 20
            eta=eta
        )
        
        # Causal Mask for Fusion Attention
        self.register_buffer('causal_mask', torch.tril(torch.ones((self.hparams.max_num_frames, self.hparams.max_num_frames))))  
        
        
    def forward_diffusion(self, pose, t, noise):
        """
        Adds noise to the pose based on the diffusion timestep t.

        Args:
            pose (torch.Tensor): Original pose tensor of shape (batch_size * max_num_frames, pose_dim).
            t (torch.Tensor): Timestep indices of shape (batch_size * max_num_frames,).
            noise (torch.Tensor): Noise tensor of shape (batch_size * max_num_frames, pose_dim).

        Returns:
            torch.Tensor: Noisy pose tensor.
        """
        if torch.any(t >= self.num_steps) or torch.any(t < 0):
            raise ValueError("Timestep t is out of bounds.")

        sqrt_alpha_cumprod_t = self.diffusion_sqrt_alpha_cumprod[t].unsqueeze(1)  # (batch_size * max_num_frames, 1)
        sqrt_one_minus_alpha_cumprod_t = self.diffusion_sqrt_one_minus_alpha_cumprod[t].unsqueeze(1)  # (batch_size * max_num_frames, 1)

        noisy_pose = sqrt_alpha_cumprod_t * pose + sqrt_one_minus_alpha_cumprod_t * noise  # (batch_size * max_num_frames, pose_dim)

        return noisy_pose

    def sample_timesteps(self, batch_size):
        """
        Randomly samples diffusion timesteps for each sample in the batch.

        Args:
            batch_size (int): Number of samples in the batch.

        Returns:
            torch.Tensor: Sampled timesteps of shape (batch_size,).
        """
        return torch.randint(0, self.hparams.num_steps, (batch_size,), device=self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        pose_sequences = batch['pose_sequences']
        masks = batch['masks']
        lengths = batch.get('lengths', None)  # Dynamically computed lengths

        batch_size, max_num_frames, _, _ = pose_sequences.size()

        # Standardize pose sequences
        pose_sequences_normalized = (pose_sequences.view(batch_size, max_num_frames, -1) - self.pose_mean) / self.pose_std  # (batch_size, max_num_frames, pose_dim)

        # Encode text
        text_embedding = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_masks
        )  # (batch_size, embed_dim=768)

        # Predict sequence length
        predicted_length = self.length_predictor(text_embedding)  # (batch_size, 1)
        predicted_length = predicted_length.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
        predicted_length = predicted_length.round().long()  # Convert to integer

        # Encode real pose sequences into embeddings
        sign_embeddings = self.sign_encoder(pose_sequences_normalized, masks)  # (batch_size, embed_dim=768)

        # Encode step embeddings
        step_numbers = torch.arange(max_num_frames, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, max_num_frames)
        step_embeddings = self.step_encoder(step_numbers)  # (batch_size, max_num_frames, embed_dim=768)

        # Encode noise embeddings
        noise = torch.randn(batch_size, max_num_frames, self.hparams.pose_dims[0] * self.hparams.pose_dims[1], device=pose_sequences.device)
        noise_embeddings = self.noise_encoder(noise.view(batch_size * max_num_frames, -1)).view(batch_size, max_num_frames, -1)  # (batch_size, max_num_frames, embed_dim=768)

        # Concatenate all embeddings
        fused_embeddings = torch.cat([
            text_embedding.unsqueeze(1).repeat(1, max_num_frames, 1),  # (batch_size, max_num_frames, 768)
            sign_embeddings.unsqueeze(1).repeat(1, max_num_frames, 1),  # (batch_size, max_num_frames, 768)
            step_embeddings,  # (batch_size, max_num_frames, 768)
            noise_embeddings  # (batch_size, max_num_frames, 768)
        ], dim=-1)  # (batch_size, max_num_frames, 3072)

        # Apply causal self-attention with causal mask
        fused_embeddings, _ = self.fusion_attention(
            fused_embeddings, 
            fused_embeddings, 
            fused_embeddings,
            attn_mask=self.causal_mask # Apply causal mask
        )  # (batch_size, max_num_frames, 3072)
        fused_embeddings = self.fusion_layer_norm(fused_embeddings)  # (batch_size, max_num_frames, 3072)

        # Predict pose
        p_h = self.sign_predictor(fused_embeddings.view(batch_size * max_num_frames, -1))  # (batch_size * max_num_frames, pose_dim)

        # Sample diffusion timesteps
        t = self.sample_timesteps(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Flatten pose sequences and masks
        pose_sequences_flat = pose_sequences_normalized.view(batch_size * max_num_frames, -1)  # (batch_size * max_num_frames, pose_dim)
        masks_flat = masks.view(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Add noise to poses
        noisy_pose = self.forward_diffusion(pose_sequences_flat, t, p_h)  # (batch_size * max_num_frames, pose_dim)

        # Compute alpha_h based on dynamic step length
        delta_h = 1 / torch.log((t + 1).float())  # (batch_size * max_num_frames,)
        alpha_h = (delta_h - 1 / torch.log((t + 2).float())).unsqueeze(1).clamp(0, 1)  # (batch_size * max_num_frames, 1)

        # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev
        # Use ground truth pose_sequences_flat as s_hat_prev
        s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat  # (batch_size * max_num_frames, pose_dim)

        # Introduce random noise to s_hat_h for robustness
        noise_for_loss = torch.randn_like(s_hat_h) * 0.1  # Adjust the noise scale
        s_hat_h_noisy = s_hat_h + noise_for_loss

        # Compute diffusion loss Ld = alpha_h * s^0 + (1 - alpha_h) * s^{h+1}
        loss_ld = (alpha_h * s_hat_h_noisy + (1 - alpha_h) * p_h).pow(2)  # MSE loss
        loss_ld = (loss_ld * masks_flat.unsqueeze(1)).sum() / (masks_flat.sum() + EPSILON)  # Average loss with epsilon for stability

        # Compute InfoNCE loss
        info_nce = self.info_nce_fn(text_embedding, sign_embeddings)  # scalar

        # Compute length prediction loss if lengths are provided
        if lengths is not None:
            length_pred = self.length_predictor(text_embedding)  # (batch_size, 1)
            length_pred = length_pred.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
            length_pred = length_pred.float()
            true_lengths = lengths.float()
            length_loss = self.length_loss_fn(length_pred, true_lengths)  # (batch_size,)
            length_loss = length_loss.mean()
        else:
            length_loss = torch.tensor(0.0, device=self.device)

        # Define weights for each loss component
        lambda_ld = 1.0          # Diffusion loss weight
        lambda_info_nce = 1.0    # InfoNCE loss weight
        lambda_length = self.hparams.lambda_length  # Length prediction loss weight

        # Total loss including length prediction loss
        loss = lambda_ld * loss_ld + lambda_info_nce * info_nce + lambda_length * length_loss

        # Log losses
        if lengths is not None:
            self.log("train_length_loss", length_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.log("train_ld", loss_ld, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_info_nce", info_nce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_length_loss_weight", lambda_length, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        pose_sequences = batch['pose_sequences']
        masks = batch['masks']
        lengths = batch.get('lengths', None)  # Dynamically computed lengths

        batch_size, max_num_frames, _, _ = pose_sequences.size()

        # Standardize pose sequences
        pose_sequences_normalized = (pose_sequences.view(batch_size, max_num_frames, -1) - self.pose_mean) / self.pose_std  # (batch_size, max_num_frames, pose_dim)

        # Encode text
        text_embedding = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_masks
        )  # (batch_size, embed_dim=768)

        # Predict sequence length
        predicted_length = self.length_predictor(text_embedding)  # (batch_size, 1)
        predicted_length = predicted_length.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
        predicted_length = predicted_length.round().long()  # Convert to integer

        # Encode real pose sequences into embeddings
        sign_embeddings = self.sign_encoder(pose_sequences_normalized, masks)  # (batch_size, embed_dim=768)

        # Encode step embeddings
        step_numbers = torch.arange(max_num_frames, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, max_num_frames)
        step_embeddings = self.step_encoder(step_numbers)  # (batch_size, max_num_frames, embed_dim=768)

        # Encode noise embeddings
        noise = torch.randn(batch_size, max_num_frames, self.hparams.pose_dims[0] * self.hparams.pose_dims[1], device=pose_sequences.device)
        noise_embeddings = self.noise_encoder(noise.view(batch_size * max_num_frames, -1)).view(batch_size, max_num_frames, -1)  # (batch_size, max_num_frames, embed_dim=768)

        # Concatenate all embeddings
        fused_embeddings = torch.cat([
            text_embedding.unsqueeze(1).repeat(1, max_num_frames, 1),  # (batch_size, max_num_frames, 768)
            sign_embeddings.unsqueeze(1).repeat(1, max_num_frames, 1),  # (batch_size, max_num_frames, 768)
            step_embeddings,  # (batch_size, max_num_frames, 768)
            noise_embeddings  # (batch_size, max_num_frames, 768)
        ], dim=-1)  # (batch_size, max_num_frames, 3072)

        # Apply causal self-attention with causal mask
        fused_embeddings, _ = self.fusion_attention(
            fused_embeddings, 
            fused_embeddings, 
            fused_embeddings,
            attn_mask=self.causal_mask  # Apply causal mask
        )  # (batch_size, max_num_frames, 3072)
        fused_embeddings = self.fusion_layer_norm(fused_embeddings)  # (batch_size, max_num_frames, 3072)

        # Predict pose
        p_h = self.sign_predictor(fused_embeddings.view(batch_size * max_num_frames, -1))  # (batch_size * max_num_frames, pose_dim)

        # Sample diffusion timesteps
        t = self.sample_timesteps(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Flatten pose sequences and masks
        pose_sequences_flat = pose_sequences_normalized.view(batch_size * max_num_frames, -1)  # (batch_size * max_num_frames, pose_dim)
        masks_flat = masks.view(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Add noise to poses
        noisy_pose = self.forward_diffusion(pose_sequences_flat, t, p_h)  # (batch_size * max_num_frames, pose_dim)

        # Compute alpha_h based on dynamic step length
        delta_h = 1 / torch.log((t + 1).float())  # (batch_size * max_num_frames,)
        alpha_h = (delta_h - 1 / torch.log((t + 2).float())).unsqueeze(1).clamp(0, 1)  # (batch_size * max_num_frames, 1)

        # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev
        # Use ground truth pose_sequences_flat as s_hat_prev
        s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat  # (batch_size * max_num_frames, pose_dim)

        # Introduce random noise to s_hat_h for robustness
        noise_for_loss = torch.randn_like(s_hat_h) * 0.1  # Adjust the noise scale
        s_hat_h_noisy = s_hat_h + noise_for_loss

        # Compute diffusion loss Ld = alpha_h * s^0 + (1 - alpha_h) * s^{h+1}
        loss_ld = (alpha_h * s_hat_h_noisy + (1 - alpha_h) * p_h).pow(2)  # MSE loss
        loss_ld = (loss_ld * masks_flat.unsqueeze(1)).sum() / (masks_flat.sum() + EPSILON)  # Average loss with epsilon for stability

        # Compute InfoNCE loss
        info_nce = self.info_nce_fn(text_embedding, sign_embeddings)  # scalar

        # Compute length prediction loss if lengths are provided
        if lengths is not None:
            length_pred = self.length_predictor(text_embedding)  # (batch_size, 1)
            length_pred = length_pred.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
            length_pred = length_pred.float()
            true_lengths = lengths.float()
            length_loss = self.length_loss_fn(length_pred, true_lengths)  # (batch_size,)
            length_loss = length_loss.mean()
        else:
            length_loss = torch.tensor(0.0, device=self.device)

        # Define weights for each loss component
        lambda_ld = 1.0          # Diffusion loss weight
        lambda_info_nce = 1.0    # InfoNCE loss weight
        lambda_length = self.hparams.lambda_length  # Length prediction loss weight

        # Total loss including length prediction loss
        loss = lambda_ld * loss_ld + lambda_info_nce * info_nce + lambda_length * length_loss

        # Log losses
        if lengths is not None:
            self.log("val_length_loss", length_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.log("val_ld", loss_ld, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_info_nce", info_nce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_length_loss_weight", lambda_length, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        pose_sequences = batch['pose_sequences']
        masks = batch['masks']
        filenames = batch['filenames'] 
        lengths = batch.get('lengths', None)  # Dynamically computed lengths

        batch_size, max_num_frames, _, _ = pose_sequences.size()

        # Standardize pose sequences
        pose_sequences_normalized = (pose_sequences.view(batch_size, max_num_frames, -1) - self.pose_mean) / self.pose_std  # (batch_size, max_num_frames, pose_dim)

        # Encode text
        text_embedding = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_masks
        )  # (batch_size, embed_dim=768)

        # Predict sequence length
        predicted_length = self.length_predictor(text_embedding)  # (batch_size, 1)
        predicted_length = predicted_length.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
        predicted_length = predicted_length.round().long()  # Convert to integer

        # Encode real pose sequences into embeddings
        sign_embeddings = self.sign_encoder(pose_sequences_normalized, masks)  # (batch_size, embed_dim=768)

        # Encode step embeddings
        step_numbers = torch.arange(max_num_frames, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, max_num_frames)
        step_embeddings = self.step_encoder(step_numbers)  # (batch_size, max_num_frames, embed_dim=768)

        # Encode noise embeddings
        noise = torch.randn(batch_size, max_num_frames, self.hparams.pose_dims[0] * self.hparams.pose_dims[1], device=pose_sequences.device)
        noise_embeddings = self.noise_encoder(noise.view(batch_size * max_num_frames, -1)).view(batch_size, max_num_frames, -1)  # (batch_size, max_num_frames, embed_dim=768)

        # Concatenate all embeddings
        fused_embeddings = torch.cat([
            text_embedding.unsqueeze(1).repeat(1, max_num_frames, 1),  # (batch_size, max_num_frames, 768)
            sign_embeddings.unsqueeze(1).repeat(1, max_num_frames, 1),  # (batch_size, max_num_frames, 768)
            step_embeddings,  # (batch_size, max_num_frames, 768)
            noise_embeddings  # (batch_size, max_num_frames, 768)
        ], dim=-1)  # (batch_size, max_num_frames, 3072)

        # Apply causal self-attention with causal mask
        fused_embeddings, _ = self.fusion_attention(
            fused_embeddings, 
            fused_embeddings, 
            fused_embeddings,
            attn_mask=self.causal_mask  # Apply causal mask
        )  # (batch_size, max_num_frames, 3072)
        fused_embeddings = self.fusion_layer_norm(fused_embeddings)  # (batch_size, max_num_frames, 3072)

        # Predict pose
        p_h = self.sign_predictor(fused_embeddings.view(batch_size * max_num_frames, -1))  # (batch_size * max_num_frames, pose_dim)

        # Sample diffusion timesteps
        t = self.sample_timesteps(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Flatten pose sequences and masks
        pose_sequences_flat = pose_sequences_normalized.view(batch_size * max_num_frames, -1)  # (batch_size * max_num_frames, pose_dim)
        masks_flat = masks.view(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Add noise to poses
        noisy_pose = self.forward_diffusion(pose_sequences_flat, t, p_h)  # (batch_size * max_num_frames, pose_dim)

        # Compute alpha_h based on dynamic step length
        delta_h = 1 / torch.log((t + 1).float())  # (batch_size * max_num_frames,)
        alpha_h = (delta_h - 1 / torch.log((t + 2).float())).unsqueeze(1).clamp(0, 1)  # (batch_size * max_num_frames, 1)

        # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev
        # Use ground truth pose_sequences_flat as s_hat_prev
        s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat  # (batch_size * max_num_frames, pose_dim)

        # Introduce random noise to s_hat_h for robustness
        noise_for_loss = torch.randn_like(s_hat_h) * 0.1  # Adjust the noise scale
        s_hat_h_noisy = s_hat_h + noise_for_loss

        # Compute diffusion loss Ld = alpha_h * s^0 + (1 - alpha_h) * s^{h+1}
        loss_ld = (alpha_h * s_hat_h_noisy + (1 - alpha_h) * p_h).pow(2)  # MSE loss
        loss_ld = (loss_ld * masks_flat.unsqueeze(1)).sum() / (masks_flat.sum() + EPSILON)  # Average loss with epsilon for stability

        # Compute InfoNCE loss
        info_nce = self.info_nce_fn(text_embedding, sign_embeddings)  # scalar

        # Compute length prediction loss if lengths are provided
        if lengths is not None:
            length_pred = self.length_predictor(text_embedding)  # (batch_size, 1)
            length_pred = length_pred.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
            length_pred = length_pred.float()
            true_lengths = lengths.float()
            length_loss = self.length_loss_fn(length_pred, true_lengths)  # (batch_size,)
            length_loss = length_loss.mean()
        else:
            length_loss = torch.tensor(0.0, device=self.device)

        # Define weights for each loss component
        lambda_ld = 1.0          # Diffusion loss weight
        lambda_info_nce = 1.0    # InfoNCE loss weight
        lambda_length = self.hparams.lambda_length  # Length prediction loss weight

        # Total loss including length prediction loss
        loss = lambda_ld * loss_ld + lambda_info_nce * info_nce + lambda_length * length_loss

        # Log losses
        if lengths is not None:
            self.log("test_length_loss", length_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.log("test_ld", loss_ld, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_info_nce", info_nce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_length_loss_weight", lambda_length, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Inverse standardize the predicted pose
        pose_predicted = s_hat_h.view(batch_size, max_num_frames, self.hparams.pose_dims[0], self.hparams.pose_dims[1])  # (batch_size, max_num_frames, 75, 2)
        pose_predicted = pose_predicted * self.pose_std + self.pose_mean  # Inverse standardization
        pose_predicted = pose_predicted.cpu().numpy()  # Convert to NumPy array

        for i in range(batch_size):
            filename = filenames[i]
            prediction = pose_predicted[i]
            self.test_filenames.append(filename)
            self.test_predictions.append(prediction)

        return loss

    def on_test_epoch_end(self):
        """
        Actions to perform at the end of the test epoch.
        Saves the predicted pose sequences as .npy files.
        """
        os.makedirs(self.test_results_dir, exist_ok=True)

        for filename, prediction in zip(self.test_filenames, self.test_predictions):
            save_path = os.path.join(self.test_results_dir, f"{filename}_prediction.npy")
            np.save(save_path, prediction)
            print(f"Saved prediction to {save_path}")

        # Clear lists after saving
        self.test_filenames = []
        self.test_predictions = []

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Optimizer and scheduler configurations.
        """
        # Separate parameters for the text encoder, generator, and length predictor
        text_params = [param for param in self.text_encoder.parameters() if param.requires_grad]
        generator_params = list(self.sign_encoder.parameters()) + \
                           list(self.step_encoder.parameters()) + \
                           list(self.noise_encoder.parameters()) + \
                           list(self.sign_predictor.parameters())  
        length_params = list(self.length_predictor.parameters())

        optimizer = optim.AdamW([
            {'params': text_params, 'lr': self.hparams.lr_text_encoder},
            {'params': generator_params, 'lr': self.hparams.lr_generator},
            {'params': length_params, 'lr': self.hparams.lr_length_predictor}
        ], weight_decay=1e-5)

        # Define a cosine scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * self.hparams.total_steps),
            num_training_steps=self.hparams.total_steps
        )

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }

    def generate_pose(self, text_input_ids, text_attention_masks, num_steps=None):
        """
        Generates poses using the DDIM sampler based on predicted length in an autoregressive manner.

        Args:
            text_input_ids (torch.Tensor): Input IDs for the text.
            text_attention_masks (torch.Tensor): Attention masks for the text.
            num_steps (int, optional): Number of steps to generate. Defaults to None.

        Returns:
            tuple: Generated pose sequence and predicted number of steps.
        """
        self.eval()
        with torch.no_grad():
            # Encode text
            text_embedding = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_masks
            )  # (batch_size, embed_dim=768)

            # Predict sequence length
            predicted_length = self.length_predictor(text_embedding)  # (batch_size, 1)
            predicted_length = predicted_length.squeeze(1).clamp(0, 100)  # Clamp to [0, 100]
            predicted_length = predicted_length.round().long()  # Convert to integer

            # If num_steps is provided, override predicted_length
            if num_steps is not None:
                num_steps_pred = torch.full((predicted_length.size(0),), num_steps, device=self.device, dtype=torch.long)
            else:
                num_steps_pred = predicted_length  # (batch_size,)

            # Determine the maximum number of steps in the batch
            max_pred_steps = num_steps_pred.max().item()

            # Encode step numbers
            step_numbers = torch.arange(0, max_pred_steps, device=self.device).unsqueeze(0).repeat(text_embedding.size(0), 1)  # (batch_size, max_pred_steps)
            step_embeddings = self.step_encoder(step_numbers)  # (batch_size, max_pred_steps, embed_dim=768)

            # Initialize noise (starting from pure noise)
            noise = torch.randn(text_embedding.size(0), max_pred_steps, self.hparams.pose_dims[0] * self.hparams.pose_dims[1], device=self.device)  # (batch_size, max_pred_steps, pose_dim)
            noise_embeddings = self.noise_encoder(noise.view(-1, self.hparams.pose_dims[0] * self.hparams.pose_dims[1]))  # (batch_size * max_pred_steps, embed_dim=768)
            noise_embeddings = noise_embeddings.view(text_embedding.size(0), max_pred_steps, -1)  # (batch_size, max_pred_steps, embed_dim=768)

            # Use DDIM Sampler to sample pose
            pose_sequence = self.ddim_sampler.sample(text_embedding, step_embeddings, noise_embeddings, device=self.device, num_steps=max_pred_steps)  # (batch_size, num_steps, 75, 2)

            # Inverse standardize the generated pose
            pose_sequence = pose_sequence * self.pose_std + self.pose_mean  # (batch_size, num_steps, 75, 2)

        return pose_sequence, num_steps_pred

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F

# Constants Definition
EPSILON = 1e-4
START_LEARNING_RATE_TEXT_ENCODER = 1e-5
START_LEARNING_RATE_GENERATOR = 1e-4
MAX_SEQ_LEN = 200
FREEZE_LAYERS = 6
MAX_LENGTH_PREDICTION = 1000  

# Function to interpolate and fill NaN values
def interpolate_nans(pose_array):
    """
    Interpolates NaN values along the frame axis in the pose array.
    """
    num_frames, num_keypoints, num_coords = pose_array.shape
    for k in range(num_keypoints):
        for c in range(num_coords):
            coord_data = pose_array[:, k, c]
            nans = np.isnan(coord_data)
            if np.any(nans):
                not_nans = ~nans
                if np.sum(not_nans) == 0:
                    # If all values are NaN, set to zero
                    pose_array[:, k, c] = 0.0
                else:
                    interp = np.interp(
                        np.where(nans)[0],
                        np.where(not_nans)[0],
                        coord_data[not_nans]
                    )
                    pose_array[:, k, c][nans] = interp
    return pose_array

# Custom Dataset Class
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
        text_inputs = self.tokenizer(text=text, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)
        text_input_ids = text_inputs['input_ids'].squeeze()  # (seq_len,)
        text_attention_mask = text_inputs['attention_mask'].squeeze()  # (seq_len,)

        # Load pose data
        pose_sequence = np.load(pose_path, allow_pickle=True).astype(np.float32)  # (num_frames, 75, 2)

        # Interpolate NaN values
        pose_sequence = interpolate_nans(pose_sequence)  # (num_frames, 75, 2)

        # Convert to torch.Tensor
        pose_sequence = torch.from_numpy(pose_sequence)  # (num_frames, 75, 2)

        # Apply transformation (e.g., normalization)
        if self.transform:
            pose_sequence = self.transform(pose_sequence)

        return text_input_ids, text_attention_mask, pose_sequence, filename

# Custom collate function
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length pose sequences and text inputs.
    """
    text_input_ids, text_attention_masks, pose_sequences, filenames = zip(*batch)

    # Stack text input IDs and attention masks
    text_input_ids = torch.stack(text_input_ids, dim=0)  # (batch_size, seq_len)
    text_attention_masks = torch.stack(text_attention_masks, dim=0)  # (batch_size, seq_len)

    # Find the maximum number of frames in the batch
    max_num_frames = max([pose_seq.size(0) for pose_seq in pose_sequences])

    # Pad pose sequences and create masks
    padded_pose_sequences = []
    masks = []
    for pose_seq in pose_sequences:
        num_frames = pose_seq.size(0)
        if num_frames < max_num_frames:
            padding_frames = max_num_frames - num_frames
            padding = torch.zeros((padding_frames, pose_seq.size(1), pose_seq.size(2)), dtype=pose_seq.dtype)
            padded_pose = torch.cat([pose_seq, padding], dim=0)
            mask = torch.cat([torch.ones(num_frames, dtype=torch.bool), torch.zeros(padding_frames, dtype=torch.bool)], dim=0)
        else:
            padded_pose = pose_seq
            mask = torch.ones(max_num_frames, dtype=torch.bool)
        padded_pose_sequences.append(padded_pose)
        masks.append(mask)

    padded_pose_sequences = torch.stack(padded_pose_sequences, dim=0)  # (batch_size, max_num_frames, 75, 2)
    masks = torch.stack(masks, dim=0)  # (batch_size, max_num_frames)

    return {
        'text_input_ids': text_input_ids,             # (batch_size, seq_len)
        'text_attention_masks': text_attention_masks, # (batch_size, seq_len)
        'pose_sequences': padded_pose_sequences,      # (batch_size, max_num_frames, 75, 2)
        'masks': masks,                               # (batch_size, max_num_frames)
        'filenames': filenames                                
    }

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Sign Encoder Class
class SignEncoder(nn.Module):
    def __init__(self, pose_dim, embed_dim=768, num_attention_blocks=2, num_heads=8, dropout=0.1):
        super(SignEncoder, self).__init__()
        # Linear layer to map pose_dim dimensions to embed_dim dimensions
        self.pose_projection = nn.Linear(pose_dim, embed_dim)  # From pose_dim to embed_dim
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim)
        # Use ModuleList instead of Sequential
        self.attention_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True) for _ in range(num_attention_blocks)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling along the frame dimension

    def forward(self, pose_sequences, masks):
        """
        pose_sequences: (batch_size, max_num_frames, 75, 2)
        masks: (batch_size, max_num_frames)
        """
        # Flatten pose_sequences to (batch_size, max_num_frames, pose_dim)
        batch_size, max_num_frames, num_keypoints, num_coords = pose_sequences.size()
        pose_dim = num_keypoints * num_coords  # 75 * 2 = 150
        pose_sequences = pose_sequences.view(batch_size, max_num_frames, pose_dim)  # (batch_size, max_num_frames, 150)

        # Pass through linear layer to map pose_dim dimensions to embed_dim dimensions
        x = self.pose_projection(pose_sequences)  # (batch_size, max_num_frames, embed_dim)

        # Apply positional encoding
        x = self.positional_encoding(x)  # (batch_size, max_num_frames, embed_dim)

        # Apply TransformerEncoderLayer sequentially, passing src_key_padding_mask
        key_padding_mask = ~masks  # (batch_size, max_num_frames)
        for layer in self.attention_blocks:
            x = layer(x, src_key_padding_mask=key_padding_mask)  # (batch_size, max_num_frames, embed_dim)

        # Pooling
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, max_num_frames)
        x = self.pool(x).squeeze(-1)  # (batch_size, embed_dim)

        # MLP
        x = self.mlp(x)  # (batch_size, embed_dim)

        return x  # Sign embeddings

# Step Encoder Class
class StepEncoder(nn.Module):
    def __init__(self, num_steps, embed_dim=768, num_attention_blocks=2, num_heads=8, dropout=0.1):
        super(StepEncoder, self).__init__()
        self.step_embedding = nn.Embedding(num_steps, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim)
        # Use ModuleList instead of Sequential
        self.attention_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True) for _ in range(num_attention_blocks)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, step_numbers):
        """
        step_numbers: (batch_size, max_num_frames)
        """
        # Get step embeddings
        x = self.step_embedding(step_numbers)  # (batch_size, max_num_frames, embed_dim)

        # Apply positional encoding
        x = self.positional_encoding(x)  # (batch_size, max_num_frames, embed_dim)

        # Apply TransformerEncoderLayer sequentially
        for layer in self.attention_blocks:
            x = layer(x)  # (batch_size, max_num_frames, embed_dim)

        # MLP
        x = self.mlp(x)  # (batch_size, max_num_frames, embed_dim)

        return x  # Step embeddings

# Noise Encoder Class
class NoiseEncoder(nn.Module):
    def __init__(self, pose_dim, embed_dim=768, num_attention_blocks=2, num_heads=8, dropout=0.1):
        super(NoiseEncoder, self).__init__()
        self.noise_embedding = nn.Linear(pose_dim, embed_dim)  # Adjusted to take pose_dim noise
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim)
        # Use ModuleList instead of Sequential
        self.attention_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True) for _ in range(num_attention_blocks)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, noise):
        """
        noise: (batch_size * max_num_frames, pose_dim)
        """
        x = self.noise_embedding(noise)  # (batch_size * max_num_frames, embed_dim)
        x = self.positional_encoding(x.unsqueeze(1)).squeeze(1)  # (batch_size * max_num_frames, embed_dim)
        for layer in self.attention_blocks:
            x = layer(x.unsqueeze(1)).squeeze(1)  # (batch_size * max_num_frames, embed_dim)
        x = self.mlp(x)  # (batch_size * max_num_frames, embed_dim)
        return x  # Noise embeddings

# Sign Predictor Class with Causal Self-Attention
class SignPredictor(nn.Module):
    def __init__(self, embed_dim=2304, num_attention_blocks=6, num_heads=12, hidden_dim=1536, dropout=0.1, pose_dim=150):
        """
        embed_dim = text_embed_dim + step_embed_dim + noise_embed_dim
        """
        super(SignPredictor, self).__init__()
        self.causal_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_attention_blocks,
            norm=nn.LayerNorm(embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pose_dim)
        )

    def forward(self, text_embeddings, step_embeddings, noise_embeddings):
        """
        text_embeddings: (batch_size * max_num_frames, text_embed_dim)
        step_embeddings: (batch_size * max_num_frames, step_embed_dim)
        noise_embeddings: (batch_size * max_num_frames, noise_embed_dim)
        """
        # Concatenate embeddings
        x = torch.cat([text_embeddings, step_embeddings, noise_embeddings], dim=1)  # (batch_size * max_num_frames, embed_dim)

        # Reshape to (batch_size, max_num_frames, embed_dim)
        batch_size = text_embeddings.size(0)
        max_num_frames = step_embeddings.size(0) // batch_size
        x = x.view(batch_size, max_num_frames, -1)  # (batch_size, max_num_frames, embed_dim)

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()  # (seq_len, seq_len)

        # Pass through causal transformer
        x = self.causal_attention(x, mask=causal_mask)  # (batch_size, max_num_frames, embed_dim)

        # Flatten back to (batch_size * max_num_frames, embed_dim)
        x = x.view(batch_size * max_num_frames, -1)  # (batch_size * max_num_frames, embed_dim)

        # Predict pose
        pose_pred = self.mlp(x)  # (batch_size * max_num_frames, pose_dim)
        return pose_pred

# Length Predictor Class with Softplus and Proper Initialization
class LengthPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2):
        super(LengthPredictor, self).__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))  # 输出一个长度预测值
        self.mlp = nn.Sequential(*layers)
        
        # Initialize the final layer to produce small outputs
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        """
        x: (batch_size, embed_dim)
        """
        length_pred = self.mlp(x)  # (batch_size, 1)
        length_pred = F.softplus(length_pred)  # Ensure length is positive
        return length_pred.squeeze(-1)  # (batch_size,)

# InfoNCE Loss Function
def info_nce_loss(text_embeddings, sign_embeddings, temperature=0.07):
    """
    InfoNCE loss function to align text embeddings with sign embeddings.
    Cross-modal alignment without normalization.
    """
    batch_size = text_embeddings.size(0)
    similarity_matrix = torch.matmul(text_embeddings, sign_embeddings.T)  # (batch_size, batch_size)
    similarity_matrix = similarity_matrix / temperature

    labels = torch.arange(batch_size).to(text_embeddings.device)
    loss_fn = nn.CrossEntropyLoss()
    loss_text_to_sign = loss_fn(similarity_matrix, labels)
    loss_sign_to_text = loss_fn(similarity_matrix.T, labels)
    loss = (loss_text_to_sign + loss_sign_to_text) / 2
    return loss

# Embedding Consistency Learning Loss Function
def ecl_loss(generated_sign_embeddings, sign_embeddings):
    """
    Embedding Consistency Learning loss function to ensure generated sign embeddings match real sign embeddings.
    Combines MSE loss and cosine similarity loss.
    """
    mse_loss = nn.functional.mse_loss(generated_sign_embeddings, sign_embeddings)
    cosine_loss = 1 - nn.functional.cosine_similarity(generated_sign_embeddings, sign_embeddings).mean()
    loss = mse_loss + cosine_loss
    return loss

# DDIM Sampler Class with Correct Causal Sampling
class DDIMSampler:
    def __init__(self, model, betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, num_steps=1000, eta=0.0):
        """
        Initializes the DDIM sampler.
        """
        self.model = model
        self.num_steps = num_steps
        self.eta = eta  # Eta=0 for deterministic DDIM

        self.betas = betas.clone()
        self.alphas = alphas.clone()
        self.alpha_cumprod = alpha_cumprod.clone()
        self.sqrt_alpha_cumprod = sqrt_alpha_cumprod.clone()
        self.sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.clone()

        # Precompute DDIM parameters
        self.alpha_bar = self.alpha_cumprod.clone()
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=self.alpha_bar.device), self.alpha_bar[:-1]])  # (num_steps,)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).clone()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar).clone()

        # Compute DDIM parameters
        self.sqrt_alpha_bar_prev = torch.sqrt(self.alpha_bar_prev).clone()

        self.ddim_sigma = (
            self.eta
            * torch.sqrt(
                (1.0 - self.alpha_bar_prev)
                / (1.0 - self.alpha_bar)
                * (1.0 - self.alpha_bar / self.alpha_bar_prev)
            )
        ).clone()
        self.ddim_mu = (
            self.sqrt_alpha_bar_prev * torch.sqrt(self.alpha_bar)
            + (1.0 - self.alpha_bar_prev)
            * torch.sqrt(self.alpha_bar)
            / (1.0 - self.alpha_bar)
        ).clone()

    def sample(self, text_embeddings, step_embeddings, noise_embeddings, device, num_steps=None):
        """
        Performs DDIM sampling to generate poses.
        """
        if num_steps is None:
            num_steps = self.num_steps

        batch_size = text_embeddings.size(0)
        pose_dim = self.model.hparams.pose_dims[0] * self.model.hparams.pose_dims[1]

        # Initialize the pose sequence with pure noise
        pose = torch.randn(batch_size, pose_dim, device=device)  # (batch_size, pose_dim)

        # Initialize s_hat_prev to zero
        s_hat_prev = torch.zeros_like(pose).to(device)

        for t_step in reversed(range(num_steps)):
            # Get current step embeddings
            current_step_embeddings = step_embeddings[:, t_step, :]  # (batch_size, hidden_dim)

            # Get current noise embeddings
            current_noise_embeddings = noise_embeddings[:, t_step, :]  # (batch_size, embed_dim)

            # Predict pose
            p_h = self.model.sign_predictor(
                text_embeddings=text_embeddings,
                step_embeddings=current_step_embeddings,
                noise_embeddings=current_noise_embeddings
            )  # (batch_size, pose_dim)

            # Compute alpha_h based on the paper's schedule
            delta_h = 1 / torch.log((t_step + 1).float() + 1)
            alpha_h = (delta_h - (1 / torch.log((t_step + 2).float() + 1))).unsqueeze(1).clamp(0, 1)  # (batch_size, 1)

            # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev
            s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev  # (batch_size, pose_dim)

            # Compute mu and sigma
            mu = (
                self.ddim_mu[t_step] * s_hat_h
                + (1.0 - self.alpha_bar_prev[t_step]) * p_h
                / self.sqrt_one_minus_alpha_bar[t_step]
            )
            sigma = self.ddim_sigma[t_step]

            # Add noise if eta > 0
            if t_step > 0:
                noise = torch.randn_like(pose)
            else:
                noise = torch.zeros_like(pose)

            # Update pose
            pose = mu + sigma * noise

            # Update s_hat_prev
            s_hat_prev = s_hat_h

        # Reshape to (batch_size, 75, 2)
        pose = pose.view(batch_size, 75, 2)
        return pose

# Main Model Class
class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
    def __init__(
        self,
        text_model_name="xlm-roberta-base",  
        pose_dims=(75, 2),
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
        num_steps=1000,
        lr_text_encoder=START_LEARNING_RATE_TEXT_ENCODER,
        lr_generator=START_LEARNING_RATE_GENERATOR,
        beta_start=1e-4,
        beta_end=0.02,
        eta=0.0,
        test_results_dir='test_predictions',
        total_steps=100000
    ):
        super(IterativeTextGuidedPoseGenerationModel, self).__init__()
        self.save_hyperparameters()
        
        self.test_results_dir = test_results_dir
        self.test_filenames = []
        self.test_predictions = []
        
        # Text Encoder: XLM-RoBERTa
        self.text_encoder = XLMRobertaModel.from_pretrained(text_model_name)
        text_embedding_dim = self.text_encoder.config.hidden_size

        # Freeze the first FREEZE_LAYERS layers of the text encoder
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False

        # Pose Encoder
        self.sign_encoder = SignEncoder(
            pose_dim=self.hparams.pose_dims[0] * self.hparams.pose_dims[1],
            embed_dim=hidden_dim,
            num_attention_blocks=2,
            num_heads=8,
            dropout=0.1
        )

        # Step Encoder
        self.step_encoder = StepEncoder(
            num_steps=self.hparams.num_steps,
            embed_dim=hidden_dim,
            num_attention_blocks=2,
            num_heads=8,
            dropout=0.1
        )

        # Noise Encoder
        self.noise_encoder = NoiseEncoder(
            pose_dim=self.hparams.pose_dims[0] * self.hparams.pose_dims[1],
            embed_dim=hidden_dim,
            num_attention_blocks=2,
            num_heads=8,
            dropout=0.1
        )

        # Sign Predictor
        self.sign_predictor = SignPredictor(
            embed_dim=text_embedding_dim + hidden_dim + hidden_dim,  # text + step + noise
            num_attention_blocks=6,
            num_heads=12,
            hidden_dim=1536,
            dropout=0.1,
            pose_dim=self.hparams.pose_dims[0] * self.hparams.pose_dims[1]
        )

        # Length Predictor
        self.length_predictor = LengthPredictor(input_dim=hidden_dim)

        # Diffusion Parameters
        self.num_steps = num_steps
        self.eta = eta
        self.betas = torch.linspace(beta_start, beta_end, self.num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # Register buffers
        self.register_buffer('diffusion_beta', self.betas.clone())
        self.register_buffer('diffusion_alpha', self.alphas.clone())
        self.register_buffer('diffusion_alpha_cumprod', self.alpha_cumprod.clone())
        self.register_buffer('diffusion_sqrt_alpha_cumprod', self.sqrt_alpha_cumprod.clone())
        self.register_buffer('diffusion_sqrt_one_minus_alpha_cumprod', self.sqrt_one_minus_alpha_cumprod.clone())

        # Define loss functions
        self.loss_fn = nn.MSELoss()
        self.info_nce_fn = info_nce_loss
        self.ecl_fn = ecl_loss
        self.length_loss_fn = nn.L1Loss()  # Changed to L1Loss for robustness

        # Initialize DDIM Sampler (do not call .to(self.device))
        self.ddim_sampler = DDIMSampler(
            model=self,
            betas=self.betas.clone(),
            alphas=self.alphas.clone(),
            alpha_cumprod=self.alpha_cumprod.clone(),
            sqrt_alpha_cumprod=self.sqrt_alpha_cumprod.clone(),
            sqrt_one_minus_alpha_cumprod=self.sqrt_one_minus_alpha_cumprod.clone(),
            num_steps=num_steps,
            eta=eta
        )

    def forward_diffusion(self, pose, t, noise):
        """
        Adds noise to the pose based on the diffusion timestep t.
        """
        if torch.any(t >= self.num_steps) or torch.any(t < 0):
            raise ValueError("Timestep t is out of bounds.")

        sqrt_alpha_cumprod_t = self.diffusion_sqrt_alpha_cumprod[t].unsqueeze(1)  # (batch_size * max_num_frames, 1)
        sqrt_one_minus_alpha_cumprod_t = self.diffusion_sqrt_one_minus_alpha_cumprod[t].unsqueeze(1)  # (batch_size * max_num_frames, 1)

        # Ensure noise is broadcasted correctly
        noisy_pose = sqrt_alpha_cumprod_t * pose + sqrt_one_minus_alpha_cumprod_t * noise  # (batch_size * max_num_frames, pose_dim)

        return noisy_pose

    def sample_timesteps(self, batch_size):
        """
        Randomly samples diffusion timesteps for each sample in the batch.
        """
        return torch.randint(0, self.hparams.num_steps, (batch_size,), device=self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        pose_sequences = batch['pose_sequences']
        masks = batch['masks']

        batch_size, max_num_frames, _, _ = pose_sequences.size()

        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_masks
        )
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token, (batch_size, text_embedding_dim)

        # Encode real pose sequences into embeddings
        sign_embeddings = self.sign_encoder(pose_sequences, masks)  # (batch_size, embed_dim)

        # Predict sequence length
        predicted_length = self.length_predictor(sign_embeddings)  # (batch_size,)
        # Assume true length is the sum of mask
        true_length = masks.sum(dim=1).float()  # (batch_size,)
        loss_length = self.length_loss_fn(predicted_length, true_length)

        # Sample noise as same dimension as pose
        pose_dim = self.hparams.pose_dims[0] * self.hparams.pose_dims[1]
        noise = torch.randn(batch_size * max_num_frames, pose_dim, device=pose_sequences.device)  # (batch_size * max_num_frames, pose_dim)

        # Sample diffusion timesteps
        t = self.sample_timesteps(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Flatten pose sequences
        pose_sequences_flat = pose_sequences.view(batch_size * max_num_frames, pose_dim)  # (batch_size * max_num_frames, pose_dim)
        masks_flat = masks.view(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Add noise to poses
        noisy_pose = self.forward_diffusion(pose_sequences_flat, t, noise)  # (batch_size * max_num_frames, pose_dim)

        # Encode timestep numbers
        h = t.view(batch_size, max_num_frames)  # (batch_size, max_num_frames)
        step_embeddings = self.step_encoder(h)  # (batch_size, max_num_frames, hidden_dim)

        # Encode noise
        noise_embeddings = self.noise_encoder(noise)  # (batch_size * max_num_frames, embed_dim)

        # Expand text embeddings to match the number of frames
        text_embedding_expanded = text_embedding.unsqueeze(1).repeat(1, max_num_frames, 1)  # (batch_size, max_num_frames, text_embed_dim)

        # Predict pose
        p_h = self.sign_predictor(
            text_embeddings=text_embedding_expanded.view(batch_size * max_num_frames, -1),
            step_embeddings=step_embeddings.view(batch_size * max_num_frames, -1),
            noise_embeddings=noise_embeddings
        )  # (batch_size * max_num_frames, pose_dim)

        # Compute alpha_h based on the paper's schedule
        delta_h = 1 / torch.log((t + 1).float() + 1)
        alpha_h = (delta_h - (1 / torch.log((t + 2).float() + 1))).unsqueeze(1).clamp(0, 1)  # (batch_size * max_num_frames, 1)

        # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat
        s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat  # (batch_size * max_num_frames, pose_dim)

        # Compute diffusion loss Ld
        loss_noise = self.loss_fn(p_h, noise)  # MSE for noise prediction
        loss_pose = self.loss_fn(s_hat_h, pose_sequences_flat)  # MSE for pose prediction
        loss_ld = (alpha_h.squeeze(1) * loss_noise + (1 - alpha_h.squeeze(1)) * loss_pose)  # Weighted loss
        loss_ld = (loss_ld * masks_flat).sum() / (masks_flat.sum() + EPSILON)  # Average loss with epsilon for stability

        # Compute Embedding Consistency Learning loss Lecl
        generated_pose = s_hat_h.view(batch_size, max_num_frames, self.hparams.pose_dims[0], self.hparams.pose_dims[1])  # (batch_size, max_num_frames, 75, 2)
        generated_sign_embeddings = self.sign_encoder(generated_pose, masks)  # (batch_size, embed_dim)
        ecl = self.ecl_fn(generated_sign_embeddings, sign_embeddings)  # scalar

        # Compute InfoNCE loss Lnce
        info_nce = self.info_nce_fn(text_embedding, sign_embeddings)  # scalar

        # Total loss
        loss = loss_ld + info_nce + ecl + loss_length  # λ1=λ2=λ3=λ4=1

        # Compute predicted length statistics
        length_diff = torch.abs(predicted_length - true_length)
        avg_length_diff = length_diff.mean()

        # Log losses and length statistics
        self.log("train_ld", loss_ld, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_info_nce", info_nce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ecl", ecl, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_length_loss", loss_length, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_length_diff", avg_length_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_predicted_length_mean", predicted_length.mean(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_predicted_length_std", predicted_length.std(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        pose_sequences = batch['pose_sequences']
        masks = batch['masks']

        batch_size, max_num_frames, _, _ = pose_sequences.size()

        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_masks
        )
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token, (batch_size, text_embedding_dim)

        # Encode real pose sequences into embeddings
        sign_embeddings = self.sign_encoder(pose_sequences, masks)  # (batch_size, embed_dim)

        # Predict sequence length
        predicted_length = self.length_predictor(sign_embeddings)  # (batch_size,)
        # Assume true length is the sum of mask
        true_length = masks.sum(dim=1).float()  # (batch_size,)
        loss_length = self.length_loss_fn(predicted_length, true_length)

        # Sample noise as same dimension as pose
        pose_dim = self.hparams.pose_dims[0] * self.hparams.pose_dims[1]
        noise = torch.randn(batch_size * max_num_frames, pose_dim, device=pose_sequences.device)  # (batch_size * max_num_frames, pose_dim)

        # Sample diffusion timesteps
        t = self.sample_timesteps(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Flatten pose sequences
        pose_sequences_flat = pose_sequences.view(batch_size * max_num_frames, pose_dim)  # (batch_size * max_num_frames, pose_dim)
        masks_flat = masks.view(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Add noise to poses
        noisy_pose = self.forward_diffusion(pose_sequences_flat, t, noise)  # (batch_size * max_num_frames, pose_dim)

        # Encode timestep numbers
        h = t.view(batch_size, max_num_frames)  # (batch_size, max_num_frames)
        step_embeddings = self.step_encoder(h)  # (batch_size, max_num_frames, hidden_dim)

        # Encode noise
        noise_embeddings = self.noise_encoder(noise)  # (batch_size * max_num_frames, embed_dim)

        # Expand text embeddings to match the number of frames
        text_embedding_expanded = text_embedding.unsqueeze(1).repeat(1, max_num_frames, 1)  # (batch_size, max_num_frames, text_embed_dim)

        # Predict pose
        p_h = self.sign_predictor(
            text_embeddings=text_embedding_expanded.view(batch_size * max_num_frames, -1),
            step_embeddings=step_embeddings.view(batch_size * max_num_frames, -1),
            noise_embeddings=noise_embeddings
        )  # (batch_size * max_num_frames, pose_dim)

        # Compute alpha_h based on the paper's schedule
        delta_h = 1 / torch.log((t + 1).float() + 1)
        alpha_h = (delta_h - (1 / torch.log((t + 2).float() + 1))).unsqueeze(1).clamp(0, 1)  # (batch_size * max_num_frames, 1)

        # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat
        s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat  # (batch_size * max_num_frames, pose_dim)

        # Compute diffusion loss Ld
        loss_noise = self.loss_fn(p_h, noise)  # MSE for noise prediction
        loss_pose = self.loss_fn(s_hat_h, pose_sequences_flat)  # MSE for pose prediction
        loss_ld = (alpha_h.squeeze(1) * loss_noise + (1 - alpha_h.squeeze(1)) * loss_pose)  # Weighted loss
        loss_ld = (loss_ld * masks_flat).sum() / (masks_flat.sum() + EPSILON)  # Average loss with epsilon for stability

        # Compute Embedding Consistency Learning loss Lecl
        generated_pose = s_hat_h.view(batch_size, max_num_frames, self.hparams.pose_dims[0], self.hparams.pose_dims[1])  # (batch_size, max_num_frames, 75, 2)
        generated_sign_embeddings = self.sign_encoder(generated_pose, masks)  # (batch_size, embed_dim)
        ecl = self.ecl_fn(generated_sign_embeddings, sign_embeddings)  # scalar

        # Compute InfoNCE loss Lnce
        info_nce = self.info_nce_fn(text_embedding, sign_embeddings)  # scalar

        # Total loss
        loss = loss_ld + info_nce + ecl + loss_length  # λ1=λ2=λ3=λ4=1

        # Compute predicted length statistics
        length_diff = torch.abs(predicted_length - true_length)
        avg_length_diff = length_diff.mean()

        # Log losses and length statistics
        self.log("val_ld", loss_ld, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_info_nce", info_nce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ecl", ecl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_length_loss", loss_length, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_length_diff", avg_length_diff, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_predicted_length_mean", predicted_length.mean(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_predicted_length_std", predicted_length.std(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
        
        
    def test_step(self, batch, batch_idx):

        text_input_ids = batch['text_input_ids']
        text_attention_masks = batch['text_attention_masks']
        pose_sequences = batch['pose_sequences']
        masks = batch['masks']
        filenames = batch['filenames'] 

        batch_size, max_num_frames, _, _ = pose_sequences.size()

        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_masks
        )
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token, (batch_size, text_embedding_dim)

        # Encode real pose sequences into embeddings
        sign_embeddings = self.sign_encoder(pose_sequences, masks)  # (batch_size, embed_dim)

        # Predict sequence length
        predicted_length = self.length_predictor(sign_embeddings)  # (batch_size,)
        # Assume true length is the sum of mask
        true_length = masks.sum(dim=1).float()  # (batch_size,)
        loss_length = self.length_loss_fn(predicted_length, true_length)

        # Sample noise as same dimension as pose
        pose_dim = self.hparams.pose_dims[0] * self.hparams.pose_dims[1]
        noise = torch.randn(batch_size * max_num_frames, pose_dim, device=pose_sequences.device)  # (batch_size * max_num_frames, pose_dim)

        # Sample diffusion timesteps
        t = self.sample_timesteps(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Flatten pose sequences
        pose_sequences_flat = pose_sequences.view(batch_size * max_num_frames, pose_dim)  # (batch_size * max_num_frames, pose_dim)
        masks_flat = masks.view(batch_size * max_num_frames)  # (batch_size * max_num_frames,)

        # Add noise to poses
        noisy_pose = self.forward_diffusion(pose_sequences_flat, t, noise)  # (batch_size * max_num_frames, pose_dim)

        # Encode timestep numbers
        h = t.view(batch_size, max_num_frames)  # (batch_size, max_num_frames)
        step_embeddings = self.step_encoder(h)  # (batch_size, max_num_frames, hidden_dim)

        # Encode noise
        noise_embeddings = self.noise_encoder(noise)  # (batch_size * max_num_frames, embed_dim)

        # Expand text embeddings to match the number of frames
        text_embedding_expanded = text_embedding.unsqueeze(1).repeat(1, max_num_frames, 1)  # (batch_size, max_num_frames, text_embed_dim)

        # Predict pose
        p_h = self.sign_predictor(
            text_embeddings=text_embedding_expanded.view(batch_size * max_num_frames, -1),
            step_embeddings=step_embeddings.view(batch_size * max_num_frames, -1),
            noise_embeddings=noise_embeddings
        )  # (batch_size * max_num_frames, pose_dim)

        # Compute alpha_h based on the paper's schedule
        delta_h = 1 / torch.log((t + 1).float() + 1)
        alpha_h = (delta_h - (1 / torch.log((t + 2).float() + 1))).unsqueeze(1).clamp(0, 1)  # (batch_size * max_num_frames, 1)

        # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat
        s_hat_h = alpha_h * p_h + (1 - alpha_h) * pose_sequences_flat  # (batch_size * max_num_frames, pose_dim)

        # Compute diffusion loss Ld
        loss_noise = self.loss_fn(p_h, noise)  # MSE for noise prediction
        loss_pose = self.loss_fn(s_hat_h, pose_sequences_flat)  # MSE for pose prediction
        loss_ld = (alpha_h.squeeze(1) * loss_noise + (1 - alpha_h.squeeze(1)) * loss_pose)  # Weighted loss
        loss_ld = (loss_ld * masks_flat).sum() / (masks_flat.sum() + EPSILON)  # Average loss with epsilon for stability

        # Compute Embedding Consistency Learning loss Lecl
        generated_pose = s_hat_h.view(batch_size, max_num_frames, self.hparams.pose_dims[0], self.hparams.pose_dims[1])  # (batch_size, max_num_frames, 75, 2)
        generated_sign_embeddings = self.sign_encoder(generated_pose, masks)  # (batch_size, embed_dim)
        ecl = self.ecl_fn(generated_sign_embeddings, sign_embeddings)  # scalar

        # Compute InfoNCE loss Lnce
        info_nce = self.info_nce_fn(text_embedding, sign_embeddings)  # scalar

        # Total loss
        loss = loss_ld + info_nce + ecl + loss_length  # λ1=λ2=λ3=λ4=1

        # Compute predicted length statistics
        length_diff = torch.abs(predicted_length - true_length)
        avg_length_diff = length_diff.mean()

        # Log losses and length statistics
        self.log("test_ld", loss_ld, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_info_nce", info_nce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_ecl", ecl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_length_loss", loss_length, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_length_diff", avg_length_diff, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_predicted_length_mean", predicted_length.mean(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_predicted_length_std", predicted_length.std(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        pose_predicted = s_hat_h.view(batch_size, max_num_frames, self.hparams.pose_dims[0], self.hparams.pose_dims[1])  # (batch_size, max_num_frames, 75, 2)
        pose_predicted = pose_predicted.cpu().numpy()  # 转换为 NumPy 数组

        for i in range(batch_size):
            filename = filenames[i]
            prediction = pose_predicted[i]
            self.test_filenames.append(filename)
            self.test_predictions.append(prediction)

        return loss

    def on_test_epoch_end(self):

        os.makedirs(self.test_results_dir, exist_ok=True)

        for filename, prediction in zip(self.test_filenames, self.test_predictions):
            save_path = os.path.join(self.test_results_dir, f"{filename}_prediction.npy")
            np.save(save_path, prediction)
            print(f"Saved prediction to {save_path}")

        self.test_filenames = []
        self.test_predictions = []
        

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        # Separate parameters for the text encoder and the generator (including Sign Encoder, Step Encoder, Noise Encoder, Sign Predictor, Length Predictor)
        text_params = [param for param in self.text_encoder.parameters() if param.requires_grad]
        generator_params = list(self.sign_encoder.parameters()) + \
                           list(self.step_encoder.parameters()) + \
                           list(self.noise_encoder.parameters()) + \
                           list(self.sign_predictor.parameters()) + \
                           list(self.length_predictor.parameters())

        optimizer = optim.AdamW([
            {'params': text_params, 'lr': self.hparams.lr_text_encoder},
            {'params': generator_params, 'lr': self.hparams.lr_generator}
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
        Generates poses using the DDIM sampler based on predicted length.
        """
        self.eval()
        with torch.no_grad():
            # Encode text
            text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_masks
            )
            text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token, (batch_size, text_embedding_dim)

            # Predict sequence length based on text embedding
            # Dummy pose input and full mask to get sign embeddings
            dummy_pose = torch.zeros(text_embedding.size(0), self.hparams.num_steps, self.hparams.pose_dims[0], self.hparams.pose_dims[1], device=self.device)
            full_mask = torch.ones(text_embedding.size(0), self.hparams.num_steps, dtype=torch.bool, device=self.device)
            sign_embeddings = self.sign_encoder(dummy_pose, full_mask)  # (batch_size, embed_dim)
            predicted_length = self.length_predictor(sign_embeddings)  # (batch_size,)

            # Clamp predicted_length to [1, self.num_steps]
            num_steps_pred = torch.clamp(predicted_length, min=1, max=self.num_steps).long()

            # Determine the maximum number of steps in the batch
            max_pred_steps = num_steps_pred.max().item()

            # Encode step numbers up to max_pred_steps
            step_numbers = torch.arange(0, max_pred_steps, device=self.device).unsqueeze(0).repeat(text_embedding.size(0), 1)  # (batch_size, max_pred_steps)
            step_embeddings = self.step_encoder(step_numbers)  # (batch_size, max_pred_steps, hidden_dim)

            # Encode noise (starting from pure noise)
            noise = torch.randn(text_embedding.size(0), max_pred_steps, self.hparams.pose_dims[0] * self.hparams.pose_dims[1], device=self.device)  # Starting with pure noise
            noise_embeddings = self.noise_encoder(noise.view(-1, self.hparams.pose_dims[0] * self.hparams.pose_dims[1]))  # (batch_size * max_pred_steps, embed_dim)

            # Sample poses
            pose = self.ddim_sampler.sample(text_embedding, step_embeddings, noise_embeddings, device=self.device, num_steps=max_pred_steps)

        return pose, num_steps_pred

# Prepare Data Loaders

# Define data transformations (Normalization removed)
transform = None  # No normalization

# Set data folder paths (modify according to actual paths)
train_french_folder_2d = "/data/zmo/Swisspose/swissubase_2569_1_0/data/train_data_french_2d"
test_french_folder_2d = "/data/zmo/Swisspose/swissubase_2569_1_0/data/test_data_french_2d"
train_german_folder_2d = "/data/zmo/Swisspose/swissubase_2569_1_0/data/train_data_german_2d"
test_german_folder_2d = "/data/zmo/Swisspose/swissubase_2569_1_0/data/test_data_german_2d"

# Initialize French and German training and testing datasets
train_dataset_french_2d = SignLanguageDataset(data_folder=train_french_folder_2d, transform=transform)
train_dataset_german_2d = SignLanguageDataset(data_folder=train_german_folder_2d, transform=transform)
test_dataset_french_2d = SignLanguageDataset(data_folder=test_french_folder_2d, transform=transform)
test_dataset_german_2d = SignLanguageDataset(data_folder=test_german_folder_2d, transform=transform)

# Concatenate French and German datasets
train_dataset = ConcatDataset([train_dataset_french_2d, train_dataset_german_2d])
test_dataset = ConcatDataset([test_dataset_french_2d, test_dataset_german_2d])

# Create Data Loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4,  
    pin_memory=True,
    drop_last=True,  
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    test_dataset, 
    batch_size=16,  
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    drop_last=False,
    collate_fn=custom_collate_fn
)

# Initialize Model and Training Parameters

# Define training parameters
max_epochs = 200
total_steps = len(train_loader) * max_epochs

# Initialize the model and pass total_steps
model = IterativeTextGuidedPoseGenerationModel(
    text_model_name="xlm-roberta-base",  # Using XLM-RoBERTa multilingual model
    pose_dims=(75, 2),  # Dimensions of each pose frame
    hidden_dim=768,       
    num_layers=6,  
    num_heads=12,         
    num_steps=1000,
    lr_text_encoder=START_LEARNING_RATE_TEXT_ENCODER,  # Separate learning rates
    lr_generator=START_LEARNING_RATE_GENERATOR,
    beta_start=1e-4,
    beta_end=0.02,
    eta=0.0, 
    total_steps=total_steps  # Pass total_steps
)

# Define Callbacks and Logger

# Define callbacks to save the best model and the last checkpoint
checkpoint_callback_best = ModelCheckpoint(
    monitor='val_loss',
    dirpath='/data/zmo/Swisspose/best_model',  
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)

checkpoint_callback_last = ModelCheckpoint(
    save_last=True,
    dirpath='/data/zmo/Swisspose/checkpoints', 
    filename='last'
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min',
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize Logger
logger = TensorBoardLogger("tb_logs", name="pose_generation_model")

# Initialize Trainer

from pytorch_lightning.strategies import DDPStrategy

trainer = pl.Trainer(
    max_epochs=max_epochs,
    devices=1,  # Adjust based on actual number of GPUs
    accelerator='gpu',  
    strategy=DDPStrategy(find_unused_parameters=True),  
    precision=16 if torch.cuda.is_available() else 32,
    enable_progress_bar=True,  
    callbacks=[checkpoint_callback_best, checkpoint_callback_last, early_stopping_callback, lr_monitor],
    gradient_clip_val=1.0, 
    accumulate_grad_batches=2,  
    log_every_n_steps=10,  # Adjust logging frequency
    logger=logger
)

# Start Training

# Define the path to the latest checkpoint
latest_checkpoint = None
last_checkpoint_path = os.path.join('/data/zmo/Swisspose/checkpoints', 'last.ckpt')
if os.path.exists(last_checkpoint_path):
    latest_checkpoint = last_checkpoint_path
    print(f"Resuming from checkpoint: {latest_checkpoint}")

# Start training, resuming from the latest checkpoint if it exists
trainer.fit(model, train_loader, val_loader, ckpt_path=latest_checkpoint)

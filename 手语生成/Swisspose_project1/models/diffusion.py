# models/diffusion.py

import torch
import torch.nn as nn

class DDIMSampler:
    def __init__(self, model, betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, num_steps=20, eta=0.0):
        """
        Initializes the DDIM sampler with dynamic step lengths.

        Args:
            model (nn.Module): The generative model.
            betas (torch.Tensor): Beta schedule.
            alphas (torch.Tensor): Alpha schedule.
            alpha_cumprod (torch.Tensor): Cumulative product of alphas.
            sqrt_alpha_cumprod (torch.Tensor): Square root of cumulative alphas.
            sqrt_one_minus_alpha_cumprod (torch.Tensor): Square root of (1 - cumulative alphas).
            num_steps (int): Number of denoising steps.
            eta (float): Noise scaling factor.
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
        Performs DDIM sampling with dynamic step lengths to generate a sequence of poses.

        Args:
            text_embeddings (torch.Tensor): (batch_size, text_embed_dim)
            step_embeddings (torch.Tensor): (batch_size, max_pred_steps, embed_dim)
            noise_embeddings (torch.Tensor): (batch_size, max_pred_steps, embed_dim)
            device (torch.device): Device to perform computation on.
            num_steps (int, optional): Number of steps to generate. Defaults to self.num_steps.

        Returns:
            torch.Tensor: Generated pose sequence with shape (batch_size, num_steps, 75, 2)
            torch.Tensor: s^0 tensor with shape (batch_size, 75, 2)
        """
        if num_steps is None:
            num_steps = self.num_steps

        batch_size = text_embeddings.size(0)
        pose_dim = self.model.hparams.pose_dims[0] * self.model.hparams.pose_dims[1]

        # Initialize the pose sequence with pure noise
        pose = torch.randn(batch_size, pose_dim, device=device)  # (batch_size, pose_dim)

        # Initialize s_hat_prev to zero
        s_hat_prev = torch.zeros_like(pose).to(device)

        # List to store pose at each step
        pose_history = []

        s_0 = None  # To store s^0

        for h in range(num_steps):
            # Current step index
            t_step = h

            # Get current step embeddings
            current_step_embeddings = step_embeddings[:, h, :] 

            # Get current noise embeddings
            current_noise_embeddings = noise_embeddings[:, h, :]  

            # Predict pose using the model
            p_h = self.model.sign_predictor(
                fused_embeddings=torch.cat([
                    text_embeddings,          
                    s_hat_prev,              
                    current_step_embeddings,
                    current_noise_embeddings # (batch_size, embed_dim)
                ], dim=1)  # Concatenated input
            )  

            # Compute delta_h and alpha_h for dynamic step lengths
            delta_h = 1 / torch.log(torch.tensor(h + 1, dtype=torch.float, device=device))
            delta_h_plus_1 = 1 / torch.log(torch.tensor(h + 2, dtype=torch.float, device=device))
            alpha_h = (delta_h - delta_h_plus_1).clamp(0, 1)  # Scalar

            # Expand alpha_h to match batch size
            alpha_h = alpha_h.expand(batch_size, 1)  # (batch_size, 1)

            # Compute s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev
            s_hat_h = alpha_h * p_h + (1 - alpha_h) * s_hat_prev  # (batch_size, pose_dim)

            # Introduce random noise to s_hat_h for robustness
            noise = torch.randn_like(s_hat_h) * 0.05  # Small random noise
            s_hat_h_noisy = s_hat_h + noise  # (batch_size, pose_dim)

            # Update pose using the predicted s_hat_h_noisy
            pose = s_hat_h_noisy  

            # Update s_hat_prev for the next step
            s_hat_prev = s_hat_h_noisy

            # Append current pose to history
            pose_history.append(pose.view(batch_size, 75, 2).clone())  # (batch_size, 75, 2)

            # Store s^0 at the first step
            if h == 0:
                s_0 = pose_history[0]  # (batch_size, 75, 2)

            print(f"Step {h+1}/{num_steps}: alpha_h={alpha_h.mean().item():.4f}")

        # Stack the pose history to form a sequence
        pose_sequence = torch.stack(pose_history, dim=1)  # (batch_size, num_steps, 75, 2)

        print(f"Final pose sequence shape: {pose_sequence.shape}")  # Should be (batch_size, num_steps, 75, 2)

        # Output s^0, which is the first pose in the sequence
        return pose_sequence, s_0

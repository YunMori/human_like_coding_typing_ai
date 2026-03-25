import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from layer3_dynamics.gan.generator import TimingGenerator
from layer3_dynamics.gan.discriminator import TimingDiscriminator
from layer3_dynamics.gan.dataset import KeystrokeDataset


def compute_gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interpolated)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=(1, 2)) - 1) ** 2).mean()
    return gp


class GANTrainer:
    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.noise_dim = config.get("noise_dim", 64)
        self.context_dim = config.get("context_dim", 32)
        self.seq_len = config.get("seq_len", 32)
        self.lambda_gp = 10.0
        self.d_steps = 5  # D:G = 5:1

        self.G = TimingGenerator(
            noise_dim=self.noise_dim,
            context_dim=self.context_dim,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 3),
            seq_len=self.seq_len,
        ).to(self.device)

        self.D = TimingDiscriminator(
            hidden_size=config.get("hidden_size", 128),
        ).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    def train(self, dataset_path: str, epochs: int = 50, batch_size: int = 64):
        dataset = KeystrokeDataset(dataset_path, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            g_losses, d_losses = [], []
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, real_batch in enumerate(pbar):
                real = real_batch.to(self.device)  # (B, seq_len, 3)
                B = real.size(0)

                # Train Discriminator
                for _ in range(self.d_steps):
                    noise = torch.randn(B, self.noise_dim, device=self.device)
                    ctx = torch.randn(B, self.context_dim, device=self.device)
                    fake = self.G(noise, ctx).detach()

                    d_real = self.D(real).mean()
                    d_fake = self.D(fake).mean()
                    gp = compute_gradient_penalty(self.D, real, fake, self.device)
                    d_loss = d_fake - d_real + self.lambda_gp * gp

                    self.opt_D.zero_grad()
                    d_loss.backward()
                    self.opt_D.step()
                    d_losses.append(d_loss.item())

                # Train Generator
                noise = torch.randn(B, self.noise_dim, device=self.device)
                ctx = torch.randn(B, self.context_dim, device=self.device)
                fake = self.G(noise, ctx)
                g_loss = -self.D(fake).mean()

                self.opt_G.zero_grad()
                g_loss.backward()
                self.opt_G.step()
                g_losses.append(g_loss.item())

                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

            logger.info(f"Epoch {epoch+1}: G={sum(g_losses)/len(g_losses):.3f}, D={sum(d_losses)/len(d_losses):.3f}")

    def save(self, g_path: str, d_path: str):
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
        logger.info(f"Saved GAN models to {g_path}, {d_path}")

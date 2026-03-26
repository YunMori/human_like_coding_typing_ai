import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from loguru import logger

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from layer3_dynamics.gan.generator import TimingGenerator
from layer3_dynamics.gan.discriminator import TimingDiscriminator
from layer3_dynamics.gan.dataset import KeystrokeDataset


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_ks(G, noise_dim: int, device: torch.device,
                real_delays: np.ndarray, n_samples: int = 512) -> float:
    """빠른 KS statistic 계산 (Early Stopping용 — 검증 데이터만 사용)."""
    if not SCIPY_AVAILABLE or len(real_delays) == 0:
        return 1.0

    # Weights from trained HMM model startprob: [SLOW, NORMAL, PAUSE, ERROR, PAUSE_LONG, FAST]
    HMM_STATE_WEIGHTS = [0.17, 0.40, 0.07, 0.17, 0.03, 0.16]
    context_dim = 32

    G.eval()
    gen_delays = []
    with torch.no_grad():
        for hmm_state, weight in enumerate(HMM_STATE_WEIGHTS):
            n = max(1, int(n_samples * weight))
            ctx = np.zeros(context_dim, dtype=np.float32)
            ctx[4] = 0.604  # complexity (actual dataset mean)
            ctx[5] = 0.523  # fatigue (actual dataset mean)
            ctx[6 + hmm_state] = 1.0
            ctx_t = torch.FloatTensor(ctx).unsqueeze(0).expand(n, -1).to(device)
            noise = torch.randn(n, noise_dim, device=device)
            timings = G(noise, ctx_t).cpu().numpy() * 1000.0
            gen_delays.extend(timings[:, :, 0].flatten().tolist())
    G.train()

    gen_arr = np.array(gen_delays)
    rng = np.random.default_rng(42)
    real_sub = rng.choice(real_delays, size=min(len(gen_arr), len(real_delays)), replace=False)
    ks_stat, _ = scipy_stats.ks_2samp(real_sub, gen_arr)
    return float(ks_stat)


class GANTrainer:
    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        self.device = torch.device(_best_device() if device == "auto" else device)
        logger.info(f"GAN trainer using device: {self.device}")
        self.noise_dim = config.get("noise_dim", 16)
        self.context_dim = config.get("context_dim", 32)
        self.seq_len = config.get("seq_len", 32)
        self.d_steps = 2
        self.g_steps = 1

        self.G = TimingGenerator(
            noise_dim=self.noise_dim,
            context_dim=self.context_dim,
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 3),
            seq_len=self.seq_len,
        ).to(self.device)

        self.D = TimingDiscriminator(
            context_dim=self.context_dim,
            hidden_size=config.get("hidden_size", 256),
        ).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    def train(self, dataset_path: str, epochs: int = 1000, batch_size: int = 64,
              eval_every: int = 50, target_ks: float = 0.10, patience: int = 5,
              best_g_path: str = "models/gan_generator_best.pth",
              best_d_path: str = "models/gan_discriminator_best.pth"):

        # ── 90/10 Train/Val 분리 (완전히 겹치지 않음) ──
        full_dataset = KeystrokeDataset(dataset_path, seq_len=self.seq_len)
        val_size = max(1000, int(len(full_dataset) * 0.10))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        logger.info(f"Train: {train_size:,} / Val: {val_size:,} sequences (90/10 split, seed=42)")

        # 검증 데이터에서 실제 delay 추출 (학습 데이터와 완전 분리)
        real_delays = np.array([
            full_dataset.sequences[idx]["timings"][t, 0] * 1000.0  # seconds → ms
            for idx in val_dataset.indices
            for t in range(full_dataset.sequences[idx]["timings"].shape[0])
        ])
        logger.info(f"Val delays for KS eval: {len(real_delays):,} samples")

        best_ks = float("inf")
        no_improve_count = 0

        for epoch in range(epochs):
            g_losses, d_losses = [], []
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(pbar):
                real = batch["timings"].to(self.device)    # (B, seq_len, 3)
                ctx_seq = batch["context"].to(self.device)  # (B, seq_len, 32)
                B = real.size(0)

                ctx = ctx_seq[:, 0, :]  # (B, 32)

                # Train Discriminator (conditional, hinge loss)
                for _ in range(self.d_steps):
                    noise = torch.randn(B, self.noise_dim, device=self.device)
                    fake = self.G(noise, ctx).detach()

                    d_real = self.D(real, ctx)
                    d_fake = self.D(fake, ctx)
                    d_loss = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()

                    self.opt_D.zero_grad()
                    d_loss.backward()
                    self.opt_D.step()
                    d_losses.append(d_loss.item())

                # Train Generator (conditional)
                noise = torch.randn(B, self.noise_dim, device=self.device)
                fake = self.G(noise, ctx)
                g_loss = -self.D(fake, ctx).mean()

                self.opt_G.zero_grad()
                g_loss.backward()
                self.opt_G.step()
                g_losses.append(g_loss.item())

                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

            logger.info(f"Epoch {epoch+1}: G={sum(g_losses)/len(g_losses):.3f}, D={sum(d_losses)/len(d_losses):.3f}")

            # ── Early Stopping: KS 평가 ──
            if SCIPY_AVAILABLE and (epoch + 1) % eval_every == 0 and len(real_delays) > 0:
                ks = _compute_ks(self.G, self.noise_dim, self.device, real_delays)

                if ks < best_ks:
                    best_ks = ks
                    no_improve_count = 0
                    self.save(best_g_path, best_d_path)
                    logger.info(f"[Eval epoch {epoch+1}] KS: {ks:.4f} → best saved")
                else:
                    no_improve_count += 1
                    logger.info(f"[Eval epoch {epoch+1}] KS: {ks:.4f} → no improvement (patience {no_improve_count}/{patience})")

                if ks < target_ks:
                    logger.success(f"Early stop: target KS {target_ks} reached (current: {ks:.4f})")
                    return

                if no_improve_count >= patience:
                    logger.warning(f"Early stop: no improvement for {patience} consecutive evals (best KS: {best_ks:.4f})")
                    return

    def save(self, g_path: str, d_path: str):
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
        logger.info(f"Saved GAN models to {g_path}, {d_path}")

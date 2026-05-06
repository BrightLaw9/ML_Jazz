"""
Modular PyTorch implementation of a Text-Conditioned Latent Diffusion Model for music.

Pipeline:
1. Raw WAV -> EnCodec integer codes
2. VAE compresses normalized code IDs -> latent z
3. Text embedding model -> text embedding
4. Text-conditioned prior p(z|text)
5. Diffusion U-Net predicts noise eps at random timesteps
6. Latent -> VAE decoder -> categorical code logits
7. Argmax logits -> integer codes
8. EnCodec decoder -> waveform

Each module can be toggled on/off for debugging.
This is intentionally modular and verifiable rather than highly optimized.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ENCODEC_IMPORT_ERROR = None

try:
    from .encodec_embedding import EncodecEmbedder, save_audio
except Exception as exc:
    try:
        from encodec_embedding import EncodecEmbedder, save_audio
    except Exception as fallback_exc:
        ENCODEC_IMPORT_ERROR = fallback_exc
        EncodecEmbedder = None
        save_audio = None


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # EnCodec code positions: num_quantizers * target_frames.
    embed_dim: int = 8192
    codebook_bins: int = 1024
    num_quantizers: int = 8

    # Per-code VAE latent dimension. Latents have shape [B, T, Q, latent_dim].
    latent_dim: int = 128
    vae_hidden_dim: int = 128

    # Text embedding dimension
    text_dim: int = 512

    # Diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    diffusion_inference_steps: int = 1000

    # Toggle modules
    use_text_conditioned_prior: bool = False
    use_diffusion: bool = True
    freeze_vae: bool = False

    # Training defaults
    test_mode: bool = False
    generate_mode: bool = False
    batch_size: int = 4 if not test_mode else 1
    learning_rate: float = 1e-4
    epochs: int = 300
    train_labels_path: str = "test_labels.json" if test_mode else "train_labels.json"
    checkpoint_dir: str = "diffusion/checkpoints"


cfg = Config()


# ============================================================
# TRAINING DATA
# ============================================================

def _load_json_with_trailing_commas(path):
    raw = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", raw)
        return json.loads(cleaned)


def load_labeled_wav_entries(
    labels_path,
    require_label=True,
    require_file_exists=True
):
    """
    Load train_labels.json entries into (wav_path, label) pairs.

    Expected JSON shape:
        [
            {
                "directory": ".",
                "labels": [{"file": "song.wav", "label": "bebop"}]
            }
        ]
    """
    labels_path = Path(labels_path).resolve()
    data = _load_json_with_trailing_commas(labels_path)
    entries = []
    skipped_empty_label = 0
    skipped_missing_file = 0

    for group in data:
        directory = Path(group.get("directory", "."))
        if not directory.is_absolute():
            directory = labels_path.parent / directory

        for item in group.get("labels", []):
            label = (item.get("label") or "").strip()
            if require_label and not label:
                skipped_empty_label += 1
                continue

            wav_path = Path(item["file"])
            if not wav_path.is_absolute():
                wav_path = directory / wav_path
            wav_path = wav_path.resolve()

            if require_file_exists and not wav_path.exists():
                skipped_missing_file += 1
                continue

            entries.append((str(wav_path), label))

    if skipped_empty_label:
        print(f"Skipped {skipped_empty_label} rows with empty labels.")
    if skipped_missing_file:
        print(f"Skipped {skipped_missing_file} rows with missing WAV files.")

    if not entries:
        raise ValueError(
            f"No trainable WAV/label pairs found in {labels_path}. "
            "Check that labels are non-empty and WAV paths exist."
        )

    return entries


class LabeledWavDataset(Dataset):
    def __init__(self, labels_path, require_label=True, require_file_exists=True):
        self.entries = load_labeled_wav_entries(
            labels_path=labels_path,
            require_label=require_label,
            require_file_exists=require_file_exists
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        wav_path, label = self.entries[index]
        return wav_path, label


def collate_wav_label_batch(batch):
    wav_paths, labels = zip(*batch)
    return list(wav_paths), list(labels)


# ============================================================
# UTILITIES
# ============================================================


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================
# TEXT EMBEDDING MODEL
# ============================================================

class TextEmbeddingModel(nn.Module):
    """
    Pretrained CLAP text encoder.

    Input:
        list[str]

    Output:
        tensor of shape [B, text_dim]
    """

    def __init__(
        self,
        text_dim=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.device = device
        self.text_dim = text_dim

        # Lazy import so the rest of the codebase can still run
        # if CLAP is temporarily disabled.
        import laion_clap

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        self.clap_model.to(device)
        self.clap_model.eval()

        # Optional projection layer if CLAP embedding dimension
        # differs from desired downstream dimension.
        self.proj = None

    @torch.no_grad()
    def forward(self, text_list):
        """
        Args:
            text_list: list[str]

        Returns:
            text_embedding: [B, text_dim]
        """

        text_embedding = self.clap_model.get_text_embedding(text_list)

        if not isinstance(text_embedding, torch.Tensor):
            text_embedding = torch.tensor(text_embedding)

        text_embedding = text_embedding.to(self.device).float()

        if self.proj is None and text_embedding.shape[-1] != self.text_dim:
            self.proj = nn.Sequential(
                nn.Linear(text_embedding.shape[-1], 1024),
                nn.SiLU(),
                nn.Linear(1024, self.text_dim)
            ).to(self.device)

        if self.proj is not None:
            text_embedding = self.proj(text_embedding)

        return text_embedding


# ============================================================
# TEXT-CONDITIONED PRIOR
# ============================================================

class TextConditionedPrior(nn.Module):
    def __init__(self, text_dim, latent_dim, num_quantizers):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers

        self.net = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        self.temporal = nn.Conv1d(512, 512, 3, padding=1)
        self.to_mu = nn.Linear(512, num_quantizers * latent_dim)
        self.to_logvar = nn.Linear(512, num_quantizers * latent_dim)

    def forward(self, text_embedding, T, Q=None):
        if Q is None:
            Q = self.num_quantizers
        if Q != self.num_quantizers:
            raise ValueError(
                f"TextConditionedPrior was initialized for {self.num_quantizers} "
                f"quantizers, but got Q={Q}."
            )

        h = self.net(text_embedding)          # [B, 512]
        h = h.unsqueeze(-1).repeat(1, 1, T)   # [B, 512, T]
        h = self.temporal(h).transpose(1, 2)  # [B, T, 512]

        mu = self.to_mu(h).view(
            text_embedding.shape[0],
            T,
            Q,
            self.latent_dim
        )
        logvar = self.to_logvar(h).view(
            text_embedding.shape[0],
            T,
            Q,
            self.latent_dim
        )
        return mu, logvar


# ============================================================
# VAE
# ============================================================

class Encoder(nn.Module):
    def __init__(
        self,
        codebook_bins,
        num_quantizers,
        embed_dim=128,
        hidden_dim=128,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_bins, embed_dim)
            for _ in range(num_quantizers)
        ])

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, hidden_dim, 5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, embed_dim, 5, padding=2),
                nn.SiLU(),
            )
            for _ in range(num_quantizers)
        ])

        self.to_mu = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, 1)
            for _ in range(num_quantizers)
        ])
        self.to_logvar = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, 1)
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """
        x: [B, T, Q]

        returns:
            mu: [B, T, Q, D]
            logvar: [B, T, Q, D]
        """
        mu_all = []
        logvar_all = []

        for q in range(self.num_quantizers):
            h = self.embeddings[q](x[:, :, q])  # [B, T, D]
            h = h.transpose(1, 2)               # [B, D, T]
            h = self.convs[q](h)

            mu = self.to_mu[q](h).transpose(1, 2)
            logvar = self.to_logvar[q](h).transpose(1, 2)

            mu_all.append(mu)
            logvar_all.append(logvar)

        return torch.stack(mu_all, dim=2), torch.stack(logvar_all, dim=2)


class Decoder(nn.Module):
    def __init__(
        self,
        codebook_bins,
        num_quantizers,
        embed_dim=128,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_bins = codebook_bins

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_bins, embed_dim))
            for _ in range(num_quantizers)
        ])

    def forward(self, z):
        """
        z: [B, T, Q, D]

        returns:
            logits: [B, T, Q, K]
        """
        logits_all = []

        for q in range(self.num_quantizers):
            z_q = z[:, :, q]                  # [B, T, D]
            codebook = self.codebooks[q]      # [K, D]
            logits_all.append(torch.matmul(z_q, codebook.t()))

        return torch.stack(logits_all, dim=2)

    def decode(self, z):
        return self.forward(z).argmax(dim=-1)


class VAE(nn.Module):
    def __init__(
        self,
        codebook_bins,
        num_quantizers,
        latent_dim,
        hidden_dim=128
    ):
        super().__init__()
        self.encoder = Encoder(
            codebook_bins=codebook_bins,
            num_quantizers=num_quantizers,
            embed_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.decoder = Decoder(
            codebook_bins=codebook_bins,
            num_quantizers=num_quantizers,
            embed_dim=latent_dim
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

class LatentUNet(nn.Module):
    def __init__(self, latent_dim, text_dim):
        super().__init__()
        time_dim = 128
        cond_dim = latent_dim + time_dim + text_dim

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Encoder (downward path)
        self.down1 = nn.Sequential(nn.Linear(cond_dim, 1024), nn.SiLU())
        self.down2 = nn.Sequential(nn.Linear(1024, 512), nn.SiLU())
        self.down3 = nn.Sequential(nn.Linear(512, 256), nn.SiLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(256, 256), nn.SiLU())

        # Decoder (upward path) — note the doubled input dims for skip connections
        self.up3 = nn.Sequential(nn.Linear(256 + 256, 512), nn.SiLU())
        self.up2 = nn.Sequential(nn.Linear(512 + 512, 1024), nn.SiLU())
        self.up1 = nn.Sequential(nn.Linear(1024 + 1024, 1024), nn.SiLU())

        self.out = nn.Linear(1024, latent_dim)

    def forward(self, z_t, t, text_embedding):
        t_emb = self.time_embed(t)
        leading_dims = z_t.shape[1:-1]
        for _ in leading_dims:
            t_emb = t_emb.unsqueeze(1)
            text_embedding = text_embedding.unsqueeze(1)

        expand_shape = (z_t.shape[0], *leading_dims, -1)
        t_emb = t_emb.expand(*expand_shape)
        text_embedding = text_embedding.expand(*expand_shape)
        x = torch.cat([z_t, t_emb, text_embedding], dim=-1)

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Bottleneck
        b = self.bottleneck(d3)

        # Decoder with skip connections
        u3 = self.up3(torch.cat([b, d3], dim=-1))
        u2 = self.up2(torch.cat([u3, d2], dim=-1))
        u1 = self.up1(torch.cat([u2, d1], dim=-1))

        return self.out(u1)
    

# ============================================================
# DIFFUSION SCHEDULER
# ============================================================

class DiffusionScheduler:
    def __init__(self, cfg):
        self.timesteps = cfg.timesteps

        betas = torch.linspace(
            cfg.beta_start,
            cfg.beta_end,
            cfg.timesteps
        )

        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_hat = alpha_hat

    def sample_timesteps(self, batch_size, device):
        return torch.randint(
            low=1,
            high=self.timesteps,
            size=(batch_size,),
            device=device
        )

    def add_noise(self, x, t):
        alpha_hat = self.alpha_hat.to(x.device)[t].view(
            x.shape[0],
            *([1] * (x.ndim - 1))
        )
        noise = torch.randn_like(x)

        x_t = (
            torch.sqrt(alpha_hat) * x +
            torch.sqrt(1 - alpha_hat) * noise
        )
        return x_t, noise

    def step(self, pred_noise, t, z_t):
        """
        pred_noise is eps; noise to be removed in the reverse process;
        z_t is latent vector at current step, z_{t-1} will be computed

        Derived based on Langevin Dynamics; 
        theta is the true desired distribution that no noised should represent
        we compute some sample of z_{t-1} from p_{theta}(x_{t-1} | x_t) follows N(mu(x_t), (sigma_t)^2I)
        (sigma_t)^2 = beta_t

        mu_t is derived to be below; sigma_t gets multiplied by a sample from N(0, 1) and added to mu_t to approximate
        the distribution of p_theta(x_{t-1})
        """

        alpha_t = self.alphas.to(z_t.device)[t]
        alpha_bar_t = self.alpha_hat.to(z_t.device)[t]

        beta_t = self.betas.to(z_t.device)[t]

        mu = (z_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)

        # Note that no noise is added at final step

        sigma_t = torch.sqrt(beta_t)
        if t > 0:
            noise = torch.randn_like(z_t)
            z_prev = mu + sigma_t * noise
        else:
            z_prev = mu

        return z_prev


# ============================================================
# FULL MODEL
# ============================================================

class MusicLatentDiffusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.text_model = TextEmbeddingModel(
            text_dim=cfg.text_dim,
            device=cfg.device
        )
        self.vae = VAE(
            codebook_bins=cfg.codebook_bins,
            num_quantizers=cfg.num_quantizers,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.vae_hidden_dim
        )
        # self.prior = TextConditionedPrior(
        #     cfg.text_dim,
        #     cfg.latent_dim,
        #     cfg.num_quantizers
        # )
        self.unet = LatentUNet(cfg.latent_dim, cfg.text_dim)
        self.scheduler = DiffusionScheduler(cfg)
        self.encodec_embedder = None
        self.count = 0

        if cfg.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

    def _get_encodec_embedder(self):
        if self.encodec_embedder is None:
            if EncodecEmbedder is None:
                raise ImportError(
                    "EncodecEmbedder could not be imported. Install the "
                    "encodec dependencies or pass a precomputed audio "
                    "code tensor."
                ) from ENCODEC_IMPORT_ERROR

            self.encodec_embedder = EncodecEmbedder(
                target_dim=self.cfg.embed_dim,
                device=self.cfg.device
            )

        return self.encodec_embedder

    def audio_codes_from_input(
        self,
        audio_input: Union[torch.Tensor, str, Path, Sequence[Union[str, Path]]]
    ):
        """
        Accept either precomputed EnCodec integer codes or WAV paths.

        Tensor input is expected to be [B, K, T]. Path input is encoded
        by diffusion/encodec_embedding.py's EncodecEmbedder.
        """
        if isinstance(audio_input, torch.Tensor):
            audio_codes = audio_input.to(self.cfg.device).long()
            if audio_codes.dim() == 2:
                num_quantizers = self.cfg.num_quantizers
                frame_length = self.cfg.embed_dim // num_quantizers
                if audio_codes.shape == (num_quantizers, frame_length):
                    audio_codes = audio_codes.unsqueeze(0)
                else:
                    audio_codes = audio_codes.reshape(
                        audio_codes.shape[0],
                        num_quantizers,
                        frame_length
                    )
            self._validate_audio_code_shape(audio_codes)
            return audio_codes

        if isinstance(audio_input, (str, Path)):
            audio_paths = [audio_input]
        elif isinstance(audio_input, Sequence):
            audio_paths = list(audio_input)
        else:
            raise TypeError(
                "audio_input must be an integer code tensor, a WAV path, "
                "or a sequence of WAV paths."
            )

        embedder = self._get_encodec_embedder()
        codes = [
            embedder.encode(str(audio_path))
            for audio_path in audio_paths
        ]
        audio_codes = torch.cat(codes, dim=0).to(self.cfg.device).long()
        self._validate_audio_code_shape(audio_codes)
        return audio_codes

    def _validate_audio_code_shape(self, audio_codes):
        if audio_codes.dim() != 3:
            raise ValueError(
                "audio code tensors must have shape [B, Q, T]; "
                f"got {tuple(audio_codes.shape)}."
            )
        if audio_codes.shape[1] != self.cfg.num_quantizers:
            raise ValueError(
                f"Expected {self.cfg.num_quantizers} quantizers in [B, Q, T], "
                f"got {audio_codes.shape[1]}."
            )

    def audio_codes_to_features(self, audio_codes):
        """
        Convert integer codes [B, Q, T] to encoder grid [B, T, Q].
        """
        self._validate_audio_code_shape(audio_codes)
        return audio_codes.transpose(1, 2).contiguous()

    def logits_to_audio_codes(self, logits):
        """
        Convert decoder logits [B, T, Q, bins] to integer codes [B, Q, T].
        """
        code_ids = logits.argmax(dim=-1)
        return code_ids.transpose(1, 2).contiguous()
    
    def save_audio_codes(
        self,
        codes,
        save_path
    ):
        codec = self._get_encodec_embedder()
        audio_waveform = codec.decode_codes_to_waveform(codes)
        save_audio(audio_waveform.detach(), save_path)

    def save_audio_embedding(self, embedding, save_path):
        """
        Backward-compatible alias for older call sites that pass hard codes.
        """
        self.save_audio_codes(embedding, save_path)

    def kl_text_conditioned(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        KL(q(z|x,text) || p(z|text))
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        kl = 0.5 * (
            logvar_p - logvar_q +
            (var_q + (mu_q - mu_p) ** 2) / var_p - 1
        )
        return kl.mean()

    def forward(self, audio_input, text_input):
        logs: Dict[str, torch.Tensor] = {}

        # text_input = "Harry Potter theme song, orchestral, movie music, magical"
        # text_input = "Ballad"
        # print(text_input)
        # text_input[0] = "Harry Potter"

        audio_codes = self.audio_codes_from_input(audio_input)
        audio_features = self.audio_codes_to_features(audio_codes)
        target_code_ids = audio_features
        
        if self.count % 10 == 0 and not self.cfg.test_mode:
            for i, codes in enumerate(audio_codes):
                if i >= 4:
                    break
                self.save_audio_codes(codes, f"original_{i}.wav")
        
        logs["audio_codes_shape"] = torch.tensor(audio_codes.shape)
        logs["audio_features_shape"] = torch.tensor(audio_features.shape)

        text_embedding = self.text_model(text_input)
        logs["text_embedding_shape"] = torch.tensor(text_embedding.shape)

        mu, logvar = self.vae.encoder(audio_features)
        z = reparameterize(mu, logvar)
        logs["latent_shape"] = torch.tensor(z.shape)
        
        # TODO: Turn this use_text_conditioned_prior to FALSE to test only diffusion
        # if self.cfg.use_text_conditioned_prior:
        #     mu_p, logvar_p = self.prior(
        #         text_embedding,
        #         T=mu.shape[1],
        #         Q=mu.shape[2]
        #     )
        #     kl_loss = self.kl_text_conditioned(mu, logvar, mu_p, logvar_p)
        # else:
        #     kl_loss = torch.tensor(0.0, device=z.device)

        # logs["kl_loss"] = kl_loss

        diffusion_loss = torch.tensor(0.0, device=z.device)

        if self.cfg.use_diffusion:
            t = self.scheduler.sample_timesteps(z.shape[0], z.device)
            z_t, noise = self.scheduler.add_noise(z, t)
            pred_noise = self.unet(z_t, t.float(), text_embedding)
            diffusion_loss = F.mse_loss(pred_noise, noise)

        logs["diffusion_loss"] = diffusion_loss

        recon_logits = self.vae.decoder(z)
        logs["recon_logits_shape"] = torch.tensor(recon_logits.shape)

        code_ce_loss = F.cross_entropy(
            recon_logits.reshape(-1, self.cfg.codebook_bins),
            target_code_ids.reshape(-1)
        )
        logs["code_ce_loss"] = code_ce_loss

        if self.count % 10 == 0 or self.cfg.test_mode:
            recon_codes = self.logits_to_audio_codes(recon_logits)
            for i, codes in enumerate(recon_codes):
                if i >= 4:
                    break
                fname = f"test_{i}.wav" if not self.cfg.test_mode else f"test_{self.count}.wav"
                self.save_audio_codes(codes, fname)

        total_loss = (
            # kl_loss + 
            diffusion_loss
            + code_ce_loss
        )
        logs["total_loss"] = total_loss

        self.count += 1
        return total_loss, logs
    
    @torch.no_grad()
    def generate_audio(self, text_input, num_steps=100):
        if num_steps > self.cfg.timesteps:
            raise ValueError(
                "num_steps greater that number of timesteps diffusion model configured for"
                f"got {num_steps}; max is {self.cfg.timesteps}"
            )
        
        self.eval()

        text_embedding = self.text_model(text_input)

        # At the greatest step, input which diffusion model was trained on was mostly noise; 
        # unet is trained to predict the noise; predicting something that holds across all (instead of each t-1 latent vector)
        # step method determinstically calculates z_{t-1}

        embedder = self._get_encodec_embedder()

        B = text_embedding.shape[0]
        T = embedder._target_frame_length()
        Q = embedder._num_quantizers()
        D = self.cfg.latent_dim

        # Nested in [B, T, Q] are D (latent dim) normally distributed numbers to denoise
        z = torch.randn(B, T, Q, D)

        for t in reversed(range(num_steps)):
            if t % 100 == 0:
                print("Completed denoising at time step t=", t)
            
            t_tensor = torch.full((B,), t, device=z.device, dtype=torch.float32)

            # Predict noise
            pred_noise = self.unet(z, t_tensor, text_embedding)

            # Step backward
            z = self.scheduler.step(pred_noise, t, z)

        # Recovers a logit based likelihood for each codebook
        recon_logits = self.vae.decoder(z)

        # Each logit is argmax and corresponding codebook is recorded in place
        recon_codes = self.logits_to_audio_codes(recon_logits)

        for i, codes in enumerate(recon_codes):
            self.save_audio_codes(codes, f"gen_{i}.wav")
        

# ============================================================
# TRAINING LOOP
# ============================================================

def save_training_checkpoint(
    model,
    optimizer,
    epoch,
    batch_index,
    global_step,
    loss,
    checkpoint_dir
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "loss": float(loss),
        "config": vars(model.cfg),
    }

    batch_path = checkpoint_dir / f"batch_{batch_index:06d}_epoch_{epoch:03d}.pt"
    latest_path = checkpoint_dir / "latest.pt"

    torch.save(checkpoint, batch_path)
    torch.save(checkpoint, latest_path)
    return batch_path


def load_training_checkpoint(
    model,
    checkpoint_path,
    device,
    optimizer=None
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model_state_dict"]

    model_state = {
        k: v for k, v in model_state.items() if (not k.startswith("prior.") and not k.startswith('encodec_loss.'))
    }

    model.load_state_dict(model_state)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "batch_index": int(checkpoint.get("batch_index", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "loss": checkpoint.get("loss"),
        "path": checkpoint_path,
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
    }


def train_model_from_labels(
    labels_path=cfg.train_labels_path,
    checkpoint_dir=cfg.checkpoint_dir,
    batch_size=cfg.batch_size,
    epochs=cfg.epochs,
    learning_rate=cfg.learning_rate,
    shuffle=True,
    require_label=True,
    require_file_exists=True,
    resume_from_latest=True,
    resume_checkpoint_path=None
):
    dataset = LabeledWavDataset(
        labels_path=labels_path,
        require_label=require_label,
        require_file_exists=require_file_exists
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wav_label_batch
    )

    model = MusicLatentDiffusionModel(cfg).to(cfg.device)
    global_step = 0
    start_epoch = 0
    resume_batch_index = -1
    resume_state = None

    print("resuming", resume_from_latest)
    if resume_from_latest:
        resume_checkpoint_path = Path(checkpoint_dir) / "latest.pt"
    
    print(resume_checkpoint_path)

    if resume_checkpoint_path is not None:
        resume_checkpoint_path = Path(resume_checkpoint_path)
        if resume_checkpoint_path.exists():
            resume_state = load_training_checkpoint(
                model=model,
                checkpoint_path=resume_checkpoint_path,
                device=cfg.device
            )
            global_step = resume_state["global_step"]
            start_epoch = resume_state["epoch"]
            resume_batch_index = resume_state["batch_index"]
            print(
                f"Resumed checkpoint {resume_state['path']} "
                f"at epoch={start_epoch + 1}, "
                f"batch={resume_batch_index + 1}, "
                f"global_step={global_step}."
            )
        else:
            print(f"No checkpoint found at {resume_checkpoint_path}; starting fresh.")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate
    )
    if resume_state is not None and resume_state["optimizer_state_dict"] is not None and False:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])

    model.train()

    for epoch in range(start_epoch, epochs):
        for batch_index, (wav_paths, labels) in enumerate(dataloader):
            if epoch == start_epoch and batch_index <= resume_batch_index:
                continue

            optimizer.zero_grad(set_to_none=True)

            loss, logs = model(wav_paths, labels)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 80 == 0:
                checkpoint_path = save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step,
                    loss=loss.detach().cpu(),
                    checkpoint_dir=checkpoint_dir
                )

            print(
                f"epoch={epoch + 1}/{epochs} "
                f"batch={batch_index + 1}/{len(dataloader)} "
                f"loss={loss.item():.6f} "
                # f"checkpoint={checkpoint_path}"
            )

            for key, value in logs.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    print(f"  {key}: {value.detach().item():.6f}")

    return model

def generate(text_input, 
             diffusion_steps=cfg.diffusion_inference_steps, 
             checkpoint_dir=cfg.checkpoint_dir):
    model = MusicLatentDiffusionModel(cfg).to(cfg.device)

    model.eval()

    resume_checkpoint_path = Path(checkpoint_dir) / "latest.pt"
    
    print("Resuming from: ", resume_checkpoint_path)

    if resume_checkpoint_path is not None:
        resume_checkpoint_path = Path(resume_checkpoint_path)
        if resume_checkpoint_path.exists():
            resume_state = load_training_checkpoint(
                model=model,
                checkpoint_path=resume_checkpoint_path,
                device=cfg.device
            )
            global_step = resume_state["global_step"]
            start_epoch = resume_state["epoch"]
            resume_batch_index = resume_state["batch_index"]
            print(
                f"Resumed checkpoint {resume_state['path']} "
                f"at epoch={start_epoch + 1}, "
                f"batch={resume_batch_index + 1}, "
                f"global_step={global_step}."
            )

    model.generate_audio(text_input, diffusion_steps)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the text-conditioned latent diffusion music model."
    )
    parser.add_argument("--labels", default=cfg.train_labels_path)
    parser.add_argument("--checkpoint-dir", default=cfg.checkpoint_dir)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Train on rows whose label is empty."
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Keep JSON rows even if the WAV file does not exist yet."
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffling."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from latest.pt in the checkpoint directory."
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume training from a specific checkpoint path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if cfg.generate_mode:
        #generate("Harry Potter theme song, orchestral, movie music, magical")
        generate(["Bebop, Charlie Parker, jazz", "Big Band, medium swing, reflective", "Jazz melody, jazz piano, All the Things you are"])
        #generate("Big Band, medium swing, reflective")
    else:
        train_model_from_labels(
            labels_path=args.labels,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            shuffle=not args.no_shuffle,
            require_label=not args.include_empty_labels,
            require_file_exists=not args.allow_missing_files,
            resume_from_latest=args.resume,
            resume_checkpoint_path=args.resume_checkpoint
        )

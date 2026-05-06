"""
EnCodec integer code extraction for latent diffusion pipeline.

This module:
1. Loads a raw WAV audio file
2. Resamples to EnCodec sample rate
3. Converts waveform -> EnCodec integer code representation
4. Produces fixed-length code tensors
5. Supports decoding back for verification

Install:
pip install torch torchaudio encodec

Reference:
https://github.com/facebookresearch/encodec
"""

import torch
import torchaudio

from encodec import EncodecModel
from encodec.utils import convert_audio


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_EMBED_DIM = 1024
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1


class EncodecEmbedder:
    def __init__(self, target_dim=1024, device=DEVICE):
        self.device = device
        self.target_dim = target_dim

        # 24kHz mono model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)
        self.model.to(device)
        self.model.eval()

    def _num_quantizers(self):
        return self.model.quantizer.get_num_quantizers_for_bandwidth(
            self.model.frame_rate,
            self.model.bandwidth
        )

    def _target_frame_length(self):
        num_quantizers = self._num_quantizers()
        if self.target_dim % num_quantizers != 0:
            raise ValueError(
                "target_dim must be divisible by the active EnCodec "
                f"quantizer count ({num_quantizers}); got {self.target_dim}."
            )
        return self.target_dim // num_quantizers

    def _fit_codes_to_target_length(self, codes):
        """
        Crop/pad EnCodec code indices [B, K, T] to this model's fixed length.
        """
        target_frames = self._target_frame_length()
        codes = codes.to(self.device).long()

        if codes.shape[-1] > target_frames:
            codes = codes[..., :target_frames]
        elif codes.shape[-1] < target_frames:
            pad_frames = target_frames - codes.shape[-1]
            padding = torch.zeros(
                *codes.shape[:-1],
                pad_frames,
                dtype=codes.dtype,
                device=codes.device
            )
            codes = torch.cat([codes, padding], dim=-1)

        return codes.clamp(0, self.model.quantizer.bins - 1)

    def load_audio(self, wav_path):
        """
        Load and convert audio into EnCodec-compatible format.

        Returns:
            waveform: [1, C, T]
        """
        wav, sr = torchaudio.load(wav_path)

        wav = convert_audio(
            wav,
            sr,
            self.model.sample_rate,
            self.model.channels
        )

        wav = wav.unsqueeze(0).to(self.device)
        return wav

    @torch.no_grad()
    def encode(self, wav_path):
        """
        WAV -> fixed-length integer EnCodec codes.

        Returns:
            codes: [1, num_quantizers, target_frames]
        """
        wav = self.load_audio(wav_path)

        encoded_frames = self.model.encode(wav)
        codes = torch.cat([frame_codes for frame_codes, _ in encoded_frames], dim=-1)
        return self._fit_codes_to_target_length(codes)

    @torch.no_grad()
    def decode(self, codes):
        return self.decode_codes_to_waveform(codes)

    @torch.no_grad()
    def codes_to_frames(self, codes):
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        codes = self._fit_codes_to_target_length(codes)

        scale = None
        if self.model.normalize:
            scale = torch.ones(codes.shape[0], 1, device=self.device)

        return [(codes.contiguous(), scale)]

    @torch.no_grad()
    def decode_codes_to_waveform(self, codes):
        frames = self.codes_to_frames(codes)
        return self.model.decode(frames)

    @torch.no_grad()
    def decode_from_codes(self, wav_path):
        """
        Verification utility:
        encode -> decode to ensure pipeline integrity.

        Returns:
            reconstructed waveform tensor
        """
        wav = self.load_audio(wav_path)
        encoded_frames = self.model.encode(wav)
        decoded = self.model.decode(encoded_frames)
        return decoded


# ============================================================
# SAVE OUTPUT AUDIO
# ============================================================


def save_audio(waveform, save_path, sample_rate=24000):
    """
    Save reconstructed waveform for listening verification.
    """
    waveform = waveform.squeeze(0).cpu()
    torchaudio.save(save_path, waveform, sample_rate)


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":
    wav_path = "Almost Like Being In Love - Red Garland (128k).wav"  # replace with your file

    embedder = EncodecEmbedder(
        target_dim=TARGET_EMBED_DIM,
        device=DEVICE
    )

    codes = embedder.encode(wav_path)

    print("==== EnCodec Code Verification ====")
    print("Code shape:", codes.shape)
    print("Code sample:", codes[0, :, :10])

    reconstructed = embedder.decode_codes_to_waveform(codes)
    save_audio(reconstructed, "reconstructed.wav")

    print("Reconstructed audio saved to reconstructed.wav")

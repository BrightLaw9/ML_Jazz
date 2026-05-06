"""
CLAP similarity loss for music latent diffusion.

This module computes weighted semantic alignment losses using
pretrained CLAP embeddings.

Inputs:
1. Original/reference audio filepath
2. Generated audio filepath
3. Target text prompt

Losses:
1. Audio-to-audio similarity loss
   Compare original audio vs generated audio

2. Audio-to-text similarity loss
   Compare generated audio vs target text prompt

Final weighted loss:
    total_loss = lambda_audio * audio_loss
               + lambda_text  * text_loss

Install:
    pip install laion-clap torch torchaudio librosa soundfile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAPSimilarityLoss(nn.Module):
    def __init__(
        self,
        lambda_audio=1.0,
        lambda_text=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.lambda_audio = lambda_audio
        self.lambda_text = lambda_text
        self.device = device

        import laion_clap

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        self.clap_model.to(device)
        self.clap_model.eval()

    def cosine_similarity_loss(self, x, y):
        """
        Convert cosine similarity into minimizable loss.

        similarity = 1 means perfect match
        loss = 0 means perfect match
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        similarity = (x * y).sum(dim=-1)
        loss = 1.0 - similarity.mean()
        return loss, similarity.mean()

    @torch.no_grad()
    def get_audio_embedding(self, audio_path):
        """
        Extract CLAP audio embedding from a filepath.

        Returns:
            tensor [1, D]
        """
        embedding = self.clap_model.get_audio_embedding_from_filelist(
            x=[audio_path],
            use_tensor=True
        )

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)

        return embedding.to(self.device).float()

    @torch.no_grad()
    def get_text_embedding(self, text_prompt):
        """
        Extract CLAP text embedding from a text prompt.

        Returns:
            tensor [1, D]
        """
        embedding = self.clap_model.get_text_embedding([text_prompt])

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)

        return embedding.to(self.device).float()

    def forward(
        self,
        original_audio_path,
        generated_audio_path,
        target_text_prompt
    ):
        """
        Compute weighted CLAP alignment loss.

        Step 1:
            original audio vs generated audio

        Step 2:
            generated audio vs target text

        Returns:
            total_loss
            logs dict
        """

        original_audio_emb = self.get_audio_embedding(original_audio_path)
        generated_audio_emb = self.get_audio_embedding(generated_audio_path)
        text_emb = self.get_text_embedding(target_text_prompt)

        # 1. Original audio vs Generated audio
        audio_loss, audio_similarity = self.cosine_similarity_loss(
            original_audio_emb,
            generated_audio_emb
        )

        # 2. Generated audio vs Target text
        text_loss, text_similarity = self.cosine_similarity_loss(
            generated_audio_emb,
            text_emb
        )

        total_loss = (
            self.lambda_audio * audio_loss +
            self.lambda_text * text_loss
        )

        logs = {
            "audio_loss": audio_loss,
            "audio_similarity": audio_similarity,
            "text_loss": text_loss,
            "text_similarity": text_similarity,
            "total_loss": total_loss,
        }

        return total_loss, logs


if __name__ == "__main__":
    loss_fn = CLAPSimilarityLoss(
        lambda_audio=1.2,
        lambda_text=1.0
    )

    original_audio = "garland_short.wav"
    generated_audio = "reconstructed.wav"
    target_prompt = "swing jazz trio with walking bass and brushed drums"

    total_loss, logs = loss_fn(
        original_audio,
        generated_audio,
        target_prompt
    )

    print("==== CLAP Similarity Loss Verification ====")
    print("Prompt:", target_prompt)
    print()

    for key, value in logs.items():
        print(f"{key}: {value}")

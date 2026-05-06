"""
CLAP-based text embedding module for music latent diffusion.

This replaces a simple learned embedding layer with a pretrained
CLAP text encoder so text prompts are mapped into a strong
semantic embedding space aligned with audio.

Install:
    pip install laion-clap torch

Example prompts:
    - "upbeat bebop jazz quartet with walking bass"
    - "melancholic solo piano in minor key"
"""

import torch
import torch.nn as nn


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


if __name__ == "__main__":
    model = TextEmbeddingModel(text_dim=512)

    prompts = [
        "upbeat bebop jazz quartet with walking bass and brushed drums",
        "melancholic solo piano with cinematic atmosphere"
    ]

    embeddings = model(prompts)

    print("==== CLAP Text Embedding Verification ====")
    print("Input prompts:")
    for p in prompts:
        print("-", p)

    print("\nEmbedding shape:", embeddings.shape)
    print("Expected batch size:", len(prompts))
    print("Sample values:", embeddings[0][:10])

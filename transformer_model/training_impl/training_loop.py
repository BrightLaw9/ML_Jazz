from torch.utils.data import DataLoader
import torch
import os
from .dataset import JazzMIDIDataset
from ..model import JazzTransformer

from ..constants import VOCAB_SIZE, SEQ_LEN, MODEL_CHECKPOINT_DIR, OPTIMIZER_CHECKPOINT_DIR, CHECKPOINT_FREQ

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    dataset = JazzMIDIDataset("train_midi_pt/", VOCAB_SIZE, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = JazzTransformer(vocab_size=VOCAB_SIZE, max_len=SEQ_LEN).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OPTIMIZER_CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(50):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)

            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % CHECKPOINT_FREQ == 0:
            # print(model.state_dict())
            torch.save(model.state_dict(), MODEL_CHECKPOINT_DIR + f"model_ckpt_{epoch}.pth")
            torch.save(optimizer.state_dict(), OPTIMIZER_CHECKPOINT_DIR + f"optim_ckpt_{epoch}.pth")

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

if __name__ == "__main__":
    print("Entering training loop...")
    train()

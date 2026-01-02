from torch.utils.data import DataLoader
import torch
import os
from .dataset import JazzMIDIDataset
from ..model import JazzTransformer

from ..constants import VOCAB_SIZE, SEQ_LEN, MODEL_CHECKPOINT_DIR, OPTIMIZER_CHECKPOINT_DIR, CHECKPOINT_FREQ, CKPT_MODEL, CKPT_OPTIM, EPOCH_NUM, NUM_EPOCHES, LEARN_RATE, TRAIN_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print("Vocab size ", VOCAB_SIZE)
    dataset = JazzMIDIDataset(TRAIN_DIR, VOCAB_SIZE, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = JazzTransformer(vocab_size=VOCAB_SIZE, max_len=SEQ_LEN).to(device)
    if CKPT_MODEL:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_DIR + CKPT_MODEL))
        print(model.state_dict())    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.97
    )
    if CKPT_OPTIM:
        optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_DIR + CKPT_OPTIM))
        print(optimizer.state_dict())
    
    epoch = EPOCH_NUM
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OPTIMIZER_CHECKPOINT_DIR, exist_ok=True)

    while epoch < NUM_EPOCHES:
        count = 0
        losses = []
        for batch in loader:
           batch = batch.to(device)
           logits = model(batch)

           loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )

           losses.append(loss.detach())
           count += 1
           if count % 1000 == 0:
               tlos = torch.stack(losses)
               mean = tlos.mean().item()
               std = tlos.std(unbiased=False).item()

               print(f"Mean: {mean:.4f}; Std: {std:.4f}")
               losses.clear()
               count = 0

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           #print(f"Batch loss: {loss.item():.4f}")

        # if epoch % CHECKPOINT_FREQ == 0:
        #     # print(model.state_dict())
        #     torch.save(model.state_dict(), MODEL_CHECKPOINT_DIR + f"model_ckpt_{epoch}.pth")
        #     torch.save(optimizer.state_dict(), OPTIMIZER_CHECKPOINT_DIR + f"optim_ckpt_{epoch}.pth")
        
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"LR: {lr}")
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
        epoch += 1


if __name__ == "__main__":
    print("Entering training loop...")
    train()

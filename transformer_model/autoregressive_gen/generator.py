import torch
import torch.nn.functional as F
from ..model import JazzTransformer
from ..constants import MODEL_CHECKPOINT_DIR, VOCAB_SIZE, SEQ_LEN, INFERENCE_MODEL, INITIAL_TOKENS_PATH, GENERATED_SAVE_DIR
import datetime

model = JazzTransformer(VOCAB_SIZE, max_len=SEQ_LEN)
model.load_state_dict(torch.load(MODEL_CHECKPOINT_DIR + INFERENCE_MODEL))

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

# primer: list of token IDs
primer_tokens = torch.load(INITIAL_TOKENS_PATH)
tokens = torch.tensor(primer_tokens, dtype=torch.long, device=device)
tokens = tokens.unsqueeze(0)  # batch size 1

generated = tokens.tolist()[0]  # start list

max_generate = 200 #1000

for _ in range(max_generate):
    logits = model(tokens)           # shape: [1, L, vocab_size]
    last_logits = logits[:, -1, :]   # take last token
    probs = F.softmax(last_logits, dim=-1)
    
    # sample next token
    next_token = torch.multinomial(probs, num_samples=1)
    
    # append
    generated.append(next_token.item())
    
    # update input sequence
    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

print(tokens)

torch.save(tokens, GENERATED_SAVE_DIR + f"sample_{datetime.datetime.now()}.pt")

import torch
import torch.nn.functional as F
from ..model import JazzTransformer
from ..constants import MODEL_CHECKPOINT_DIR, VOCAB_SIZE, SEQ_LEN, INFERENCE_MODEL, INITIAL_TOKENS_PATH, GENERATED_SAVE_DIR, NOTE_ON_OFFSET, DUR_OFFSET, TIME_SHIFT_OFFSET
from .decoder import token_to_event
from datetime import datetime
import os
import random

model = JazzTransformer(VOCAB_SIZE, max_len=SEQ_LEN)
# for param in model.parameters():
#     print(param.size())
model.load_state_dict(torch.load(MODEL_CHECKPOINT_DIR + INFERENCE_MODEL, map_location=torch.device('cpu')))

#print(model.state_dict())

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

# primer: list of token IDs
primer_tokens = torch.load(INITIAL_TOKENS_PATH)
LENGTH = 1000
if len(primer_tokens) >= LENGTH:
    start = random.randint(0, len(primer_tokens) - LENGTH)
    while (NOTE_ON_OFFSET <= primer_tokens[start] < DUR_OFFSET) or (DUR_OFFSET <= primer_tokens[start] < TIME_SHIFT_OFFSET):
        start = random.randint(0, len(primer_tokens) - LENGTH)
    primer_tokens = primer_tokens[start:start+LENGTH]

tokens = torch.tensor(primer_tokens, dtype=torch.long, device=device)
tokens = tokens.unsqueeze(0)  # batch size 1

generated = tokens.tolist()[0]  # start list

max_generate = 300 #1000

for count in range(max_generate):
    logits = model(tokens)           # shape: [1, L, vocab_size]
    last_logits = logits[:, -1, :]   # take last token
    probs = F.softmax(last_logits, dim=-1)
    
    #print("Probs, ", probs)
    # sample next token
    next_token = torch.multinomial(probs, num_samples=1)
    
    # append
    #generated.append(next_token.item())
    
    # update input sequence
    print("Token #: ", count, "; Token: ", token_to_event(next_token.item()))

    tokens = torch.cat([tokens, next_token], dim=1)

print(tokens)

os.makedirs(GENERATED_SAVE_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
torch.save(tokens, GENERATED_SAVE_DIR + f"sample_{ts}.pt")

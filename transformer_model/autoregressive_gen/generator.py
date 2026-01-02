import torch
import torch.nn.functional as F
from ..model import JazzTransformer
from ..constants import MODEL_CHECKPOINT_DIR, VOCAB_SIZE, SEQ_LEN, INFERENCE_MODEL, INITIAL_TOKENS_PATH, GENERATED_SAVE_DIR
from .decoder import token_to_event
import datetime
import os

model = JazzTransformer(VOCAB_SIZE, max_len=SEQ_LEN)
# for param in model.parameters():
#     print(param.size())
model.load_state_dict(torch.load(MODEL_CHECKPOINT_DIR + INFERENCE_MODEL, map_location=torch.device('cpu')))

#print(model.state_dict())

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

# primer: list of token IDs
primer_tokens = torch.load(INITIAL_TOKENS_PATH)
tokens = torch.tensor(primer_tokens, dtype=torch.long, device=device)
tokens = tokens.unsqueeze(0)  # batch size 1

generated = tokens.tolist()[0]  # start list

max_generate = 200 #1000

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
torch.save(tokens, GENERATED_SAVE_DIR + f"sample_{datetime.datetime.now()}.pt")

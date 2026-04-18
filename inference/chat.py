import torch
import torch.nn.functional as F
import sentencepiece as spm
from model.mini_gpt import GPT, GPTConfig

# ======================
# SETTINGS
# ======================

MODEL_PATH = "gpt_125m_final.pt"
TOKENIZER_PATH = "tokenizer/mini.model"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.9
REPETITION_PENALTY = 1.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# TOKENIZER
# ======================

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

# ======================
# MODEL
# ======================

config = GPTConfig()
model = GPT(config)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

BLOCK_SIZE = config.block_size

# Faster inference on GPU
if DEVICE == "cuda":
    model = model.half()

# ======================
# GENERATION
# ======================

def generate_stream(prompt):

    # Better prompt format for base LLM
    prompt = f"User: {prompt}\nAssistant:"

    input_ids = sp.encode(prompt)

    generated = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    print("\nModel: ", end="", flush=True)

    # ✅ proper SentencePiece streaming
    previous_text = sp.decode(generated[0].tolist())

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            if generated.size(1) > BLOCK_SIZE:
                generated = generated[:, -BLOCK_SIZE:]

            logits = model(generated)
            logits = logits[:, -1, :]

            # repetition penalty
            for token in set(generated[0].tolist()):
                logits[:, token] /= REPETITION_PENALTY

            logits = logits / TEMPERATURE

            # ===== TOP-K =====
            if TOP_K is not None:
                values, _ = torch.topk(logits, TOP_K)
                min_values = values[:, -1].unsqueeze(1)

                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, float("-inf")),
                    logits
                )

            # ===== TOP-P =====
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True
            )

            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1),
                dim=-1
            )

            remove = cumulative_probs > TOP_P
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = 0

            indices_to_remove = sorted_indices[remove]
            logits[:, indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            generated = torch.cat(
                (generated, next_token),
                dim=1
            )

            if next_token.item() == sp.eos_id():
                break

            # ✅ decode FULL sequence (correct way)
            new_text = sp.decode(generated[0].tolist())

            # print only new generated part
            print(
                new_text[len(previous_text):],
                end="",
                flush=True
            )

            previous_text = new_text

    print("\n")


# ======================
# CHAT LOOP
# ======================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    generate_stream(user_input)
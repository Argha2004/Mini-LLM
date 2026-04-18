import torch
import math
import sentencepiece as spm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.mini_gpt import GPT, GPTConfig

# ==============================
# WINDOWS STABILITY
# ==============================
torch.set_num_threads(1)

# ==============================
# SETTINGS
# ==============================
MODEL_1_PATH = "gpt_125m_final.pt"    #New model path
MODEL_2_PATH = "gpt_125m_final.pt"   #Old Model Path
TOKENIZER_PATH = "tokenizer/mini.model"
VAL_TEXT_PATH = "data/raw_val_split.txt"

BLOCK_SIZE = 128
BATCH_SIZE = 2          # Safe for GTX 1650
MAX_BATCHES = 200       # Fast evaluation limit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

# ==============================
# LOAD TOKENIZER
# ==============================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

# ==============================
# SAFE STREAMING DATASET
# ==============================
class TextDataset(Dataset):
    def __init__(self, text_path, block_size):

        print("Streaming validation text safely...")
        self.block_size = block_size
        tokens = []

        with open(text_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):

                line = line.strip()
                if not line:
                    continue

                ids = sp.encode(line)
                tokens.extend(ids)

                if i % 50000 == 0 and i > 0:
                    print(f"Loaded {i} lines")

        self.data = torch.tensor(tokens, dtype=torch.long)
        print("Total tokens:", len(self.data))

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y


# ==============================
# LOAD DATA
# ==============================
val_dataset = TextDataset(VAL_TEXT_PATH, BLOCK_SIZE)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

print("Total batches:", len(val_loader))

# ==============================
# LOAD MODEL
# ==============================
def load_model(path):

    print(f"\nLoading model: {path}")

    model = GPT(GPTConfig())

    checkpoint = torch.load(path, map_location=DEVICE)

    # Supports both save formats
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ==============================
# PERPLEXITY FUNCTION
# ==============================
def calculate_perplexity(model, dataloader):

    total_loss = 0
    total_tokens = 0

    print("\nStarting evaluation...")

    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):

            if step >= MAX_BATCHES:
                print("Reached benchmark limit")
                break

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)  # Only input x

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)

            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            if step % 20 == 0:
                print(f"Batch {step}/{MAX_BATCHES}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


# ==============================
# BENCHMARK
# ==============================
model1 = load_model(MODEL_1_PATH)
ppl1 = calculate_perplexity(model1, val_loader)

print(f"\nModel 1 Perplexity: {ppl1:.4f}")

model2 = load_model(MODEL_2_PATH)
ppl2 = calculate_perplexity(model2, val_loader)

print(f"\nModel 2 Perplexity: {ppl2:.4f}")

print("\n==========================")
print("FINAL RESULT")
print("==========================")
print(f"Model 1 PPL : {ppl1:.4f}")
print(f"Model 2 PPL : {ppl2:.4f}")

if ppl1 < ppl2:
    print("Model 1 is better")
else:
    print("Model 2 is better")
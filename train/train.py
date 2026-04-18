import torch
from model.mini_gpt import GPT, GPTConfig
import sentencepiece as spm
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import math
import random

MAX_STEPS = 10_000_000
BLOCK_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.backends.cudnn.benchmark = True

checkpoint_path = "checkpoint.pt"
final_model_path = "gpt_125m_final.pt"
metrics_file = "metrics.json"

# ======================
# TOKENIZER
# ======================
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/mini.model")

def encode(text):
    return sp.encode(text)

# ======================
# MODEL
# ======================
model = GPT(GPTConfig()).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

start_step = 0

# ======================
# LOAD CHECKPOINT
# ======================
def load_checkpoint():
    global start_step

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting fresh.")
        return

    print("Loading checkpoint...")

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    try:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0)

        print("Resuming from step:", start_step)

    except Exception as e:
        print("Checkpoint incompatible.")
        print(e)
        print("Starting fresh.")

load_checkpoint()

# ======================
# LOAD METRICS
# ======================
loss_history = []
ppl_history = []
acc_history = []

if os.path.exists(metrics_file):

    with open(metrics_file, "r") as f:
        m = json.load(f)

        loss_history = m["loss"]
        ppl_history = m["perplexity"]
        acc_history = m["accuracy"]

# ======================
# DATA
# ======================
with open("data/raw_train.txt", encoding="utf-8") as f:
    data = f.readlines()

print("Total lines:", len(data))

# ======================
# UTILITIES
# ======================
def token_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()

def get_sample():

    line = random.choice(data)
    tokens = encode(line)

    if len(tokens) < 2:
        return None, None

    tokens = tokens[:BLOCK_SIZE]

    x = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
    y = torch.tensor(tokens[1:]).unsqueeze(0).to(device)

    return x.long(), y.long()

# ======================
# SAVE CHECKPOINT
# ======================
def save_checkpoint(step):

    tmp_file = checkpoint_path + ".tmp"

    torch.save({
        "step": step + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, tmp_file)

    os.replace(tmp_file, checkpoint_path)

# ======================
# SAVE METRICS
# ======================
def save_metrics():

    if len(loss_history) < 5:
        return

    plt.figure(figsize=(14,5))

    plt.subplot(131)
    plt.plot(loss_history)
    plt.title("Loss")

    plt.subplot(132)
    plt.plot(ppl_history)
    plt.title("Perplexity")

    plt.subplot(133)
    plt.plot(acc_history)
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

    with open(metrics_file, "w") as f:
        json.dump({
            "loss": loss_history,
            "perplexity": ppl_history,
            "accuracy": acc_history
        }, f)

# ======================
# TRAIN LOOP
# ======================
try:

    model.train()

    for step in tqdm(range(start_step, MAX_STEPS)):

        x, y = get_sample()
        if x is None:
            continue

        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        acc = token_accuracy(logits, y)
        ppl = math.exp(loss.item())

        loss_history.append(loss.item())
        ppl_history.append(ppl)
        acc_history.append(acc)

        if step % 100 == 0:

            print(
                f"Step {step} | "
                f"Loss {loss.item():.4f} | "
                f"PPL {ppl:.2f} | "
                f"Acc {acc:.3f}"
            )

            save_checkpoint(step)
            save_metrics()

        if step % 1000 == 0 and step > 0:

            torch.save(
                model.state_dict(),
                f"backup_model_{step}.pt"
            )

except KeyboardInterrupt:

    print("\nTraining paused safely")

    save_checkpoint(step)
    save_metrics()

    torch.save(
        model.state_dict(),
        final_model_path
    )

    print("Model saved:", final_model_path)

    exit()

# ======================
# FINAL SAVE
# ======================
print("\nTraining complete")

torch.save(
    model.state_dict(),
    final_model_path
)

save_metrics()

print("Final model saved")
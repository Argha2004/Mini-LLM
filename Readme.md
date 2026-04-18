# 🧠 Mini-LLM (125M GPT from Scratch)

A lightweight GPT-style language model (~125M parameters) built and trained from scratch using PyTorch.
Designed to run entirely on consumer hardware (GTX 1650), demonstrating a complete LLM pipeline from data → tokenizer → training → inference.

---

## 🚀 Overview

This project implements a decoder-only Transformer (GPT architecture) and trains it on web text data using an efficient, memory-conscious pipeline.

Unlike using pre-trained models, this project focuses on **building everything from scratch**, including:

* Tokenizer
* Model architecture
* Training loop
* Dataset pipeline
* Inference system

---

## 🏗️ Architecture

| Component       | Value                          |
| --------------- | ------------------------------ |
| Model Type      | GPT (Decoder-only Transformer) |
| Parameters      | ~125M                          |
| Layers          | 12                             |
| Attention Heads | 12                             |
| Hidden Size     | 768                            |
| Context Length  | 1024                           |
| Vocabulary Size | 50,000                         |

---

## ⚙️ Features

* ✅ Custom SentencePiece tokenizer (BPE)
* ✅ Streaming dataset pipeline (low RAM usage)
* ✅ Token packing for efficient training
* ✅ Mixed precision (FP16) training
* ✅ Gradient accumulation (simulates larger batch sizes)
* ✅ Checkpoint saving
* ✅ Local text generation (chat-style inference)

---

## 🖥️ Hardware Used

* GPU: NVIDIA GTX 1650 (4GB VRAM)
* CPU: AMD Ryzen 5 5600H
* RAM: 16GB DDR4
* Storage: 1TB SSD

---

## 📂 Project Structure

```
Mini-LLM/
│
├── data/
│   └── download.py    #For Downloading SlimPajama-6B Dataset  
│
├── tokenizer/
│   └──tokenizer.py 
│
├── model/
│   └── gpt.py
│
├── train/
│   ├── train.py
│
├── inference/
│   └── chat.py
|
├── benchmark
│   └──mark.py       #For Checking The Benchmark Between Your Old and New Model
│
├── License
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/Mini-LLM.git
cd Mini-LLM

python -m venv llm
llm\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 📚 Dataset

Uses **OpenWebText** (public web corpus).

Download dataset:

```bash
python data/download.py
```

Recommended:

* Start with ~500k samples (~8–10GB)
* Scale gradually

---

## 🔤 Train Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

Outputs:

* `mini.model`
* `mini.vocab`

---

## 🏋️ Train Model

```bash
python train/train.py
```

Training includes:

* Streaming data loading
* Gradient accumulation
* Mixed precision (FP16)

---

## 💬 Run Inference

```bash
python inference/chat.py
```

Example:

```
Input: Hello, how are you?
Output: Hello, how are you doing today? I hope you are doing well...
```

---

## 📊 Training Details

| Metric          | Value                 |
| --------------- | --------------------- |
| Batch Size      | 1 (with accumulation) |
| Effective Batch | 8                     |
| Learning Rate   | 3e-4                  |
| Optimizer       | AdamW                 |
| Precision       | FP16                  |

---

## ⚡ Performance (GTX 1650)

* ~20–30 tokens/sec
* Stable training with low VRAM
* Usable model after ~3–5 days

---

## 📈 Results

* Learns sentence structure and grammar
* Generates coherent short text
* Demonstrates core LLM behavior at small scale

---

## ⚠️ Limitations

* Not comparable to large-scale models (GPT-3/4)
* Limited reasoning ability
* Trained on relatively small dataset

---

## 🔮 Future Improvements

* Flash Attention / faster attention kernels
* Better dataset (multi-source corpus)
* Instruction tuning (chat-style alignment)
* LoRA fine-tuning
* Web UI (Gradio)

---

## 🤝 Contributing

Contributions are welcome. Feel free to open issues or pull requests.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Arghadeep Pakhira
Computer Science Student | ML & Systems Enthusiast

---

## ⭐ Acknowledgements

* OpenWebText dataset
* HuggingFace Datasets
* PyTorch
* SentencePiece

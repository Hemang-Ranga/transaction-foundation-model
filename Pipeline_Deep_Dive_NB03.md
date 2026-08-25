# Deep Dive: Transaction Foundation Model Pipeline
## Part 2: Foundation Model Training (Notebook 03)

This document provides a comprehensive, deep-dive explanation of **Notebook 03** and its underlying configurations (`configs/pretrain_financial_decoder.yaml` & `scripts/train_decoder_model.py`). The primary goal of this stage is to take the tokenized corpora generated in Notebook 02 and use them to train a Transformer Foundation Model from scratch.

---

### 1. The Core Objective: Self-Supervised Learning
The goal here is **not** to train a fraud detector (yet). The goal is to train a Foundation Model. 

The model is trained using **Causal Language Modeling (Next-Token Prediction)**. It looks at a sequence of tokens and tries to predict what token comes next. 
```text
Input Context:  <bos> AMT_3 MERCH_1498 CAT_RETAIL ...
Target Token:   MCC_5912
```
Because the training corpus contains millions of transactions, the model is forced to learn the deep, underlying patterns of human financial behavior to successfully predict the next token. It learns:
- **Temporal patterns**: (e.g., Coffee at `HOUR_08` is common).
- **Merchant relationships**: (e.g., `CAT_AIRLINES` is usually followed by large `AMT_x` tokens).
- **User profiles**: It learns what a specific user (`CUST_1001`) typically does, and when they deviate from their norm.

---

### 2. The NeMo AutoModel Framework
Training a large Transformer model involves intense engineering: distributed multi-GPU training, mixed precision, complex optimizers, and checkpointing. 

To abstract this away, the pipeline uses **NVIDIA NeMo AutoModel**. 
- **What is it?** A framework designed to seamlessly train and fine-tune Large Language Models.
- **How it's used here:** We use the `TrainFinetuneRecipeForNextTokenPrediction` recipe in `scripts/train_decoder_model.py`. We don't write custom PyTorch training loops. Instead, we define the entire training job declaratively in a single YAML file (`configs/pretrain_financial_decoder.yaml`).
- NeMo automatically handles scaling via **FSDP2 (Fully Sharded Data Parallelism)**. If you run this on 1 GPU or 8x A100 GPUs using `torchrun`, NeMo handles the distribution of weights, gradients, and optimizer states without any code changes.
- It automatically saves and consolidates the final model into a standard **Hugging Face `safetensors` format**, making it completely interoperable with the standard `transformers` library for downstream tasks.

---

### 3. The Model Architecture
Instead of using an older architecture like GPT-2, this pipeline uses the state-of-the-art **LLaMA architecture** (`transformers.LlamaConfig`), but initializes it completely from scratch with a custom, compact size.

**Architectural Specs:**
- **Parameters**: ~29 Million (Small enough to train quickly, large enough to capture complex patterns).
- **Hidden Size**: 512 
- **Layers**: 8 
- **Vocab Size**: ~6,251 (Matching exactly the output size of our custom `FinancialTabularTokenizer`).

**Why LLaMA? Key Architectural Upgrades:**
1. **RoPE (Rotary Position Embeddings)**: Unlike standard absolute positional embeddings, RoPE encodes *relative* distances between tokens. This is crucial for sequences of up to 4096 tokens, allowing the model to better understand the time gaps between distant transactions.
2. **GQA (Grouped Query Attention)**: The model uses 8 Query heads but only 2 Key/Value heads. This drastically reduces the memory footprint during inference (KV cache) while maintaining the accuracy of Multi-Head Attention.
3. **SwiGLU & RMSNorm**: Modern activation functions and normalization layers that provide faster convergence and better stability than older GELU/LayerNorm setups.

---

### 4. Q&A: Deep Dive into the Mechanics

**Q: If the end goal is fraud detection, why are there NO fraud labels used in this training phase?**
**A:** This is the beauty of Foundation Models. Fraud is extremely rare (often <0.1% of transactions). If you train a model from scratch *only* on fraud labels, it struggles to learn because the signal is so sparse. By using self-supervised next-token prediction, the model learns the incredibly complex representation of what "Normal" looks like from 99.9% of the data. Later (in Notebook 04/05), we extract these rich representations and use a much simpler model (XGBoost) to easily draw a boundary around the anomalies (Fraud).

**Q: How does the model know which token belongs to which column (e.g., that `AMT_3` is an amount and not a ZIP code)?**
**A:** It doesn't explicitly know the tabular schema. However, because our tokenizer strictly enforces a 12-token repeating structure (Amount -> Merchant -> Category -> ... -> Customer ID), the model implicitly learns the grammar. It learns that position $N$ in a transaction is always an Amount, and position $N+1$ is always a Merchant.

**Q: Every user sequence in the corpus is 4096 tokens long. Does the model predict just one token at the end?**
**A:** No. In Causal Language Modeling, the model predicts the *next* token for **every single position** in the sequence simultaneously. For a sequence of 4096 tokens, the model calculates a loss over 4095 predictions at once. This makes training extremely data-efficient.

**Q: What is `MaskedCrossEntropy` in the YAML config?**
**A:** It is the standard loss function for Causal LMs. The "Masked" part refers to the causal mask (a lower-triangular matrix) applied during self-attention, ensuring that when predicting token $t+1$, the model is mathematically blocked from "looking ahead" at tokens $t+1$ to $4096$.

**Q: Can I swap the architecture to something else, like Mistral or Qwen?**
**A:** Yes. Because NeMo AutoModel uses the Hugging Face `from_config` API, you only need to change `_target_: transformers.LlamaConfig` to `transformers.MistralConfig` in the YAML file and update the relevant architecture parameters. No code changes are required.

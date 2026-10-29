# AGENT_HANDOFF: Transaction Foundation Model (M.Tech Thesis)

> **Audience**: AI Agents (Antigravity, Claude, ChatGPT, etc.) or Human Researchers picking up work on the GPU Server.  
> **Repository**: [https://github.com/Hemang-Ranga/transaction-foundation-model](https://github.com/Hemang-Ranga/transaction-foundation-model)  
> **Branch**: `main`  
> **Author**: Hemang Ranga (hemangranga@gmail.com)  
> **Thesis Topic**: Towards a Native Tabular Foundation Model: Variable-Length Field–Value Serialization and Boundary-Conditioned Representations for Financial Transactions.

---

## 1. Executive Summary & Thesis Objective

This repository builds upon NVIDIA's **Transaction Foundation Model (TFM)** blueprint. The goal of this M.Tech thesis is to transition transaction foundation models from rigid, dataset-specific, fixed-schema sequences into **schema-agnostic, variable-length tabular foundation models**.

### The Core Problem with the Baseline
In the baseline NVIDIA architecture:
- Every transaction is forced into a fixed sequence of exactly 12 fields (e.g. `[User, Card, Year, Month, Day, Time, Amount, Use, Merchant, City, State, Zip]`).
- Field identity is purely positional (token position 1 = User, token position 7 = Amount).
- Missing values require artificial imputation or dummy tokens (`-1`).
- Transactions cannot have variable numbers of columns (e.g. 13 vs 16 vs 25 features across different datasets).
- Pooling is naively done on the arbitrary "last token" of the fixed sequence.

### Our Research Formulation: Bipartite Serialization with `<EOT>`
Instead of a fixed 12-token tuple, each transaction is serialized as a variable-length sequence of explicit field-value pairs terminated by an explicit End-of-Transaction token:
```text
<BOS> <FIELD_A> VALUE_A <FIELD_B> VALUE_B ... <FIELD_K> VALUE_K <EOT>
```
1. **Field Identity Travels With Value**: `[<AMOUNT>, 123.45, <MERCHANT>, 82736, <MCC>, 5411, <EOT>]`.
2. **Dynamic Length**: If a transaction has 5 fields or 25 fields, it is serialized naturally without zero-padding corruption.
3. **Missing Field Robustness**: Missing fields are simply omitted from the sequence rather than imputed with dummy tokens.
4. **Causal Attention Compatibility**: Trained with standard causal language modeling (next-token prediction), preserving autoregressive mathematical consistency without intra-transaction information leakage.
5. **`<EOT>` Representation Hypothesis**: The transformer hidden state at the `<EOT>` position serves as the transaction-level embedding (since causal attention allows `<EOT>` to attend to all preceding field-value tokens).

---

## 2. Current State of Progress

| Milestone | Status | Key Output / Files | Notes |
|---|---|---|---|
| **Phase 1: Baseline Reproduction** | ✅ Completed | Notebooks `01`-`05` | Baseline reproduced on TabFormer & IEEE-CIS dataset. |
| **Stage 1 Presentation** | ✅ Delivered | `CS25MTECH14005_Hemang_Stage1 (1).pptx` | Stage 1 thesis defense completed. |
| **Comprehensive Architectural Audit** | ✅ Completed | `THESIS_BASELINE_ANALYSIS.md` | Formal critique, mathematical formulation, and thesis roadmap. |
| **Experiment 0: Baseline Tokenizer Audit** | ✅ Completed | `experiments/EXPERIMENT_LOG.md` | Discovered tight coupling in `financial_pipeline.py` & NeMo AutoModel. |
| **Experiment 1: Bipartite Tokenizer Prototype** | ✅ Passed | `experiments/01_prototype.py` | Built & validated variable-length bipartite serialization and sequence packing. |
| **Experiment 2: Model Training Smoke Test** | ✅ Passed | `experiments/02_smoke_test.py` | PyTorch causal transformer trained; loss dropped 2.9594 → 0.4910; `<EOT>` prediction verified. |
| **Phase 3 / Exp 3: GPU Pretraining & Evaluation** | 🔄 **NEXT STEP** | `experiments/` / `scripts/` | **To be executed on the GPU server.** |

---

## 3. Directory Map & Key Files

```text
transaction-foundation-model/
├── AGENT_HANDOFF.md                  <-- [YOU ARE HERE] Quick onboarding guide
├── THESIS_BASELINE_ANALYSIS.md       <-- Comprehensive research document, literature critique & roadmap
├── CS25MTECH14005_Hemang_Stage1 (1).pptx <-- Thesis Stage 1 Presentation slides
├── README.md                         <-- Original NVIDIA developer guide
│
├── experiments/                      <-- All M.Tech Thesis experimental code
│   ├── EXPERIMENT_LOG.md             <-- Running log of all hypothesis-driven experiments
│   ├── 01_prototype.py               <-- Validated BipartiteTokenizer implementation
│   └── 02_smoke_test.py              <-- Causal Transformer Decoder smoke test on synthetic transactions
│
├── notebooks/                        <-- Baseline reproduction notebooks
│   ├── 01_dataset_baseline.ipynb
│   ├── 02_seq_preproc_tokenization.ipynb
│   ├── 03_foundation_model_training.ipynb
│   ├── 04_inference_embedding_extraction.ipynb
│   └── 05_xgboost_fraud_detection.ipynb
│
├── src/                              <-- Baseline source modules
│   ├── tokenizer/financial_pipeline.py <-- Baseline rigid 12-field GPU tokenizer
│   ├── decoder_inference.py          <-- Baseline inference and last-token pooling
│   └── ...
└── scripts/                          <-- Baseline CLI scripts
    ├── train_decoder_model.py        <-- Baseline NeMo AutoModel next-token training script
    └── ...
```

---

## 4. Immediate Next Steps on the GPU Server

When you (the agent or researcher) connect to the GPU server, proceed with **Experiment 3**:

### Step 1: Environment Setup
You need a GPU environment with PyTorch and CUDA. Either:
- **Option A (NVIDIA NeMo Container - Recommended by blueprint)**:
  ```bash
  docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    --shm-size=8g \
    -p 8888:8888 \
    --ulimit memlock=-1 \
    nvcr.io/nvidia/nemo:25.09.01
  ```
- **Option B (Conda / venv with GPU PyTorch & RAPIDS)**:
  ```bash
  conda create -n tfm_gpu python=3.10 -y
  conda activate tfm_gpu
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install pandas numpy scikit-learn xgboost tqdm transformers
  # Optional: cuDF / RAPIDS if running GPU-accelerated dataframe ops
  ```

### Step 2: Validate Smoke Test on GPU
Run the existing smoke test on GPU to verify PyTorch/CUDA availability:
```bash
python experiments/02_smoke_test.py
```

### Step 3: Implement Experiment 3 (Controlled Representation Comparison)
1. **Data Source**: Use the processed IEEE-CIS transaction dataset (or TabFormer dataset).
2. **Pipeline**:
   - Wrap `experiments/01_prototype.py` into a dataset builder that processes real transaction rows into bipartite sequences (`<FIELD>` `VALUE` ... `<EOT>`).
   - Train a decoder model (either using `scripts/train_decoder_model.py` / NeMo AutoModel or standard HuggingFace `LlamaForCausalLM` / GPT-2 style causal decoder).
3. **Downstream Comparison**:
   - Extract `<EOT>` token embeddings for test transactions.
   - Extract baseline last-token embeddings.
   - Train downstream XGBoost classifiers on fraud detection.
   - Compare **ROC-AUC** and **AUPRC (PR-AUC)**.
4. **Log Results**: Record findings in `experiments/EXPERIMENT_LOG.md`.

---

## 5. Critical Technical Rules to Maintain
1. **Preserve Causal Attention**: Do not introduce bidirectional attention within transactions during next-token prediction, as it leaks future tokens and invalidates the causal LM objective.
2. **Explicit Bipartite Pairing**: Always keep field identity paired with values: `<FIELD_NAME>` followed by its value token.
3. **No Padding Intrusion**: Missing fields should be omitted, not replaced with arbitrary padding tokens.
4. **Treat `<EOT>` Pooling as a Hypothesis**: Measure whether `<EOT>` pooling outperforms or matches mean pooling and last-token pooling empirically.

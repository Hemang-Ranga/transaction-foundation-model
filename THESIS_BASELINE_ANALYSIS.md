# Thesis Baseline Analysis: Transaction Foundation Model

This document provides a comprehensive analysis of the repository based on NVIDIA's Transaction Foundation Model. It breaks down the architecture, identifies assumptions and limitations, and proposes concrete research directions for an M.Tech thesis.

## 1. Architecture and Data-Flow Diagram

```mermaid
flowchart TD
    subgraph Data Preparation [Notebook 01 & 02: Data Prep & Tokenization]
        A[Raw IEEE-CIS CSVs] --> B(Preprocessing & Feature Mapping)
        B --> C{Temporal Split}
        C -->|Train (80%)| D(Train Parquet)
        C -->|Val (10%)| E(Val Parquet)
        C -->|Test (10%)| F(Test Parquet)
        
        D --> G[FinancialTabularTokenizer]
        E --> G
        G -->|12 Tokens / Transaction| H[(Tokenized Sequences\nMax Len: 4096)]
    end

    subgraph Foundation Model Training [Notebook 03: Pre-training]
        H --> I(CLM Dataset Builder)
        I -->|Shifted Input/Labels| J[NeMo AutoModel]
        J --> K[Llama Decoder-Only\n29M Params, 8 Layers]
        K -->|Masked Cross Entropy| L(Next-Token Prediction)
        L -.->|Update Weights| K
        K --> M[(Saved Weights\nsafetensors)]
    end

    subgraph Embedding & Downstream [Notebook 04 & 05: Inference & XGBoost]
        F --> G2[FinancialTabularTokenizer]
        G2 --> N[Frozen Llama Model]
        M --> N
        N -->|Last-Token Pooling| O[512d Embeddings]
        O --> P(PCA Reduction)
        P --> Q[64d Embeddings]
        
        F --> R(Raw 13d Tabular Features)
        
        Q --> S(Concatenation)
        R --> S
        
        S -->|77d Combined Features| T[XGBoost Classifier]
        T --> U([Fraud Prediction\nOutput: AUPRC / ROC-AUC])
    end
```

## 2. Important Components Explained

1. **Preprocessing & Grouping (NB 01)**: Since the IEEE-CIS dataset lacks explicit `User` and `Card` IDs, proxies are created by concatenating columns (e.g., `card1` to `card5` for Card, plus `addr` and `email` for User). This allows grouping transactions into sequential histories.
2. **FinancialTabularTokenizer (src/tokenizer)**: A custom pipeline that maps tabular rows into a flat sequence of tokens. It strictly enforces a 12-token grammar per transaction: `[AMT, MERCH, CAT, MCC, HOUR, DOW, MONTH, CARD, CHIP, ZIP3, STATE, CUST]`. Numeric values (Amount) are bucketed; categoricals are hashed or mapped.
3. **Causal Language Modeling (CLM) Dataset**: Transactions are flattened into a 1D sequence of up to 4096 tokens (approx. 341 transactions). The task is next-token prediction: given tokens $0$ to $t$, predict token $t+1$.
4. **NeMo Llama Architecture**: A decoder-only transformer with modern features: RoPE (Rotary Position Embeddings) for relative distance, GQA (Grouped Query Attention) for memory efficiency, and SwiGLU activations. It is sized compactly (~29M params, 8 layers, 512 hidden size) for tabular data.
5. **Embedding Extraction (decoder_inference.py)**: The trained model is frozen and used for inference. To get a single representation for a transaction, the hidden state of the *last non-padding token* (the 12th token of the transaction) is extracted from the last layer. 
6. **PCA Reduction**: The 512-dimensional embeddings are reduced to 64 dimensions. This prevents the downstream XGBoost model from overfitting to the high-dimensional noise and balances the feature space against the 13 original tabular features.
7. **Downstream XGBoost (NB 05)**: A supervised model trained to predict fraud (`isFraud`). The best performing model concatenates the raw tabular features (13d) with the PCA-reduced embeddings (64d), yielding a 77d feature vector. This combined approach significantly boosts Average Precision (AP).

## 3. Assumptions Made by the Baseline

- **Strict Tabular Grammar**: The model assumes that the order of the 12 fields within a transaction doesn't negatively impact learning, and that the model will implicitly learn the schema from the repeating pattern.
- **Last-Token Sufficiency**: It assumes that the 12th token (Customer ID) contains all necessary contextual information about the entire transaction and the user's history, simply because it mathematically "attended" to the previous tokens.
- **Temporal Uniformity**: Standard RoPE assumes tokens are equally spaced. The baseline assumes the model can infer actual time elapsed purely from the `HOUR`, `DOW`, and `MONTH` categorical tokens, without explicitly passing continuous time deltas between transactions.
- **Unsupervised Pre-training is Sufficient**: It assumes that simply predicting the next field (e.g., predicting the merchant after seeing the amount) is enough to cluster anomalous behavior, without explicit anomaly or contrastive objectives during pre-training.

## 4. Weaknesses and Limitations

- **Embedding Bottleneck**: Last-token pooling compresses the entire transaction's nuance (Amount, Merchant, MCC) into the final token. This is an NLP paradigm (like a `[SEP]` or `[EOS]` token) that may not map well to tabular structures where the first token (Amount) is highly predictive.
- **High Dimensionality for Trees**: 512d embeddings are hostile to tree-based models (XGBoost). The necessity of PCA implies that a lot of the foundation model's output is either noise or redundant for the downstream task.
- **Lack of Velocity Modeling**: Fraud is highly dependent on "velocity" (e.g., 5 transactions in 2 minutes). While `HOUR` and `MONTH` are tokenized, the exact time elapsed between consecutive transactions is completely abstracted away by the uniform positional embeddings.
- **Sparse Fraud Signal in Pre-training**: Because the foundation model never sees fraud labels, its embeddings describe "general behavior" rather than "fraud vs. normal." This is why the embeddings alone perform poorly in NB05.

## 5. Components Easy to Modify Experimentally

- **Embedding Extraction Strategy**: `src/decoder_inference.py` is extremely modular. Changing how the 512d vector is pooled (e.g., mean pooling, concatenating specific tokens, attention pooling) requires no retraining of the foundation model.
- **Tokenizer Pipeline**: `src/tokenizer/financial_pipeline.py` is built linearly. Adding new tokens (like the optional `time_delta_s`), reordering tokens, or changing binning strategies is trivial.
- **Downstream Fusion**: Notebook 05 simply concatenates DataFrames. Experimenting with different fusion techniques (e.g., using a Multi-Layer Perceptron instead of XGBoost, or cross-attention fusion) is highly accessible.

## 6. Components Leading to Plausible Research Contributions

- **Tabular-Aware Pooling Mechanisms**: Moving beyond NLP-centric pooling to methods specifically designed for sequential tabular structures.
- **Time-Aware Attention / Continuous Time Modeling**: Replacing discrete time tokens with continuous time integration in Transformers for financial velocity modeling.
- **Self-Supervised Contrastive Formulations**: Augmenting the CLM objective with contrastive losses to explicitly separate "normal" transaction sequences from artificially corrupted ones.

---

## 7. Proposed Research Directions

### Direction 1: Tabular-Aware Field-Level Embedding Aggregation
* **Hypothesis**: Replacing last-token pooling with a mechanism that explicitly aggregates or concatenates critical field tokens (e.g., Amount, Merchant, MCC) will yield richer representations, improving downstream fraud detection.
* **Motivation**: In a causal model, the last token (Customer ID) attends to previous tokens, but it forces compression. Crucial fraud signals (like anomalous Amounts) are separated from the final token by 10 other fields. NLP uses last-token pooling, but tabular data needs field-level visibility.
* **Exact Modification**: Modify `decoder_inference.py`. Instead of extracting `hidden_states[:, seq_lengths, :]`, extract the hidden states corresponding specifically to the positions of the `AMT`, `MERCH`, and `MCC` tokens for that transaction. Concatenate these (e.g., 3 * 512 = 1536d), apply PCA to reduce to 64d, and use this new embedding.
* **Experiment Required**: Run the modified inference script on the test set using the *already trained* frozen Llama weights. Train the downstream XGBoost model on these new embeddings.
* **Baseline Comparison**: Compare test AUPRC against the baseline's 77d Combined model (which uses last-token pooling).
* **Expected Measurable Outcome**: A statistically significant increase in AUPRC due to the downstream model having direct access to context-aware embeddings of the Amount and Merchant, rather than a compressed summary.
* **Potential Novelty**: Proposing a tabular-specific pooling strategy for sequential Foundation Models, highlighting the inadequacy of standard NLP pooling for structured transaction data.
* **Computational Cost**: **Very Low**. Requires zero retraining of the foundation model. Only modifies the inference/extraction script and the XGBoost training step.
* **Possible Failure Mode**: Concatenating multiple token embeddings might introduce too much noise, reducing the efficacy of PCA and causing the XGBoost model to overfit.

### Direction 2: Time-Aware Attention for Velocity-Based Fraud
* **Hypothesis**: Integrating the continuous time elapsed between transactions explicitly into the Transformer's positional encoding (or as a continuous token) will significantly improve the model's ability to detect velocity-based fraud (e.g., rapid card testing).
* **Motivation**: Fraudsters execute multiple transactions rapidly. Currently, the RoPE positional embeddings treat two transactions 1 second apart identically to two transactions 3 days apart. The discrete tokens (`HOUR`, `DOW`) are too coarse to capture micro-velocity.
* **Exact Modification**: Enable `include_time_delta=True` in `financial_pipeline.py`. Alternatively (and for a stronger contribution), modify the `LlamaConfig` and `modeling_llama.py` (via HuggingFace) to implement Time-aware Relative Positional Encodings (e.g., injecting delta-t into the attention weight calculation, similar to TiSASRec).
* **Experiment Required**: Retrain the foundation model from scratch using the modified architecture/tokenizer. Extract embeddings, and evaluate on the downstream XGBoost pipeline. Perform a sub-population analysis evaluating AUPRC specifically on "high-velocity" users vs. "low-velocity" users.
* **Baseline Comparison**: Compare overall AUPRC against the baseline. More importantly, demonstrate a superior performance gap specifically on sequences with short inter-transaction times.
* **Expected Measurable Outcome**: Higher overall AUPRC, driven largely by massive improvements in detecting bursts of fraudulent transactions.
* **Potential Novelty**: Bridging sequential recommendation architectures (which handle irregular time gaps well) with Financial Foundation Models.
* **Computational Cost**: **High**. Requires full pre-training of the Foundation Model (Notebook 03) from scratch, which takes significant GPU hours, followed by inference and XGBoost training.
* **Possible Failure Mode**: Time gaps in the IEEE dataset might be extremely noisy or heavy-tailed, causing the model to over-index on time and ignore categorical features like MCC or Merchant.

### Direction 3: Sequence-Level Contrastive Learning (SimCLR for Transactions)
* **Hypothesis**: Augmenting the Next-Token Prediction (CLM) objective with a Sequence-Level Contrastive Loss will map user behaviors into a more separable latent space, resulting in embeddings that are highly predictive of fraud even without raw tabular features.
* **Motivation**: CLM learns transition probabilities (what token comes next). It does not explicitly force the global embedding space to cluster similar users or push away anomalous behavior. The baseline embeddings perform terribly on their own (AUPRC 0.0168) because they lack this global structural constraint.
* **Exact Modification**: Modify the training loop (which currently uses NeMo's `MaskedCrossEntropy`). Introduce data augmentations for user sequences (e.g., dropping random transactions, masking amounts, shuffling chronological order slightly). During training, pass two augmented views of the same sequence. Calculate the CLM loss as usual, but add an InfoNCE contrastive loss over the sequence-level pooled embeddings to pull augmented views of the same user together, and push different users apart.
* **Experiment Required**: Retrain the model with the combined loss (Total = CLM + Contrastive). Extract embeddings and evaluate the "Embeddings-Only" XGBoost model.
* **Baseline Comparison**: The primary comparison is against the baseline's "Embeddings-Only (64d)" model. You want to show that contrastively trained embeddings are vastly superior on their own. Visualize the latent space using t-SNE.
* **Expected Measurable Outcome**: A massive lift in the Embeddings-Only AUPRC (e.g., jumping from 0.0168 to >0.08), and a moderate lift in the Combined AUPRC. Better clustering in t-SNE visualizations.
* **Potential Novelty**: Applying self-supervised contrastive representation learning to multi-variate financial transaction sequences.
* **Computational Cost**: **Very High**. Requires writing a custom training loop or heavily modifying the NeMo recipe, processing multiple augmented views per batch (doubling memory requirements), and full pre-training.
* **Possible Failure Mode**: Augmentations for tabular sequences are notoriously difficult to get right. If the augmentation destroys the semantic meaning of the sequence (e.g., removing a crucial fraud indicator), the contrastive loss will force the model to ignore important features, deteriorating performance.

### Direction 4: Unified Tabular-Aware Foundation Model (Generation + Understanding)
* **Hypothesis**: Injecting tabular structural priors (Field/Column Embeddings) into the architecture and utilizing a multi-objective pre-training strategy (Causal Next-Token for generation + Bidirectional Masking for understanding) will yield a true foundation model capable of both highly realistic synthetic data generation and superior downstream representation.
* **Motivation**: The baseline relies on a flat 1D sequence and a purely causal (GPT-style) objective. While causal models are great at generating the next token, they often produce weaker dense embeddings compared to bidirectional models (like BERT). Furthermore, without structural priors, generative models can easily "break" the strict 12-column tabular schema when synthesizing data.
* **Exact Modification**: 
  1. **Architecture**: Introduce "Field Embeddings" (similar to positional embeddings, but indicating column index 1 to 12) added to the token embeddings before the first Transformer layer. 
  2. **Objective**: Implement a Unified framework (like UniLM). Train the model to alternate between Causal Language Modeling (for synthetic generation) and Masked Language Modeling (randomly masking out 15% of the fields and forcing the model to reconstruct them using bidirectional context).
* **Experiment Required**: 
  1. **Synthetic Generation**: Feed the model a few starting tokens and generate thousands of synthetic user histories. Evaluate realism using statistical distances (e.g., Kolmogorov-Smirnov test) against the real dataset.
  2. **Understanding**: Extract embeddings using the bidirectional context and train the downstream XGBoost fraud model.
* **Baseline Comparison**: Compare generation quality against a standard Markov Chain or GAN baseline. Compare fraud AUPRC against the baseline 77d XGBoost model.
* **Expected Measurable Outcome**: The synthetic data will maintain perfect tabular grammar without drifting, and the AUPRC on the fraud task will jump significantly due to the richer bidirectional embeddings.
* **Potential Novelty**: Adapting Unified Language Modeling (UniLM) architectures specifically for multi-variate tabular-temporal data, treating synthetic generation and fraud detection as two sides of the same coin.
* **Computational Cost**: **Very High**. Requires modifying the core embedding layer architecture and the loss calculation within the training loop, followed by extensive pre-training.
* **Possible Failure Mode**: Balancing two distinct loss objectives (Causal and Masked) can be notoriously unstable during pre-training. One objective might dominate the gradient, causing the model to either become a great generator but poor at representation, or vice versa.

### Direction 5: The "Native Tabular LLM" (Bipartite Tokenization & Variable Length)
* **Hypothesis**: By transitioning from atomic, fixed-length tokens (e.g., `AMT_4`) to bipartite field-value pairs (`<AMOUNT>` `4`) separated by a dynamic `<EOT>` token, the foundation model will become natively schema-agnostic, capable of processing datasets with varying column counts, missing data, and arbitrary field order.
* **Motivation**: LLMs understand variable-length sentences ending in a period. Tabular models should understand variable-length field sets ending in an `<EOT>`. The current baseline strictly relies on a fixed 12-token sequence. If a new dataset has 16 columns, or lacks a "ZIP Code", the baseline fails. Furthermore, in the baseline, the semantic meaning (Field Identity) is fused into the value, requiring the model to implicitly infer structural relationships based on absolute position.
* **Exact Modification**: 
  1. **Bipartite Tokenizer**: Rewrite `financial_pipeline.py`. Instead of yielding a single fused token per field, yield two: a special Field Token (`<MERCHANT>`) followed by the Value Token (`82736`).
  2. **Variable Sequence & `<EOT>`**: Concatenate these pairs dynamically per transaction (allowing missing fields to simply be absent) and terminate each transaction with an `<EOT>` (End Of Transaction) token.
  3. **Intra-Transaction Bidirectional Attention**: *(Future Extension)* Modify the causal attention mask so that all tokens *within* a single transaction (before the next `<EOT>`) can attend to each other bidirectionally. **For the first experiment, this is explicitly excluded** because it may expose future field/value tokens to prediction positions, causing information leakage. The initial implementation must use standard causal attention over the serialized sequence.
  4. **`<EOT>` Pooling (Hypothesis)**: Extract the hidden state of the `<EOT>` token to serve as the embedding vector for the entire transaction. This is a hypothesis to be rigorously tested against the existing last-token pooling method.
* **Schema Robustness Analysis (Goals to Investigate)**:
  - *Level 1 (Variable Length)*: Supported natively by the dynamic field-value list and `<EOT>` token.
  - *Level 2 (Variable Order)*: Not solved automatically under causal attention. Requires a controlled experiment (e.g., randomized field order during training) to test invariance.
  - *Level 3 (Missing Fields)*: Supported dynamically; the model processes whatever field tokens are present before `<EOT>`.
  - *Level 4 (Additional/Unseen Fields)*: Requires vocabulary handling (e.g., `<UNK_FIELD>`) to not crash on unseen data.
  - *Level 5 (Different names for similar concepts)* & *Level 6 (Incompatible fields)*: Not solved by this architecture alone. Requires a shared ontology or schema mapping prior to tokenization.
* **Experiment Required**: Prototype the variable-length serialization on a single dataset first. Test `<EOT>` prediction and loss behavior. Only proceed to cross-dataset transfer (investigating zero-shot capabilities) once the base architecture is validated.
* **Expected Measurable Outcome**: The goal is to investigate whether this variable-length sequence representation improves downstream AUPRC on fraud detection when compared fairly to the rigid baseline, and whether it enables generalization across datasets.

---

## 8. Proposed Master Thesis Plan: "The Native Tabular Foundation Model"

To achieve the ultimate goal of a "True Transaction Foundation Model" that is schema-agnostic, generative, and highly performant on downstream tasks, the thesis should execute the following cohesive pipeline:

1. **Phase 1: Bipartite Tokenization Prototype**
   Rewrite the tokenizer to map a single dataset into a vocabulary of `<FIELD>` `<VALUE>` pairs, concluding each row with an `<EOT>` token. Validate the sequence packing and boundary tests.
2. **Phase 2: Tabular-Aware Causal Transformer Baseline**
   Train a small model-training smoke test using standard causal attention over the new serialized representation. Investigate whether the loss behaves sensibly and `<EOT>` is predicted correctly.
3. **Phase 3: Controlled Representation Comparison**
   Design a fair comparison between the original fixed-schema representation and the proposed variable-length representation on downstream fraud tasks. Test the `<EOT>` pooling hypothesis against last-token pooling.
4. **Phase 4: Schema Robustness & Cross-Dataset Transferability**
   If validated, investigate robustness to missing fields and randomized field order. Finally, attempt cross-dataset transferability (investigating zero-shot/few-shot goals) using a unified ontology.

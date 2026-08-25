# Deep Dive: Transaction Foundation Model Pipeline
## Part 1: Sequential Preprocessing and Tokenization (Notebook 02)

This document provides a comprehensive, deep-dive explanation of **Notebook 02** and its underlying source code (`src/tokenizer/financial_pipeline.py`). The primary goal of this stage in the pipeline is to convert raw, tabular financial transactions into a sequence of standardized "tokens" that a Transformer language model can ingest to learn user spending behavior over time.

---

### 1. The Core Problem: Why Custom Tokenization?
Standard Large Language Model (LLM) tokenizers, like GPT-2's Byte Pair Encoding (BPE), are optimized for natural language, not tabular financial data. If you feed a raw transaction like `$42.75, Walmart #8832, 95113` into a standard LLM tokenizer, several issues arise:

1. **Loss of Semantic Meaning**: `$42.75` might be split into `['$', '42', '.', '75']`. To a model, the subword `42` here looks exactly like `42` in an address. It loses the magnitude of the dollar amount.
2. **Entity Fragmentation**: `3527213246127876953` (a merchant ID) explodes into 10+ meaningless digit subwords.
3. **Context Window Exhaustion**: A single transaction takes ~39 tokens with BPE. In a standard 4096-token context window, the model could only "look back" at ~102 transactions.

**The Solution**: A **Domain-Specific Financial Tokenizer**. It maps every field in a transaction row to exactly **one semantically meaningful token** (1 transaction = 12 tokens). This reduces token count by 3-4x, allowing the model to see ~315 transactions within a 4096 context window, capturing months of sequential spending history.

---

### 2. The 12-Step Financial Tokenizer Pipeline
The tokenization is governed by the `FinancialTokenizerPipeline` class in `src/tokenizer/financial_pipeline.py`. It takes a raw cuDF DataFrame and processes it through 12 specific steps.

#### A. Preprocessing Step
Before tokenizing, the data is cleaned and normalized:
- Dollar signs (`$`) are stripped.
- Dates (`Year`, `Month`, `Day`, `Time`) are concatenated and cast to proper `datetime` objects to easily extract components.
- ZIP codes are cleaned (e.g., removing `.0`).

#### B. The 12 Token Fields
Each transaction is converted into exactly 12 tokens. Here is the deep dive into how each field is generated:

1. **Amount (`AMT_x`)**:
   - **Logic**: It uses Log-Scale Binning. Instead of exact dollar amounts (which are too sparse), it classifies the amount into 7 fixed bins: `$0-10`, `$10-50`, `$50-100`, `$100-500`, `$500-1000`, `$1000-5000`, and `$5000+`.
   - **Token Example**: `AMT_1` (for a $42.75 transaction).
   
2. **Merchant (`MERCH_x`)**:
   - **Logic**: The merchant name is converted to uppercase, stripped of special characters, and then **hashed** into a fixed number of buckets (default 2000). 
   - **Why**: Standardizes varying merchant representations and provides a privacy layer by anonymizing the explicit name into a bounded vocabulary.
   - **Token Example**: `MERCH_667`

3. **Industry Category (`CAT_x`)**:
   - **Logic**: Maps the Merchant Category Code (MCC) to high-level industry ranges (e.g., 5000-5599 maps to `RETAIL`, 3000-3299 maps to `AIRLINES`).
   - **Token Example**: `CAT_RETAIL`

4. **Merchant Category Code (`MCC_x`)**:
   - **Logic**: Directly maps known MCC codes to a token.
   - **Token Example**: `MCC_5411`

5-7. **Temporal Tokens (`HOUR_x`, `DOW_x`, `MONTH_x`)**:
   - **Logic**: Time is extracted and padded. Hour of day (0-23), Day of week (0-6), Month (1-12).
   - **Why**: Allows the model to learn temporal habits (e.g., spending heavily on weekends, or buying coffee at `HOUR_08`).
   - **Token Examples**: `HOUR_09`, `DOW_2`, `MONTH_01`

8. **Card Index (`CARD_x`)**:
   - **Logic**: Clamps the user's card index (0-9).
   - **Token Example**: `CARD_0`

9. **Chip Type (`CHIP_x`)**:
   - **Logic**: Maps transaction type to `SWIPE`, `CHIP`, or `ONLINE`.
   - **Token Example**: `CHIP_CHIP`

10. **ZIP Code Region (`ZIP3_x`)**:
   - **Logic**: Truncates the ZIP code to the first 3 digits.
   - **Why**: A 5-digit zip code provides too vast a vocabulary and privacy risks. A 3-digit zip code perfectly captures the regional geographic location.
   - **Token Example**: `ZIP3_951`

11. **State (`STATE_x`)**:
   - **Logic**: Standard 2-letter US state abbreviations.
   - **Token Example**: `STATE_CA`

12. **Customer ID (`CUST_x`)**:
   - **Logic**: User IDs bounded to a max index.
   - **Token Example**: `CUST_1001`

---

### 3. Generating the Sequence Corpus (Context Window Deep Dive)
Once individual transactions are tokenized, they must be formatted into **sequences** for the language model to read, just like sentences in a book.

#### Grouping by User Context
Transactions are not fed into the model randomly. The pipeline groups all transactions by **`['user', 'card']`** and sorts them chronologically (`time_full`). 
This ensures the foundation model's context window represents the **true historical timeline** of a specific user's specific credit card. 

#### Chunking and Framing
Because an LM has a maximum context window limit (e.g., 4096 tokens), the pipeline chunks a user's long history into blocks of exactly **315 transactions**.
- `315 transactions * (12 tokens per transaction + 1 separator token) + <bos> + <eos> ≈ 4096 tokens`.

The pipeline frames these sequences using special control tokens:
- `<bos>`: Beginning of sequence.
- `<sep>`: Separator between individual transactions.
- `<eos>`: End of sequence.

**Final Formatted Output Sequence:**
```text
<bos> AMT_1 MERCH_667 ... CUST_1001 <sep> AMT_3 MERCH_882 ... <eos>
```

#### Q&A: Deep Dive into the Context Window

**Q: Is it a separate context window for separate user+card combinations?**
**A:** Yes, absolutely. Transactions are strictly grouped by `user` and `card`. Different `user+card` combinations will never be mixed into the same sequence or context window. This ensures the model only learns the continuous behavior of one specific account at a time.

**Q: What if a user+card has only 20 transactions?**
**A:** If a user only has 20 transactions, they form a single sequence (chunk) that is much shorter than 4096 tokens. The tokenizer outputs this exact sequence of 20 transactions without padding. Later, during the foundation model training phase (in NeMo), these shorter sequences are dynamically padded with `<pad>` tokens (or packed together) to fill the GPU efficiently.

**Q: What happens if a user+card exceeds 4096 tokens (e.g., 360 transactions)?**
**A:** If a user has 360 transactions, the tokenizer splits them chronologically based on the chunk size limit (315 transactions). 
- **Chunk 1**: Transactions 1 through 315.
- **Chunk 2**: Transactions 316 through 360.
These chunks are written as entirely separate lines in the corpus. During standard pretraining, the model processes them as independent training examples. The Transformer's self-attention mechanism cannot "look across" the boundary from Chunk 2 back into Chunk 1.

**Q: Does the model know that Chunk 1 and Chunk 2 belong to the same user if they are split?**
**A:** Implicitly, yes. Because every transaction includes the `CUST_x` and `CARD_x` tokens, the model sees the exact same identity tokens in both chunks. However, because they are processed as separate sequences, it must rely on its learned internal representations of that `CUST_x` token rather than direct attention to the previous transactions.

**Q: What if a user has zero transactions?**
**A:** They simply won't appear in the tokenized output, because the `groupby` operation operates only on existing transaction rows.

#### Final Corpora
The notebook processes the Train, Validation, and Test parquet splits (generated in Notebook 01) and saves these sequential text chunks as:
- `train_corpus.txt`
- `val_corpus.txt`
- `test_corpus.txt`

These text files are now the exact equivalent of "text documents" that you would use to train a Large Language Model (like LLaMA or GPT), setting the stage perfectly for **Notebook 03 (Foundation Model Pretraining)**.

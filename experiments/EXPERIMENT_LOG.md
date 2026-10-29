# Experiment Log

## Experiment 0: Baseline Audit
**Hypothesis**: The existing baseline successfully loads data, tokenizes into a fixed 12-token grammar, and trains a Causal LM using standard next-token prediction, where embeddings are extracted via last-token pooling.
**Changed Component**: None.
**Unchanged Components**: All.
**Dataset & Schema**: IEEE-CIS mapped to Tabformer schema (12 strict fields).
**Result**: Verified by inspecting `src/tokenizer/financial_pipeline.py` (fixed 12 fields per transaction) and `src/decoder_inference.py` (last-token pooling).
**Interpretation**: The baseline works, but it strictly couples field identity to value (e.g. `AMT_4`), flattening transactions into rigid sequences without explicit boundaries, limiting schema robustness.
**Next Action**: Implement a variable-length bipartite serialization prototype to decouple field identity from values.

## Experiment 1: Variable-Length Serialization Prototype
**Hypothesis**: A tokenizer can reliably construct transactions as bipartite field-value pairs (`<FIELD>` `VALUE`) terminated by an `<EOT>` token, gracefully handling missing fields and producing valid input IDs and attention masks for causal language modeling.
**Changed Component**: Tokenizer and dataset sequence packer.
**Unchanged Components**: Dataset (IEEE-CIS temporal splits).
**Dataset & Schema**: IEEE-CIS.
**Status**: Completed and passed. The tokenizer correctly dynamically constructs variable-length sequences, handles missing fields without padding corruption, and generates valid label sequences for causal LM.

## Experiment 2: Small Model-Training Smoke Test
**Hypothesis**: A causal Transformer model can successfully process the packed variable-length bipartite sequences, compute sensible masked cross-entropy loss, and learn to predict valid field/value tokens and transaction boundaries (`<EOT>`).
**Changed Component**: Raw PyTorch Transformer Decoder training loop.
**Unchanged Components**: `BipartiteTokenizer` from Exp 1.
**Dataset & Schema**: Synthetic IEEE-CIS subset (AMOUNT, MERCHANT, MCC).
**Status**: Completed and passed.
**Result**: Loss successfully converged from 2.9594 to 0.4910 over 50 epochs. The model successfully predicted logical next tokens (e.g. predicting `<MCC>` or `<EOT>` after observing `MERCHANT` values) demonstrating that the sequence serialization is mathematically compatible with standard causal attention.
**Next Action**: Implement the full cross-validation and controlled representation comparison (Experiment 3) using the full dataset and NeMo AutoModel once the environment dependencies are available.

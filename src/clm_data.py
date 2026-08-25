import sys
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import torch
from torch.utils.data import Dataset

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

class FinancialCLMDataset(Dataset):
    def __init__(self, sequences: List[List[int]], seq_length: int = 4096, pad_token_id: int = 0):
        self.sequences = sequences
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        
        # Remove padding if the tokenizer already padded it
        while len(tokens) > 0 and tokens[-1] == self.pad_token_id:
            tokens.pop()
            
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[:self.seq_length + 1]
            
        # Shift inputs and labels for causal LM
        # To predict token[i], the model should see tokens[0...i-1]
        input_tokens = tokens[:-1] if len(tokens) > 1 else tokens
        label_tokens = tokens[1:] if len(tokens) > 1 else tokens

        input_ids = np.full(self.seq_length, self.pad_token_id, dtype=np.int64)
        input_ids[:len(input_tokens)] = input_tokens

        labels = np.full(self.seq_length, -100, dtype=np.int64)
        labels[:len(label_tokens)] = label_tokens

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }

def load_corpus_and_tokenize(data_path: Union[str, Path], merchant_hash_size: int = 2000, seq_length: int = 4096) -> FinancialCLMDataset:
    from tokenizer.financial_tokenizer import FinancialTabularTokenizer
    tokenizer = FinancialTabularTokenizer(merchant_hash_size=merchant_hash_size, category_hierarchy=True, temporal_encoding=True)
    data_path = Path(data_path)
    print(f"Loading corpus from {data_path}...")
    sequences = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            token_ids = tokenizer.encode(line, max_length=seq_length + 1)
            sequences.append(token_ids)
    print(f"  Loaded {len(sequences):,} sequences (seq_length={seq_length})")
    return FinancialCLMDataset(sequences=sequences, seq_length=seq_length, pad_token_id=tokenizer.pad_token_id)

def build_financial_clm_dataset(data_path: str, merchant_hash_size: int = 2000, seq_length: int = 4096, **kwargs) -> FinancialCLMDataset:
    return load_corpus_and_tokenize(data_path=data_path, merchant_hash_size=merchant_hash_size, seq_length=seq_length)

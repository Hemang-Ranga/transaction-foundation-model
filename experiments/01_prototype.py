import unittest

class BipartiteTokenizer:
    """
    A prototype tokenizer for Native Tabular Foundation Models.
    Serializes a row into [<FIELD_ID>, <VALUE>, ... <EOT>].
    """
    def __init__(self, fields):
        self.fields = fields
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.eot_token = "<EOT>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.vocab = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
            self.eot_token: 4,
        }
        
        for f in fields:
            self.vocab[f"<{f}>"] = len(self.vocab)
            
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
    def add_value_to_vocab(self, field, value):
        token = str(value)
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.id_to_token[self.vocab[token]] = token
            
    def encode_row(self, row, drop_fields=None):
        """Encode a single dictionary representing a transaction."""
        tokens = []
        for f in self.fields:
            if drop_fields and f in drop_fields:
                continue
            val = row.get(f)
            if val is not None and val != "":
                tokens.append(self.vocab[f"<{f}>"])
                val_str = str(val)
                tokens.append(self.vocab.get(val_str, self.vocab[self.unk_token]))
        tokens.append(self.vocab[self.eot_token])
        return tokens

    def pack_sequences(self, rows, seq_length=128):
        """Pack a list of transactions into a sequence, yielding causal LM outputs."""
        seq = [self.vocab[self.bos_token]]
        for row in rows:
            encoded = self.encode_row(row)
            if len(seq) + len(encoded) > seq_length:
                break
            seq.extend(encoded)
            
        while len(seq) < seq_length + 1:
            seq.append(self.vocab[self.pad_token])
            
        input_ids = seq[:-1]
        labels = seq[1:].copy()
        
        for i in range(len(labels)):
            if labels[i] == self.vocab[self.pad_token]:
                labels[i] = -100
                
        attention_mask = [1 if t != self.vocab[self.pad_token] else 0 for t in input_ids]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

class TestBipartiteSerialization(unittest.TestCase):
    def setUp(self):
        self.fields = ["AMOUNT", "MERCHANT", "MCC"]
        self.tokenizer = BipartiteTokenizer(self.fields)
        self.tokenizer.add_value_to_vocab("AMOUNT", "100.0")
        self.tokenizer.add_value_to_vocab("MERCHANT", "WALMART")
        self.tokenizer.add_value_to_vocab("MCC", "5411")
        self.tokenizer.add_value_to_vocab("AMOUNT", "50.0")
        self.tokenizer.add_value_to_vocab("MERCHANT", "TARGET")
        
    def test_full_transaction(self):
        row = {"AMOUNT": "100.0", "MERCHANT": "WALMART", "MCC": "5411"}
        tokens = self.tokenizer.encode_row(row)
        
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[-1], self.tokenizer.vocab["<EOT>"])
        self.assertEqual(tokens[0], self.tokenizer.vocab["<AMOUNT>"])
        
    def test_missing_field(self):
        row = {"AMOUNT": "50.0", "MERCHANT": "TARGET"}
        tokens = self.tokenizer.encode_row(row, drop_fields=["MCC"])
        
        self.assertEqual(len(tokens), 5)
        self.assertEqual(tokens[-1], self.tokenizer.vocab["<EOT>"])
        
    def test_packing_and_labels(self):
        rows = [
            {"AMOUNT": "100.0", "MERCHANT": "WALMART", "MCC": "5411"},
            {"AMOUNT": "50.0", "MERCHANT": "TARGET"} # missing MCC
        ]
        
        result = self.tokenizer.pack_sequences(rows, seq_length=15)
        input_ids = result["input_ids"]
        labels = result["labels"]
        mask = result["attention_mask"]
        
        self.assertEqual(len(input_ids), 15)
        self.assertEqual(len(labels), 15)
        
        self.assertEqual(input_ids[0], self.tokenizer.vocab["<BOS>"])
        self.assertEqual(labels[0], self.tokenizer.vocab["<AMOUNT>"])
        
        pad_id = self.tokenizer.vocab["<PAD>"]
        num_valid = 1 + 7 + 5 # BOS + row1 + row2 = 13 tokens
        self.assertEqual(sum(mask), 13)
        self.assertEqual(input_ids[13], pad_id)
        self.assertEqual(labels[13], -100)

if __name__ == "__main__":
    unittest.main()

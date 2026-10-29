import torch
import torch.nn as nn
import torch.optim as optim
from importlib import import_module
import sys
import os

# Import BipartiteTokenizer from 01_prototype.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
prototype = import_module("01_prototype")
BipartiteTokenizer = prototype.BipartiteTokenizer

class TinyCausalModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        # Causal mask: upper triangular is -inf
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        emb = self.embedding(x)
        out = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.fc_out(out)
        return logits

def run_smoke_test():
    print("Initializing BipartiteTokenizer...")
    fields = ["AMOUNT", "MERCHANT", "MCC"]
    tokenizer = BipartiteTokenizer(fields)
    
    # Add some dummy values
    amounts = ["10.0", "50.0", "100.0"]
    merchants = ["AMAZON", "WALMART", "STARBUCKS"]
    mccs = ["5411", "5812", "5942"]
    
    for v in amounts: tokenizer.add_value_to_vocab("AMOUNT", v)
    for v in merchants: tokenizer.add_value_to_vocab("MERCHANT", v)
    for v in mccs: tokenizer.add_value_to_vocab("MCC", v)
    
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}")
    
    print("Generating synthetic transactions...")
    import random
    random.seed(42)
    rows = []
    for _ in range(500):
        row = {
            "AMOUNT": random.choice(amounts),
            "MERCHANT": random.choice(merchants),
        }
        # Randomly omit MCC to test missing fields
        if random.random() > 0.3:
            row["MCC"] = random.choice(mccs)
        rows.append(row)
        
    print("Packing sequences...")
    # Pack into batches of seq_length
    seq_length = 32
    batch_size = 16
    
    # Pack all rows into one giant sequence, then slice
    all_batches = []
    for i in range(0, len(rows), 5):
        chunk = rows[i:i+5]
        packed = tokenizer.pack_sequences(chunk, seq_length=seq_length)
        all_batches.append(packed)
        
    if len(all_batches) < batch_size:
        print("Not enough data generated.")
        return
        
    # Create tensors
    input_ids = torch.tensor([b["input_ids"] for b in all_batches[:batch_size]], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in all_batches[:batch_size]], dtype=torch.long)
    
    print("Initializing TinyCausalModel...")
    model = TinyCausalModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("Starting smoke test training loop...")
    epochs = 50
    initial_loss = None
    final_loss = None
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(input_ids)
        # Flatten for CE Loss
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease!"
    
    print("Testing <EOT> prediction...")
    model.eval()
    with torch.no_grad():
        # Take a partial transaction and see what it predicts
        # e.g. <AMOUNT> 10.0 <MERCHANT> AMAZON -> expected next could be <MCC> or <EOT>
        test_row = {"AMOUNT": "10.0", "MERCHANT": "AMAZON"}
        tokens = tokenizer.encode_row(test_row)
        # Drop the <EOT> so the model has to predict it or MCC
        prompt = [tokenizer.vocab["<BOS>"]] + tokens[:-1] 
        input_tensor = torch.tensor([prompt])
        
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        predicted_token_id = torch.argmax(next_token_logits).item()
        predicted_token = tokenizer.id_to_token[predicted_token_id]
        
        print(f"Prompt tokens: {[tokenizer.id_to_token[t] for t in prompt]}")
        print(f"Predicted next token: {predicted_token}")
        
        assert predicted_token in ["<EOT>", "<MCC>"], f"Predicted unexpected token: {predicted_token}"

    print("Experiment 2 Smoke Test PASSED.")

if __name__ == "__main__":
    run_smoke_test()

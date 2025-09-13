# train.py
import torch
import torch.nn as nn
from model import SimpleGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
vocab_size = 100
embed_size = 64
num_layers = 2
heads = 4
forward_expansion = 4
dropout = 0.1
max_length = 32
lr = 3e-4

# Model
model = SimpleGPT(vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Fake training data (you'll replace this with real tokens later)
batch_size = 4
input_ids = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)
targets = input_ids.clone().to(device)

# Training loop
model.train()
for epoch in range(10):
    outputs = model(input_ids)
    outputs = outputs.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    loss = loss_fn(outputs, targets_flat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

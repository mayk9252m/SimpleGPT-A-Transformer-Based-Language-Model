# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_len, embed_size = x.shape
        H = self.heads

        values = self.values(x).view(N, seq_len, H, -1)
        keys = self.keys(x).view(N, seq_len, H, -1)
        queries = self.queries(x).view(N, seq_len, H, -1)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, seq_len, self.embed_size)

        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.attention(x)
        x = self.dropout(self.norm1(attn + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size).to(device)
        self.position_embedding = nn.Embedding(max_length, embed_size).to(device)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion).to(device)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size).to(device)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            out = layer(out)
        return self.fc_out(out)

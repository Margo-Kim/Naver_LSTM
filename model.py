# model.py
import torch.nn as nn
import torch.nn.functional as F 
import torch
mps_device = torch.device("mps")

class LSTM(nn.Module):
    def __init__(self,vocab_len, dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=dim, padding_idx=vocab_len -1)
        self.lstm = nn.LSTM(dim, hidden_size= hidden_dim, num_layers= num_layers, bidirectional=True, batch_first=True, dropout=dropout)  #dim is the input size of lstm 
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.fc1 = nn.Linear(hidden_dim *2, 64) 
        self.fc = nn.Linear(64, 1)
        
    
    
    def forward(self, x):
        emb = self.embedding(x)
        output, _  = self.lstm(emb)
        M = self.tanh1(output)
        
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = F.softmax((M * self.w).sum(dim=2), dim=1).unsqueeze(-1)
        out = output * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        
        return out
        
        
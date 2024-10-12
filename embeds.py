import torch, math 
from torch import nn 

class EmbeddingLayer(nn.Module):
    def __init__(self, dimentions: int, vocab_size: int):
        super().__init__() 
        self.dimentions = dimentions 
        self.embeddings = nn.Embedding(vocab_size, dimentions) 

    def forward(self, input):
        return self.embeddings(input) * math.sqrt(self.dimentions) 
    
class PositionalEncoding(nn.Module):
    def __init__(self, dimensions, max_seq_length, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) 
        positional_encoding = torch.zeros(max_seq_length, dimensions) 

        pos = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0) / dimensions))
 
        positional_encoding[:, 0::2] = torch.sin(pos * div_term) 
        positional_encoding[:, 1::2] = torch.cos(pos * div_term) 
        positional_encoding = positional_encoding.unsqueeze(0) 

        self.register_buffer('positional_encoding', positional_encoding) 

    def forward(self, input_embedding):
        input_embedding = input_embedding + self.positional_encoding[:, :input_embedding.size(1), :]
        return self.dropout(input_embedding)
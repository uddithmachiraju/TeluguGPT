import torch 
from torch import nn 
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, dimensions, num_heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.num_heads = num_heads
        assert dimensions % num_heads == 0, 'Dimensions must be divisible by number of Heads'

        self.d_k = dimensions // num_heads

        self.w_q = nn.Linear(dimensions, dimensions, bias=False)
        self.w_k = nn.Linear(dimensions, dimensions, bias=False)
        self.w_v = nn.Linear(dimensions, dimensions, bias=False)
        self.w_o = nn.Linear(dimensions, dimensions, bias=False)

    def forward(self, q, k, v, encoder_mask=None):
        batch_size = q.size(0)

        # Linear projections
        query = self.w_q(q)  # (batch_size, seq_len, dimensions)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape and transpose for multi-head attention
        # New shape: (batch_size, num_heads, seq_len, d_k)
        query = query.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        # Attention Score: (batch_size, num_heads, seq_len, seq_len)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if encoder_mask is not None:
            # Assuming encoder_mask shape: (batch_size, 1, 1, seq_len) or similar
            attention_score = attention_score.masked_fill(encoder_mask == 0, -1e9)

        attention_weights = torch.softmax(attention_score, dim=-1)

        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        # Attention Output: (batch_size, num_heads, seq_len, d_k)
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate heads and put through final linear layer
        # First, transpose to (batch_size, seq_len, num_heads, d_k)
        attention_output = attention_output.transpose(1, 2).contiguous()

        # Then, reshape to (batch_size, seq_len, num_heads * d_k)
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.d_k)

        # Final linear layer
        multihead_output = self.w_o(attention_output)

        return multihead_output
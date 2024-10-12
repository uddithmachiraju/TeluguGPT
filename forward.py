import torch 
from torch import nn 

class FeedForward(nn.Module):
    def __init__(self, dimentions, feedForwardDim, dropoutRate):
        super().__init__()

        self.dropout = nn.Dropout(dropoutRate)
        self.layer_1 = nn.Linear(dimentions, feedForwardDim)
        self.layer_2 = nn.Linear(feedForwardDim, dimentions) 

    def forward(self, input):
        return self.layer_2(
            self.dropout(
                torch.relu(self.layer_1(input)) 
            )
        )
    
class LayerNormalization(nn.Module):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(512)) 
        self.beta = nn.Parameter(torch.zeros(512)) 

    def forward(self, input):
        mean = input.mean(dim = -1, keepdim = True)
        std = input.std(dim = -1, keepdim = True)
        return self.gamma * (input - mean) / (std + self.eps) + self.beta 
    
class AddAndNorm(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) 
        self.layerNorm = LayerNormalization() 

    def forward(self, input, subLayer):
        return input + self.dropout(
            subLayer(self.layerNorm(input))
        )
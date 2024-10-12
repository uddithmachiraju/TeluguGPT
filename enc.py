from torch import nn 
from forward import AddAndNorm, LayerNormalization

class EncoderBlock(nn.Module):
    def __init__(self, MultiHeadAttention, FeedForward, dropout):
        super().__init__() 
        self.multiHeadAttention = MultiHeadAttention
        self.feedForward = FeedForward 
        self.add_norm_1 = AddAndNorm(dropout) 
        self.add_norm_2 = AddAndNorm(dropout) 

    def forward(self, encoderInput, encoderMask):
        encoderInput = self.add_norm_1(
            encoderInput, lambda encoderInput: self.multiHeadAttention(
                encoderInput, encoderInput, encoderInput, encoderMask
            )
        ) 
        encoderInput = self.add_norm_2(encoderInput, self.feedForward)
        return encoderInput 
    
class Encoder(nn.Module):
    def __init__(self, EncoderBlockList: nn.ModuleList):
        super().__init__() 
        self.EncoderBlockList = EncoderBlockList 
        self.layerNorm = LayerNormalization() 

    def forward(self, encoderInput, encoderMask):
        for encoderBlock in self.EncoderBlockList:
            encoderInput = encoderBlock(encoderInput, encoderMask)
        encoderOutput = self.layerNorm(encoderInput)
        return encoderOutput 
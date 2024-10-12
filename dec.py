from torch import nn 
from forward import AddAndNorm, LayerNormalization 

class DecoderBlock(nn.Module):
    def __init__(self, maskedMultiHeadAttention, crossMultiHeadAttention, FeedForward, dropout):
        super().__init__() 
        self.maskedAttention = maskedMultiHeadAttention 
        self.crossAttention = crossMultiHeadAttention 
        self.feedForward = FeedForward 
        self.addNorm_1 = AddAndNorm(dropout) 
        self.addNorm_2 = AddAndNorm(dropout) 
        self.addNorm_3 = AddAndNorm(dropout) 

    def forward(self, decoderInput, encoderOutput, encoderMask, decoderMask):
        decoderInput = self.addNorm_1(
            decoderInput, lambda decoderInput: self.maskedAttention(
                decoderInput, decoderInput, decoderInput, decoderMask
            )
        )

        decoderInput = self.addNorm_2(
            decoderInput, lambda decoderInput: self.crossAttention(
                decoderInput, encoderOutput, encoderOutput, encoderMask
            )
        )

        decoderInput = self.addNorm_3(decoderInput, self.feedForward) 
        return decoderInput 
    
class Decoder(nn.Module):
    def __init__(self, decoderBlockList: nn.ModuleList):
        super().__init__() 
        self.decoderBlockList = decoderBlockList 
        self.layerNorm = LayerNormalization()

    def forward(self, decoderInput, encoderOutput, encoderMask, decoderMask):
        for decoder in self.decoderBlockList:
            decoderInput = decoder(
                decoderInput, encoderOutput, encoderMask, decoderMask
            )
        decoderOutput = self.layerNorm(decoderInput) 
        return decoderOutput 
    
class ProjectionLayer(nn.Module):
    def __init__(self, dimentions, vocabSize):
        super().__init__() 
        self.projectionLayer = nn.Linear(dimentions, vocabSize)

    def forward(self, decoderOutput):
        output = self.projectionLayer(decoderOutput) 
        return output 
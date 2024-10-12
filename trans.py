import torch 
from torch import nn 

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, sourceEmbed, 
                 targetEmbed, sourcePos, targetPos, projectionLayer):
        super().__init__() 

        self.sourceEmbeds = sourceEmbed 
        self.sourcePosEnc = sourcePos 
        self.encoder = encoder 

        self.targetEmbeds = targetEmbed
        self.targetPosEnc = targetPos 
        self.decoder = decoder 

        self.projectionLayer = projectionLayer 

    def encode(self, encoderInput, encodermask):
        encoderInput = self.sourceEmbeds(encoderInput) 
        encoderInput = self.sourcePosEnc(encoderInput) 

        encoderOutput = self.encoder(encoderInput, encodermask) 
        return encoderOutput 
    
    def decode(self, encoderOutput, encoderMask, decoderInput, decoderMask):
        decoderInput = self.targetEmbeds(decoderInput) 
        decoderInput = self.targetPosEnc(decoderInput) 
        decoderOutput = self.decoder(decoderInput, encoderOutput, encoderMask, decoderMask)
        return decoderOutput 
    
    def project(self, decoderOutput):
        return self.projectionLayer(decoderOutput)
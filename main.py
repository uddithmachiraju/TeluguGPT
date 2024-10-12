import os, math, torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader, random_split 
from pathlib import Path 
from datasets import load_dataset 
from tqdm import tqdm 
from tokenizers import Tokenizer
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer 
from tokenizers.pre_tokenizers import Whitespace

from embeds import EmbeddingLayer, PositionalEncoding
from trans import Transformer 
from attn import MultiHeadAttention 
from forward import FeedForward
from enc import EncoderBlock, Encoder 
from dec import DecoderBlock, Decoder, ProjectionLayer

directories = ["./TeluguGPT", "./tokenizer_english", "./tokenizer_telugu"]

# Create directories if they do not exist
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load the datasets 
train_dataset = load_dataset("Helsinki-NLP/opus-100", "en-te", split='train')
validation_dataset = load_dataset("Helsinki-NLP/opus-100", "en-te", split='validation')

# limit the number of data in dataset
raw_train_dataset, rt_to_skip = random_split(train_dataset, [2500, len(train_dataset) - 2500]) 
raw_validation_dataset, vt_to_skip = random_split(validation_dataset, [2500, len(validation_dataset) - 2500]) 

def get_ds_iterator(raw_train_dataset, language): 
    for data in raw_train_dataset:
        yield data['translation'][language] 

tokenizer_english = Tokenizer(BPE(unk_token = "[UNK]")) 
trainer_english = BpeTrainer(min_frequency = 2, special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]) 

tokenizer_english.pre_tokenizer = Whitespace()
tokenizer_english.train_from_iterator(get_ds_iterator(raw_train_dataset, "en"), trainer = trainer_english) 
tokenizer_english.save("./tokenizer_english/tokenizer_en.json") 

tokenizer_telugu = Tokenizer(BPE(unk_token = "[UNK]")) 
trainer_telugu = BpeTrainer(min_frequency = 2, special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]) 

tokenizer_telugu.pre_tokenizer = Whitespace() 
tokenizer_telugu.train_from_iterator(get_ds_iterator(raw_train_dataset, 'te'), trainer = trainer_telugu) 
tokenizer_telugu.save("./tokenizer_telugu/tokenizer_te.json") 

tokenizer_english = Tokenizer.from_file("./tokenizer_english/tokenizer_en.json") 
tokenizer_telugu = Tokenizer.from_file("./tokenizer_telugu/tokenizer_te.json") 

source_vocab_size = tokenizer_english.get_vocab_size() 
target_vocab_size = tokenizer_telugu.get_vocab_size() 

max_seq_length_source = 0 
max_seq_length_target = 0 

for data in raw_train_dataset:
    encoder_ids = tokenizer_english.encode(data['translation']['en']).ids 
    decoder_ids = tokenizer_telugu.encode(data['translation']['te']).ids 
    max_seq_length_source = max(max_seq_length_source, len(encoder_ids))
    max_seq_length_target = max(max_seq_length_target, len(decoder_ids)) 

print(f'Max Sequence Length Source: {max_seq_length_source}') 
print(f'Max Sequence Length Target: {max_seq_length_target}') 

max_seq_length = 200 

def causal_mask(size):
    # Creating a square matrix of dimensions 'size x size' filled with ones
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0

# Prepare dataset and Dataloader
class EncodeDataset(Dataset):
    def __init__(self, raw_dataset, max_seq_length):
        self.raw_dataset = raw_dataset 
        self.max_seq_length = max_seq_length 

    def __len__(self):
        return len(self.raw_dataset) 
    
    def __getitem__(self, index):

        # fetching the single data for the given index value that consist of both english and malay language.
        raw_text = self.raw_dataset[index]

        # separating text by source and target lanaguage which will be later used for encoding.
        source_text = raw_text['translation']['en']
        target_text = raw_text['translation']['te']

        # Encoding source text with with english tokenizer and target text with malay tokenizer
        source_text_encoded = tokenizer_english.encode(source_text).ids
        target_text_encoded = tokenizer_telugu.encode(target_text).ids

        # Convert the CLS, SEP and PAD tokens to their corresponding index id in vocabulary using tokenizer [the id would be same with either tokenizers]
        CLS_ID = torch.tensor([tokenizer_telugu.token_to_id("[CLS]")], dtype=torch.int64)
        SEP_ID = torch.tensor([tokenizer_telugu.token_to_id("[SEP]")], dtype=torch.int64)
        PAD_ID = torch.tensor([tokenizer_telugu.token_to_id("[PAD]")], dtype=torch.int64)

        # To train the model, the sequence lenth of each input should be equal max seq length. Hence additional number of padding will be added to the input sequence if the lenth is not equal to the max seq length.
        num_source_padding = self.max_seq_length - len(source_text_encoded) - 2
        num_target_padding = self.max_seq_length - len(target_text_encoded) - 1

        encoder_padding = torch.tensor([PAD_ID] * num_source_padding, dtype = torch.int64)
        decoder_padding = torch.tensor([PAD_ID] * num_target_padding, dtype = torch.int64)

        # encoder_input has the first token as start of senstence - CLS_ID, followed by source encoding which is then followed by the end of sentence token - SEP.
        # To reach the required max_seq_len, addition PAD token will be added at the end.
        encoder_input = torch.cat([CLS_ID, torch.tensor(source_text_encoded, dtype=torch.int64), SEP_ID, encoder_padding], dim=0)

        # decoder_input has the first token as start of senstence - CLS_ID, followed by target encoding.
        # To reach the required max_seq_len, addition PAD token will be added at the end. There is no end of sentence token - SEP in decoder input.
        decoder_input = torch.cat([CLS_ID, torch.tensor(target_text_encoded, dtype=torch.int64), decoder_padding ], dim=0)

        # target_label is required for the loss calculation during training to compare between the predicted and target label.
        # target_label has the first token as target encoding followed by actual target encoding. There is no start of sentence token - CLS in target label.
        # To reach the required max_seq_len, addition PAD token will be added at the end.
        target_label = torch.cat([torch.tensor(target_text_encoded, dtype=torch.int64),SEP_ID,decoder_padding], dim=0)

        # Since we've added extra padding token with input encoding, we don't want this token to be trained by model.
        # So, we'll use encoder mask to nullify the padding value prior to producing output of self attention in encoder block
        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()

        # We don't want any token to get influence the future token during the decoding stage. Hence, Causal mask is being implemented during masked multihead attention to handle this.
        decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'target_label': target_label,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'source_text': source_text,
            'target_text': target_text
        }

train_ds = EncodeDataset(raw_train_dataset, max_seq_length) 
validation_ds = EncodeDataset(raw_validation_dataset, max_seq_length) 

train_dataloader = DataLoader(train_ds, batch_size = 5, shuffle = True) 
validation_dataloader = DataLoader(validation_ds, batch_size = 1, shuffle = True) 

# print(train_ds.__getitem__(200))

   
def buildModel(sourceVocabSize, targetVocabSize, sourceSeqLength, targetSeqLength, dimentions, 
               numBlocks = 6, numHeads = 8, dropout = 0.1, feedForwardDim = 2048):
    sourceEmbeds = EmbeddingLayer(dimentions, sourceVocabSize) 
    targetEmbeds = EmbeddingLayer(dimentions, targetVocabSize) 

    sourcePos = PositionalEncoding(dimentions, sourceSeqLength, dropout)
    targetPos = PositionalEncoding(dimentions, targetSeqLength, dropout) 

    encoderBlockList = []
    for _ in range(numBlocks):
        multiHeadAttention = MultiHeadAttention(dimentions, numHeads, dropout)
        feedForward = FeedForward(dimentions, feedForwardDim, dropout)
        encoderBlock = EncoderBlock(multiHeadAttention, feedForward, dropout) 
        encoderBlockList.append(encoderBlock) 
    encoder = Encoder(nn.ModuleList(encoderBlockList))  
    
    decoderBlockList = []
    for _ in range(numBlocks):
        maskedMultiHeadAttention = MultiHeadAttention(dimentions, numHeads, dropout)
        crossHeadAttention = MultiHeadAttention(dimentions, numHeads, dropout) 
        feedForward = FeedForward(dimentions, feedForwardDim, dropout)
        decoderBlock = DecoderBlock(maskedMultiHeadAttention, crossHeadAttention, feedForward, dropout) 
        decoderBlockList.append(decoderBlock) 
    decoder = Decoder(nn.ModuleList(decoderBlockList))  

    projectionLayer = ProjectionLayer(dimentions, target_vocab_size)

    model = Transformer(encoder, decoder, sourceEmbeds, targetEmbeds, 
                        sourcePos, targetPos, projectionLayer) 

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    
    return model 

model = buildModel(tokenizer_english.get_vocab_size(), tokenizer_telugu.get_vocab_size(), max_seq_length, max_seq_length, dimentions = 512).to(device = 'cpu') 
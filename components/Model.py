import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, iids, mask, tiids, tmask):
        outputs = self.bert(input_ids=iids, attention_mask=mask)
        hidden_states = outputs.last_hidden_state
        tgt = self.bert(input_ids=tiids, attention_mask=tmask).last_hidden_state
        decoder_output = self.decoder(tgt=tgt, memory=hidden_states)
        
        sequence_output = decoder_output[:, -1, :]
        
        logits = self.classifier(sequence_output)
        
        probs = self.sigmoid(logits)
        
        return probs
    

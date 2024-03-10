import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json

def map_one_to_many_to_one_to_one(data_path):
    def one_to_one(text):
        return list(map(lambda x: (x['text'], text['text'], x['sentiment']), text['targets']))
        
    with open(data_path) as fp:
        data = json.load(fp)
        
    return [item for sublist in map(one_to_one, data) for item in sublist]

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = map_one_to_many_to_one_to_one(data_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenise(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        iids, mask = self.tokenise(sample[0])
        tiids, tmask = self.tokenise(sample[1])
        label = torch.tensor(1.0 if sample[-1] == 'positive' else 0.0)
        
        return ((iids, mask, tiids, tmask), label)

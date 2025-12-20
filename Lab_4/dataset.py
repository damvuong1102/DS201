import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import Config
from vocab import Vocab

class PhoMTDataset(Dataset):
    def __init__(self, json_path, vocab, src_key, tgt_key):
        self.vocab = vocab
        self.data = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            # Lưu lại token dạng số luôn để train cho nhanh
            src_text = item[src_key]
            tgt_text = item[tgt_key]
            
            src_indices = self.vocab.encode(src_text)
            tgt_indices = self.vocab.encode(tgt_text)
            
            self.data.append((torch.tensor(src_indices), torch.tensor(tgt_indices)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, padding_value=Config.PAD_IDX, batch_first=True)
    tgt_pad = pad_sequence(tgt_batch, padding_value=Config.PAD_IDX, batch_first=True)
    return src_pad, tgt_pad

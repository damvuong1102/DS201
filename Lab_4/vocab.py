import json
from collections import Counter
from config import Config

class Vocab:
    def __init__(self):
        self.stoi = {
            "<pad>": Config.PAD_IDX,
            "<bos>": Config.BOS_IDX,
            "<eos>": Config.EOS_IDX,
            "<unk>": Config.UNK_IDX,
        }
        self.itos = {v: k for k, v in self.stoi.items()}
        self.total_src_tokens = 4
        self.total_tgt_tokens = 4

    def build_vocab(self, json_path, src_key, tgt_key, min_freq=2):
        print(f"Đang xây dựng Vocab từ {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        src_counter = Counter()
        tgt_counter = Counter()

        for item in data:
            # Tokenize đơn giản bằng split(), có thể dùng spacy nếu muốn xịn hơn
            src_tokens = item[src_key].lower().split()
            tgt_tokens = item[tgt_key].lower().split()
            
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        # Add Source Tokens
        for word, freq in src_counter.items():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.stoi)
        
        self.total_src_tokens = len(self.stoi)
        
        # Add Target Tokens (Lưu ý: Thường người ta tách 2 vocab riêng, 
        # nhưng để đơn giản cho lab này ta gộp chung hoặc chỉ add thêm từ mới)
        for word, freq in tgt_counter.items():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.stoi)
                
        self.total_tgt_tokens = len(self.stoi) # Tổng size vocab chung
        self.itos = {v: k for k, v in self.stoi.items()}
        print(f"Vocab size: {len(self.stoi)}")

    def encode(self, text):
        # Text string -> List of Indices
        tokens = text.lower().split()
        return [Config.BOS_IDX] + [self.stoi.get(token, Config.UNK_IDX) for token in tokens] + [Config.EOS_IDX]

    def decode(self, indices):
        # List of Indices -> Text String
        tokens = []
        for idx in indices:
            if idx == Config.EOS_IDX: break
            if idx in [Config.BOS_IDX, Config.PAD_IDX]: continue
            tokens.append(self.itos.get(idx, "<unk>"))
        return " ".join(tokens)

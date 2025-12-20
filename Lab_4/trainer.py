import torch
from config import Config
from torchmetrics.text.rouge import ROUGEScore

def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, tgt in iterator:
        src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt)
        
        # Output: [bs, trg_len, vocab_size], Tgt: [bs, trg_len]
        # Flatten để tính loss, bỏ qua cột đầu tiên của output (tương ứng với dự đoán từ đầu vào <bos>)
        output_dim = output.shape[-1]
        loss = criterion(output.reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def calculate_rouge(model, iterator, vocab):
    model.eval()
    rouge = ROUGEScore()
    preds_text = []
    targets_text = []
    
    with torch.no_grad():
        for src, tgt in iterator:
            src = src.to(Config.DEVICE)
            # Gọi hàm predict (lưu ý bạn cần implement predict trong models.py)
            # Ở đây dùng logic giả định nếu chưa implement
            pred_indices = model.predict(src) 
            if pred_indices is None: continue 
            
            for i in range(src.size(0)):
                p = vocab.decode(pred_indices[i].tolist())
                t = vocab.decode(tgt[i].tolist())
                preds_text.append(p)
                targets_text.append(t)
                
    if not preds_text: return 0.0
    return rouge(preds_text, targets_text)

import json
import numpy as np
import os
from sklearn.metrics import f1_score, classification_report

class NERDataLoader:
    def __init__(self, train_path, test_path, max_len=50):
        self.max_len = max_len
        
        # Load raw data
        self.train_data = self.load_json(train_path)
        self.test_data = self.load_json(test_path)
        
        # Từ điển
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"<PAD>": 0} # Tag 0 dành cho padding
        self.idx2tag = {0: "<PAD>"}
        
        self.build_vocab_and_tags()
        
    def load_json(self, path):
        if not os.path.exists(path):
            print(f"File {path} not found. Creating dummy data.")
            return []
            
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            # --- PHẦN SỬA ĐỔI: Đọc từng dòng (JSON Lines) ---
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    # Parse từng dòng thành 1 dict
                    obj = json.loads(line)
                    # PhoNER thường có format: {"words": [...], "tags": [...]}
                    if 'words' in obj and 'tags' in obj:
                        samples.append(obj)
                except json.JSONDecodeError:
                    continue 
            # -----------------------------------------------
        return samples

    def build_vocab_and_tags(self):
        # Xây dựng vocab từ tập train
        print(f"Building vocab from {len(self.train_data)} samples...")
        for sample in self.train_data:
            for word in sample['words']:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
            for tag in sample['tags']:
                if tag not in self.tag2idx:
                    new_id = len(self.tag2idx)
                    self.tag2idx[tag] = new_id
                    self.idx2tag[new_id] = tag
        
        self.num_words = len(self.word2idx)
        self.num_tags = len(self.tag2idx)
        print(f"[Info] Vocab Size: {self.num_words}, Num Tags: {self.num_tags}")
        print(f"Tags mapping: {self.tag2idx}")

    def process_batch(self, data):
        X, Y, Masks = [], [], []
        
        for sample in data:
            words = sample['words']
            tags = sample['tags']
            
            # Tokenize & Encode
            x_seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
            y_seq = [self.tag2idx.get(t, 0) for t in tags] # 0 là PAD tag
            
            # Padding / Truncating
            cur_len = len(x_seq)
            if cur_len < self.max_len:
                pad_len = self.max_len - cur_len
                x_seq += [0] * pad_len
                y_seq += [0] * pad_len
                mask = [1] * cur_len + [0] * pad_len 
            else:
                x_seq = x_seq[:self.max_len]
                y_seq = y_seq[:self.max_len]
                mask = [1] * self.max_len
                
            X.append(x_seq)
            Y.append(y_seq)
            Masks.append(mask)
            
        return np.array(X), np.array(Y), np.array(Masks)
# --- 1. Các hàm Activation ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(out):
    return out * (1 - out)

def tanh(x):
    return np.tanh(x)

def d_tanh(out):
    return 1 - out**2

def softmax_3d(x):
    # x shape: (Batch, Time, Classes)
    # Trừ max để ổn định
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# --- 2. Optimizer (Adam) ---
class AdamOptimizer:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.params:
            if k in grads:
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k]**2)
                m_hat = self.m[k] / (1 - self.beta1**self.t)
                v_hat = self.v[k] / (1 - self.beta2**self.t)
                self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# --- 3. Uni-directional LSTM Layer (Cơ sở cho BiLSTM) ---
class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, return_sequences=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences # True cho bài toán NER
        
        limit = np.sqrt(6 / (input_dim + hidden_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim + hidden_dim, 4 * hidden_dim))
        self.b = np.zeros((4 * hidden_dim,))
        
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

    def forward(self, x):
        # x: (Batch, Time, Input_dim)
        B, T, D = x.shape
        H = self.hidden_dim
        
        # States
        h = np.zeros((B, T, H))
        c = np.zeros((B, T, H))
        
        h_t = np.zeros((B, H))
        c_t = np.zeros((B, H))
        
        cache = []
        
        for t in range(T):
            x_t = x[:, t, :]
            concat = np.hstack((h_t, x_t)) # (B, H+D)
            
            gates = np.dot(concat, self.W) + self.b
            
            # Slice gates
            f = sigmoid(gates[:, :H])
            i = sigmoid(gates[:, H:2*H])
            o = sigmoid(gates[:, 2*H:3*H])
            g = tanh(gates[:, 3*H:])
            
            c_t = f * c_t + i * g
            h_t = o * tanh(c_t)
            
            h[:, t, :] = h_t
            c[:, t, :] = c_t
            
            cache.append((concat, f, i, o, g, c_t, tanh(c_t)))
            
        self.cache = (x, cache)
        return h # (B, T, H)

    def backward(self, dh):
        # dh: Gradient từ lớp trên truyền xuống (Batch, Time, Hidden)
        # Trong NER, dh có giá trị ở mọi time step T
        x, cache_steps = self.cache
        B, T, D = x.shape
        H = self.hidden_dim
        
        dx = np.zeros_like(x)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        
        # BPTT: Duyệt ngược từ T-1 về 0
        for t in reversed(range(T)):
            concat, f, i, o, g, c_t, tanh_c = cache_steps[t]
            
            # Gradient tổng hợp tại bước t: từ lớp trên (dh[:, t]) + từ bước tương lai (dh_next)
            dh_t = dh[:, t, :] + dh_next
            
            # Output gate
            do = dh_t * tanh_c
            do_raw = do * d_sigmoid(o)
            
            # Cell state
            dc = dc_next + (dh_t * o * d_tanh(tanh_c))
            
            # Cell input (g)
            dg = dc * i
            dg_raw = dg * d_tanh(g)
            
            # Input gate
            di = dc * g
            di_raw = di * d_sigmoid(i)
            
            # Forget gate
            if t > 0:
                c_prev = cache_steps[t-1][5]
            else:
                c_prev = np.zeros((B, H))
            df = dc * c_prev
            df_raw = df * d_sigmoid(f)
            
            # Gộp gradient gates
            d_gates = np.hstack((df_raw, di_raw, do_raw, dg_raw))
            
            # Gradients cho W, b
            dW += np.dot(concat.T, d_gates)
            db += np.sum(d_gates, axis=0)
            
            # Gradient truyền về input và hidden trước
            d_concat = np.dot(d_gates, self.W.T)
            dh_next = d_concat[:, :H]
            dx[:, t, :] = d_concat[:, H:]
            
            # Gradient cho c_prev
            dc_next = dc * f
            
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # Clip gradients
        np.clip(dW, -5, 5, out=dW)
        
        return dx

# --- 4. Bidirectional LSTM Layer ---
class BiLSTMLayer:
    def __init__(self, input_dim, hidden_dim):
        # Forward LSTM
        self.fw_lstm = LSTMLayer(input_dim, hidden_dim)
        # Backward LSTM
        self.bw_lstm = LSTMLayer(input_dim, hidden_dim)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2 # Concat

    def forward(self, x):
        # x: (B, T, D)
        
        # 1. Forward Pass
        out_fw = self.fw_lstm.forward(x)
        
        # 2. Backward Pass: Đảo ngược chuỗi input theo chiều thời gian (axis=1)
        x_rev = np.flip(x, axis=1)
        out_bw_rev = self.bw_lstm.forward(x_rev)
        
        # Đảo ngược lại output backward để khớp với time step gốc
        out_bw = np.flip(out_bw_rev, axis=1)
        
        # 3. Concatenate
        # out: (B, T, 2*H)
        out = np.concatenate((out_fw, out_bw), axis=-1)
        
        return out

    def backward(self, dout):
        # dout: (B, T, 2*H)
        H = self.hidden_dim
        
        # Tách gradient cho fw và bw
        dout_fw = dout[:, :, :H]
        dout_bw = dout[:, :, H:]
        
        # 1. Backprop Forward LSTM
        dx_fw = self.fw_lstm.backward(dout_fw)
        
        # 2. Backprop Backward LSTM
        # Cần đảo ngược gradient bw trước khi truyền vào (vì lúc forward ta đã đảo input)
        dout_bw_rev = np.flip(dout_bw, axis=1)
        dx_bw_rev = self.bw_lstm.backward(dout_bw_rev)
        dx_bw = np.flip(dx_bw_rev, axis=1)
        
        # 3. Tổng hợp gradient cho input x
        dx = dx_fw + dx_bw
        return dx
    
class BiLSTMNERModel:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_tags):
        # 1. Embeddings
        self.emb_matrix = np.random.randn(vocab_size, embed_dim) * 0.01
        
        # 2. Stacked BiLSTMs
        self.layers = []
        curr_dim = embed_dim
        for i in range(num_layers):
            self.layers.append(BiLSTMLayer(curr_dim, hidden_dim))
            curr_dim = hidden_dim * 2 
            
        # 3. Output Layer
        self.fc_W = np.random.randn(curr_dim, num_tags) * 0.01
        self.fc_b = np.zeros((num_tags,))
        
        # Init Params (Lưu trọng số)
        self.params = {'emb': self.emb_matrix, 'fc_W': self.fc_W, 'fc_b': self.fc_b}
        for i, layer in enumerate(self.layers):
            self.params[f'l{i}_fw_W'] = layer.fw_lstm.W
            self.params[f'l{i}_fw_b'] = layer.fw_lstm.b
            self.params[f'l{i}_bw_W'] = layer.bw_lstm.W
            self.params[f'l{i}_bw_b'] = layer.bw_lstm.b

    def forward(self, x_idx):
        self.x_emb = self.emb_matrix[x_idx]
        out = self.x_emb
        for layer in self.layers:
            out = layer.forward(out)
        
        self.h_last = out
        logits = np.dot(out, self.fc_W) + self.fc_b
        return softmax_3d(logits)

    def backward(self, y_true_onehot, probs, mask):
        B, T, K = probs.shape
        
        # 1. Gradient Loss
        mask_exp = mask[:, :, np.newaxis]
        d_logits = (probs - y_true_onehot) * mask_exp / B 
        
        # 2. Gradient FC
        h_flat = self.h_last.reshape(-1, self.h_last.shape[-1])
        d_logits_flat = d_logits.reshape(-1, K)
        
        d_fc_W = np.dot(h_flat.T, d_logits_flat)
        d_fc_b = np.sum(d_logits_flat, axis=0)
        
        # Dictionary chứa Gradient (KHÔNG ĐƯỢC GHI ĐÈ SELF.PARAMS)
        grads = {
            'fc_W': d_fc_W,
            'fc_b': d_fc_b,
            'emb': np.zeros_like(self.emb_matrix) # Placeholder cho emb
        }
        
        # 3. Backprop BiLSTMs
        d_h = np.dot(d_logits, self.fc_W.T)
        
        for i in reversed(range(len(self.layers))):
            d_h = self.layers[i].backward(d_h)
            
            # Lưu gradient vào dict grads
            grads[f'l{i}_fw_W'] = self.layers[i].fw_lstm.grads['W']
            grads[f'l{i}_fw_b'] = self.layers[i].fw_lstm.grads['b']
            grads[f'l{i}_bw_W'] = self.layers[i].bw_lstm.grads['W']
            grads[f'l{i}_bw_b'] = self.layers[i].bw_lstm.grads['b']
            
        return grads
def compute_loss_acc(probs, y_idx, mask):
    """
    Tính Cross Entropy Loss và Accuracy, có xét đến mask (bỏ qua padding)
    probs: (Batch, Time, Num_Tags) - Output từ model
    y_idx: (Batch, Time) - Nhãn thật dạng index
    mask: (Batch, Time) - Mask (1 là token thật, 0 là padding)
    """
    B, T, K = probs.shape
    
    # Flatten để tính toán dễ dàng hơn
    flat_probs = probs.reshape(-1, K)
    flat_y = y_idx.reshape(-1)
    flat_mask = mask.reshape(-1)
    
    # Cross Entropy (chỉ tính trên các token thật, mask=1)
    # Thêm 1e-9 để tránh log(0)
    # Chọn xác suất của đúng class thật (flat_y)
    correct_class_probs = flat_probs[np.arange(len(flat_y)), flat_y]
    correct_log_probs = -np.log(correct_class_probs + 1e-9)
    
    # Tổng loss chỉ trên các phần tử mask=1
    loss = np.sum(correct_log_probs * flat_mask) / np.sum(flat_mask)
    
    # Accuracy
    preds = np.argmax(flat_probs, axis=1)
    correct = (preds == flat_y) * flat_mask # Chỉ tính đúng sai trên token thật
    acc = np.sum(correct) / np.sum(flat_mask)
    
    return loss, acc, preds, flat_y, flat_mask
# --- Config ---
TRAIN_PATH = 'train_syllable.json'
TEST_PATH = 'test_syllable.json' # Dùng làm Dev/Test
EMBED_DIM = 100
HIDDEN_DIM = 64     
NUM_LAYERS = 2     
BATCH_SIZE = 16     
EPOCHS = 5
LEARNING_RATE = 0.01

# 1. Load Data
loader = NERDataLoader(TRAIN_PATH, TEST_PATH)
X_train, y_train, mask_train = loader.process_batch(loader.train_data)

# 2. Init Model
model = BiLSTMNERModel(vocab_size=loader.num_words, 
                       embed_dim=EMBED_DIM, 
                       hidden_dim=HIDDEN_DIM, 
                       num_layers=NUM_LAYERS, 
                       num_tags=loader.num_tags)

optimizer = AdamOptimizer(model.params, lr=LEARNING_RATE)

print(f"Retrying Training (Layers={NUM_LAYERS}, LR={LEARNING_RATE})...")

for epoch in range(EPOCHS):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for i in range(0, len(X_train), BATCH_SIZE):
        # Batching
        batch_idx = indices[i:i+BATCH_SIZE]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        mask_batch = mask_train[batch_idx]
        
        # One-hot Y
        B, T = y_batch.shape
        y_onehot = np.zeros((B, T, loader.num_tags))
        for b in range(B):
            for t in range(T):
                y_onehot[b, t, y_batch[b, t]] = 1
        
        # Forward & Loss
        probs = model.forward(X_batch)
        loss, acc, _, _, _ = compute_loss_acc(probs, y_batch, mask_batch)
        total_loss += loss
        
        # Backward & Update
        grads = model.backward(y_onehot, probs, mask_batch)
        optimizer.step(grads) # Update weights trong model.params
        
        # Sync Params: Copy trọng số mới từ model.params vào các layers
        model.fc_W = model.params['fc_W']
        model.fc_b = model.params['fc_b']
        for idx, layer in enumerate(model.layers):
            layer.fw_lstm.W = model.params[f'l{idx}_fw_W']
            layer.fw_lstm.b = model.params[f'l{idx}_fw_b']
            layer.bw_lstm.W = model.params[f'l{idx}_bw_W']
            layer.bw_lstm.b = model.params[f'l{idx}_bw_b']
            
        # Metrics Accumulation
        preds_batch = np.argmax(probs, axis=-1)
        valid_mask = mask_batch == 1
        all_preds.extend(preds_batch[valid_mask])
        all_labels.extend(y_batch[valid_mask])

    # End Epoch Report
    avg_loss = total_loss / (len(X_train) // BATCH_SIZE)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {train_f1:.4f}")
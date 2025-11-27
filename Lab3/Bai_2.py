import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# --- 1. ACTIVATION FUNCTIONS & DERIVATIVES ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(sigmoid_out):
    return sigmoid_out * (1 - sigmoid_out)

def tanh(x):
    return np.tanh(x)

def d_tanh(tanh_out):
    return 1 - tanh_out ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# --- 2. ADAM OPTIMIZER (REUSED) ---
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for key in self.params:
            if key not in grads: continue
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# --- 3. GRU CELL ---
class GRUCell:
    """
    Gated Recurrent Unit (GRU).
    Formulas:
    z = sigmoid(Wz . [h_prev, x] + bz)  (Update gate)
    r = sigmoid(Wr . [h_prev, x] + br)  (Reset gate)
    h_hat = tanh(Wh . [r * h_prev, x] + bh) (Candidate hidden)
    h = (1 - z) * h_prev + z * h_hat
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        limit = np.sqrt(6 / (input_dim + hidden_dim))
        
        # 1. Weights cho Gates (Update z, Reset r) - Gộp lại để tính nhanh
        # Shape: (input + hidden, 2 * hidden)
        self.W_gates = np.random.uniform(-limit, limit, (input_dim + hidden_dim, 2 * hidden_dim))
        self.b_gates = np.zeros((2 * hidden_dim,))
        
        # 2. Weights cho Candidate Hidden State (h_hat)
        # Tách riêng W_x (cho input) và W_h (cho hidden) để áp dụng reset gate r
        self.W_x_h = np.random.uniform(-limit, limit, (input_dim, hidden_dim))
        self.W_h_h = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim))
        self.b_h = np.zeros((hidden_dim,))
        
        self.params = {
            'W_gates': self.W_gates, 'b_gates': self.b_gates,
            'W_x_h': self.W_x_h, 'W_h_h': self.W_h_h, 'b_h': self.b_h
        }
        self.grads = {k: None for k in self.params}
        self.cache = None

    def forward(self, x, h_prev):
        """
        x: (Batch, Time, Input_dim)
        h_prev: (Batch, Hidden_dim)
        """
        batch_size, time_steps, _ = x.shape
        hs = np.zeros((batch_size, time_steps, self.hidden_dim))
        cache_steps = []
        h_t = h_prev
        
        for t in range(time_steps):
            x_t = x[:, t, :] # (Batch, Input)
            
            # 1. Tính Z và R gates
            concat_gates = np.hstack((h_t, x_t))
            gates_out = np.dot(concat_gates, self.W_gates) + self.b_gates
            
            # Split: z (Update), r (Reset)
            z = sigmoid(gates_out[:, :self.hidden_dim])
            r = sigmoid(gates_out[:, self.hidden_dim:])
            
            # 2. Tính Candidate Hidden State (h_hat)
            # h_hat = tanh(W_x*x + W_h*(r * h_prev) + b)
            r_h_prev = r * h_t # Reset gate áp dụng lên hidden state cũ
            
            # Tính tuyến tính cho h_hat
            h_hat_raw = np.dot(x_t, self.W_x_h) + np.dot(r_h_prev, self.W_h_h) + self.b_h
            h_hat = tanh(h_hat_raw)
            
            # 3. Final Hidden State
            # h_t = (1 - z) * h_prev + z * h_hat
            h_next = (1 - z) * h_t + z * h_hat
            
            hs[:, t, :] = h_next
            
            # Lưu cache
            cache_steps.append((x_t, h_t, z, r, r_h_prev, h_hat))
            
            h_t = h_next # Cập nhật cho bước sau
            
        self.cache = (x, cache_steps)
        return hs, h_t

    def backward(self, dh_next_layer):
        """
        BPTT cho GRU.
        dh_next_layer: Gradient từ lớp trên (Batch, Time, Hidden)
        """
        x, cache_steps = self.cache
        batch_size, time_steps, _ = x.shape
        
        # Init gradients
        dx = np.zeros_like(x)
        dW_gates = np.zeros_like(self.W_gates)
        db_gates = np.zeros_like(self.b_gates)
        dW_x_h = np.zeros_like(self.W_x_h)
        dW_h_h = np.zeros_like(self.W_h_h)
        db_h = np.zeros_like(self.b_h)
        
        dh_next_t = np.zeros((batch_size, self.hidden_dim))
        
        for t in reversed(range(time_steps)):
            x_t, h_prev, z, r, r_h_prev, h_hat = cache_steps[t]
            
            # Gradient tổng tại bước t
            dh = dh_next_layer[:, t, :] + dh_next_t
            
            # 1. Gradient qua công thức h_t = (1 - z) * h_prev + z * h_hat
            # dL/dh_hat
            dh_hat = dh * z
            dh_hat_raw = dh_hat * d_tanh(h_hat)
            
            # dL/dz
            dz = dh * (h_hat - h_prev)
            dz_raw = dz * d_sigmoid(z)
            
            # dL/dh_prev (phần 1: từ cổng update trực tiếp)
            dh_prev_1 = dh * (1 - z)
            
            # 2. Gradient qua Candidate Weights (W_x_h, W_h_h)
            dW_x_h += np.dot(x_t.T, dh_hat_raw)
            dW_h_h += np.dot(r_h_prev.T, dh_hat_raw)
            db_h += np.sum(dh_hat_raw, axis=0)
            
            # Gradient truyền về input x (phần từ candidate)
            dx_t_1 = np.dot(dh_hat_raw, self.W_x_h.T)
            
            # Gradient truyền về r_h_prev
            dr_h_prev = np.dot(dh_hat_raw, self.W_h_h.T)
            
            # 3. Gradient qua Reset Gate (r)
            # r_h_prev = r * h_prev
            dr = dr_h_prev * h_prev
            dr_raw = dr * d_sigmoid(r)
            
            # dL/dh_prev (phần 2: từ candidate qua reset gate)
            dh_prev_2 = dr_h_prev * r
            
            # 4. Gradient qua Gates Weights (W_gates)
            # gộp gradient z và r
            d_gates_raw = np.hstack((dz_raw, dr_raw))
            concat_gates = np.hstack((h_prev, x_t))
            
            dW_gates += np.dot(concat_gates.T, d_gates_raw)
            db_gates += np.sum(d_gates_raw, axis=0)
            
            # Gradient truyền về input/hidden từ gates
            d_concat = np.dot(d_gates_raw, self.W_gates.T)
            
            dh_prev_3 = d_concat[:, :self.hidden_dim] # Phần hidden
            dx_t_2 = d_concat[:, self.hidden_dim:]   # Phần input
            
            # 5. Tổng hợp Gradient
            dx[:, t, :] = dx_t_1 + dx_t_2
            dh_next_t = dh_prev_1 + dh_prev_2 + dh_prev_3
            
        # Lưu vào dictionary
        self.grads['W_gates'] = dW_gates
        self.grads['b_gates'] = db_gates
        self.grads['W_x_h'] = dW_x_h
        self.grads['W_h_h'] = dW_h_h
        self.grads['b_h'] = db_h
        
        # Clip Gradients
        for k in self.grads:
            np.clip(self.grads[k], -5, 5, out=self.grads[k])
            
        return dx

# --- 4. DEEP GRU CLASSIFIER (STACKED) ---
class DeepGRUClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        self.layers = []
        self.params = {}
        
        # Stacked GRU Layers
        curr_input_dim = embed_dim
        for idx in range(num_layers):
            layer_name = f'gru_{idx}'
            layer = GRUCell(curr_input_dim, hidden_dim)
            self.layers.append(layer)
            # Mapping params
            for k, v in layer.params.items():
                self.params[f'{layer_name}_{k}'] = v
            curr_input_dim = hidden_dim
            
        # Fully Connected Layer
        self.fc_W = np.random.randn(hidden_dim, num_classes) * 0.01
        self.fc_b = np.zeros((num_classes,))
        self.params['fc_W'] = self.fc_W
        self.params['fc_b'] = self.fc_b
        
    def forward(self, x):
        batch_size = x.shape[0]
        layer_input = x
        
        for layer in self.layers:
            h0 = np.zeros((batch_size, layer.hidden_dim))
            # GRU chỉ trả về hs (sequence) và h_last
            layer_input, _ = layer.forward(layer_input, h0)
            
        # Lấy output bước cuối cùng của lớp cuối
        h_last = layer_input[:, -1, :]
        
        logits = np.dot(h_last, self.fc_W) + self.fc_b
        self.cache_fc = (h_last, logits)
        return softmax(logits)

    def backward(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        
        # Gradient Loss
        d_logits = (y_pred - y_true) / batch_size
        h_last, _ = self.cache_fc
        
        # FC Gradients
        dfc_W = np.dot(h_last.T, d_logits)
        dfc_b = np.sum(d_logits, axis=0)
        
        # Gradient truyền xuống GRU cuối
        dh_last = np.dot(d_logits, self.fc_W.T)
        
        time_steps = self.layers[0].cache[0].shape[1]
        hidden_dim = self.layers[0].hidden_dim
        
        # Chuẩn bị gradient sequence (Many-to-One: chỉ bước cuối có grad từ loss)
        dh_seq = np.zeros((batch_size, time_steps, hidden_dim))
        dh_seq[:, -1, :] = dh_last
        
        # Backprop qua các lớp GRU
        current_grad = dh_seq
        for i in reversed(range(len(self.layers))):
            current_grad = self.layers[i].backward(current_grad)
            
            # Cập nhật params dict từ layer grads
            layer_name = f'gru_{i}'
            for k in self.layers[i].grads:
                self.params[f'{layer_name}_{k}'] = self.layers[i].grads[k]

        self.params['fc_W'] = dfc_W
        self.params['fc_b'] = dfc_b
        
        return self.params

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# --- 5. MAIN TRAINING (Demo) ---
# Giả lập dữ liệu
BATCH_SIZE = 32
EMBED_DIM = 100
HIDDEN_DIM = 256
SEQ_LEN = 40
NUM_CLASSES = 3
NUM_LAYERS = 5
EPOCHS = 5

print("Initializing GRU model...")
X_train = np.random.randn(100, SEQ_LEN, EMBED_DIM)
y_train_idx = np.random.randint(0, NUM_CLASSES, 100)
y_train = np.eye(NUM_CLASSES)[y_train_idx]

model = DeepGRUClassifier(None, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES)
optimizer = AdamOptimizer(model.params, lr=0.001)

print(f"Training Stacked GRU ({NUM_LAYERS} layers)...")
for epoch in range(EPOCHS):
    permutation = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]
    
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_shuffled[i:i+BATCH_SIZE]
        y_batch = y_shuffled[i:i+BATCH_SIZE]
        
        # 1. Forward
        probs = model.forward(X_batch)
        
        # 2. Loss
        loss = cross_entropy_loss(probs, y_batch)
        epoch_loss += loss
        
        # 3. Backward
        grads = model.backward(y_batch, probs)
        
        # 4. Update
        optimizer.step(grads)
        
        # Metrics
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(y_batch, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        # Sync params lại vào các layer objects (quan trọng cho vòng lặp sau)
        for idx, layer in enumerate(model.layers):
            layer_name = f'gru_{idx}'
            layer.W_gates = model.params[f'{layer_name}_W_gates']
            layer.b_gates = model.params[f'{layer_name}_b_gates']
            layer.W_x_h = model.params[f'{layer_name}_W_x_h']
            layer.W_h_h = model.params[f'{layer_name}_W_h_h']
            layer.b_h = model.params[f'{layer_name}_b_h']
        model.fc_W = model.params['fc_W']
        model.fc_b = model.params['fc_b']

    avg_loss = epoch_loss / (len(X_train) // BATCH_SIZE)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | F1-Score: {f1:.4f} | Acc: {acc:.4f}")
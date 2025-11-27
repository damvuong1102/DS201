import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# --- 1. CÁC HÀM KÍCH HOẠT & HỖ TRỢ ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(sigmoid_out):
    # Đạo hàm sigmoid: f'(x) = f(x) * (1 - f(x))
    return sigmoid_out * (1 - sigmoid_out)

def tanh(x):
    return np.tanh(x)

def d_tanh(tanh_out):
    # Đạo hàm tanh: f'(x) = 1 - f(x)^2
    return 1 - tanh_out ** 2

def softmax(x):
    # Trừ max để ổn định số học (tránh exp quá lớn gây overflow)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# --- 2. BỘ TỐI ƯU ADAM (MANUAL IMPLEMENTATION) ---
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()} # Moment bậc 1
        self.v = {k: np.zeros_like(v) for k, v in params.items()} # Moment bậc 2
        self.t = 0 # Time step

    def step(self, grads):
        self.t += 1
        for key in self.params:
            if key not in grads: continue
            
            # 1. Cập nhật moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # 2. Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 3. Cập nhật tham số
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# --- 3. LỚP LSTM ĐƠN LẺ (SINGLE LSTM LAYER) ---
class LSTMLayer:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo Xavier (Glorot) Initialization
        limit = np.sqrt(6 / (input_dim + hidden_dim))
        
        # Gộp trọng số 4 cổng (Forget, Input, Cell, Output) vào 1 ma trận lớn để tính nhanh
        # Shape: (input_dim + hidden_dim, 4 * hidden_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim + hidden_dim, 4 * hidden_dim))
        self.b = np.zeros((4 * hidden_dim,))
        
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        """
        x: (Batch, Time, Input_dim)
        """
        batch_size, time_steps, _ = x.shape
        
        # Lưu trữ trạng thái qua các bước thời gian
        hs = np.zeros((batch_size, time_steps, self.hidden_dim))
        cs = np.zeros((batch_size, time_steps, self.hidden_dim))
        
        # Cache để dùng cho backward
        cache_steps = [] 
        
        h_t = h_prev
        c_t = c_prev
        
        for t in range(time_steps):
            x_t = x[:, t, :] # (Batch, Input_dim)
            
            # Nối input và hidden state cũ: (Batch, Input_dim + Hidden_dim)
            concat = np.hstack((h_t, x_t))
            
            # Tính toán tuyến tính gộp: (Batch, 4 * Hidden_dim)
            gates_raw = np.dot(concat, self.W) + self.b
            
            # Tách ra 4 phần: i, f, o, g (g là candidate cell)
            # Mỗi phần có shape (Batch, Hidden_dim)
            f_raw = gates_raw[:, :self.hidden_dim]
            i_raw = gates_raw[:, self.hidden_dim : 2*self.hidden_dim]
            g_raw = gates_raw[:, 2*self.hidden_dim : 3*self.hidden_dim]
            o_raw = gates_raw[:, 3*self.hidden_dim :]
            
            # Kích hoạt
            f = sigmoid(f_raw)
            i = sigmoid(i_raw)
            g = tanh(g_raw)
            o = sigmoid(o_raw)
            
            # Cập nhật Cell state & Hidden state
            c_t = f * c_t + i * g
            h_t = o * tanh(c_t)
            
            hs[:, t, :] = h_t
            cs[:, t, :] = c_t
            
            # Lưu cache tại bước t
            cache_steps.append((concat, f, i, g, o, c_t, tanh(c_t)))
            
        self.cache = (x, cache_steps)
        return hs, h_t, c_t

    def backward(self, dh_next_layer):
        """
        dh_next_layer: Gradient từ lớp phía trên (hoặc loss) truyền xuống
        Shape: (Batch, Time, Hidden_dim)
        """
        x, cache_steps = self.cache
        batch_size, time_steps, _ = x.shape
        
        # Khởi tạo gradient
        dx = np.zeros_like(x)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        
        dh_next_t = np.zeros((batch_size, self.hidden_dim))
        dc_next_t = np.zeros((batch_size, self.hidden_dim))
        
        # Backpropagation Through Time (BPTT) - Đi ngược từ T về 0
        for t in reversed(range(time_steps)):
            concat, f, i, g, o, c_t, tanh_c_t = cache_steps[t]
            
            # Gradient tổng hợp tại bước t: từ lớp trên + từ bước t+1
            dh = dh_next_layer[:, t, :] + dh_next_t
            
            # 1. Gradient qua Output gate
            do = dh * tanh_c_t
            do_raw = do * d_sigmoid(o)
            
            # 2. Gradient qua Cell state
            dc = dc_next_t + (dh * o * d_tanh(tanh_c_t))
            
            # 3. Gradient qua Candidate cell (g)
            dg = dc * i
            dg_raw = dg * d_tanh(g)
            
            # 4. Gradient qua Input gate
            di = dc * g
            di_raw = di * d_sigmoid(i)
            
            # 5. Gradient qua Forget gate (cần c_prev)
            if t > 0:
                c_prev = cache_steps[t-1][5]
            else:
                c_prev = np.zeros_like(c_t)
            
            df = dc * c_prev
            df_raw = df * d_sigmoid(f)
            
            # Gộp gradient 4 cổng: (Batch, 4 * Hidden_dim)
            d_gates = np.hstack((df_raw, di_raw, dg_raw, do_raw))
            
            # Tính gradient cho Weights và Bias
            # dW += concat.T * d_gates
            dW += np.dot(concat.T, d_gates)
            db += np.sum(d_gates, axis=0)
            
            # Tính gradient cho input (để truyền về bước t-1 và lớp dưới)
            d_concat = np.dot(d_gates, self.W.T)
            
            # Tách gradient: phần cho hidden state trước (dh_prev) và phần cho input x (dx)
            dh_next_t = d_concat[:, :self.hidden_dim]
            dx[:, t, :] = d_concat[:, self.hidden_dim:]
            
            # Gradient cho c_prev
            dc_next_t = dc * f
            
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # Clip gradients để tránh bùng nổ (Exploding Gradients)
        np.clip(self.grads['W'], -5, 5, out=self.grads['W'])
        np.clip(self.grads['b'], -5, 5, out=self.grads['b'])
        
        return dx

# --- 4. MÔ HÌNH DEEP LSTM (STACKED) ---
class DeepLSTMClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        self.layers = []
        self.params = {}
        
        # 1. Embedding Layer (Giả lập, thực tế dùng pre-trained từ bước trước)
        # Trong bài tập này, ta coi input x đã là embeddings
        
        # 2. Stacked LSTM Layers
        curr_input_dim = embed_dim
        for idx in range(num_layers):
            layer_name = f'lstm_{idx}'
            layer = LSTMLayer(curr_input_dim, hidden_dim)
            self.layers.append(layer)
            self.params[f'{layer_name}_W'] = layer.W
            self.params[f'{layer_name}_b'] = layer.b
            curr_input_dim = hidden_dim # Input của lớp sau là hidden của lớp trước
            
        # 3. Fully Connected (Output) Layer
        self.fc_W = np.random.randn(hidden_dim, num_classes) * 0.01
        self.fc_b = np.zeros((num_classes,))
        self.params['fc_W'] = self.fc_W
        self.params['fc_b'] = self.fc_b
        
    def forward(self, x):
        """
        x: (Batch, Time, Embed_dim)
        """
        batch_size = x.shape[0]
        layer_input = x
        
        # Truyền qua 5 lớp LSTM
        for layer in self.layers:
            h0 = np.zeros((batch_size, layer.hidden_dim))
            c0 = np.zeros((batch_size, layer.hidden_dim))
            # Output của lớp này là Input của lớp kế tiếp
            layer_input, _, _ = layer.forward(layer_input, h0, c0)
            
        # Lấy output tại bước thời gian cuối cùng (Many-to-One)
        # layer_input lúc này là hs của lớp cuối cùng: (Batch, Time, Hidden)
        h_last = layer_input[:, -1, :] 
        
        # Lớp Fully Connected
        logits = np.dot(h_last, self.fc_W) + self.fc_b
        self.cache_fc = (h_last, logits)
        
        return softmax(logits)

    def backward(self, y_true, y_pred):
        """
        y_true: One-hot (Batch, Num_classes)
        y_pred: Softmax output (Batch, Num_classes)
        """
        batch_size = y_true.shape[0]
        
        # 1. Gradient Cross-Entropy + Softmax
        # dL/dz = y_pred - y_true
        d_logits = (y_pred - y_true) / batch_size
        
        h_last, _ = self.cache_fc
        
        # Gradient cho FC Layer
        dfc_W = np.dot(h_last.T, d_logits)
        dfc_b = np.sum(d_logits, axis=0)
        
        # Gradient truyền ngược về LSTM cuối cùng (tại bước thời gian cuối)
        dh_last = np.dot(d_logits, self.fc_W.T) # (Batch, Hidden)
        
        # Chuẩn bị gradient toàn chuỗi cho lớp LSTM cuối (chỉ bước cuối có gradient từ loss)
        # Các bước trước đó gradient = 0 (vì Many-to-One)
        time_steps = self.layers[0].cache[0].shape[1]
        hidden_dim = self.layers[0].hidden_dim
        
        dh_seq = np.zeros((batch_size, time_steps, hidden_dim))
        dh_seq[:, -1, :] = dh_last
        
        # 2. Backprop qua 5 lớp LSTM (Ngược từ lớp 4 về 0)
        current_grad = dh_seq
        for i in reversed(range(len(self.layers))):
            current_grad = self.layers[i].backward(current_grad)
            
            # Cập nhật params dict để Optimizer lấy
            layer_name = f'lstm_{i}'
            self.params[f'{layer_name}_W'] = self.layers[i].grads['W']
            self.params[f'{layer_name}_b'] = self.layers[i].grads['b']

        self.params['fc_W'] = dfc_W
        self.params['fc_b'] = dfc_b
        
        return self.params

def cross_entropy_loss(y_pred, y_true):
    # Thêm epsilon để tránh log(0)
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# --- 5. MAIN TRAINING SCRIPT ---

# Giả lập dữ liệu đã xử lý từ bước trước (Để code chạy được độc lập)
# Trong thực tế, bạn sẽ lấy X_train_emb từ DataLoader ở nhiệm vụ 1
BATCH_SIZE = 32
EMBED_DIM = 100
HIDDEN_DIM = 256
SEQ_LEN = 40
NUM_CLASSES = 3
NUM_LAYERS = 5
EPOCHS = 5

# Tạo dữ liệu giả lập để test code
print("Đang khởi tạo dữ liệu giả lập...")
X_train = np.random.randn(100, SEQ_LEN, EMBED_DIM) # 100 mẫu
y_train_indices = np.random.randint(0, NUM_CLASSES, 100)
# One-hot encoding y_train
y_train = np.eye(NUM_CLASSES)[y_train_indices]

# Khởi tạo Model
model = DeepLSTMClassifier(vocab_size=None, embed_dim=EMBED_DIM, 
                           hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, 
                           num_classes=NUM_CLASSES)

# Khởi tạo Optimizer
# Lưu ý: Optimizer giữ tham chiếu đến model.params
optimizer = AdamOptimizer(model.params, lr=0.001)

# Training Loop
print(f"Bắt đầu training Deep LSTM ({NUM_LAYERS} layers)...")

for epoch in range(EPOCHS):
    # Shuffle dữ liệu
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    # Mini-batch gradient descent
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        # Lấy batch
        X_batch = X_train_shuffled[i:i+BATCH_SIZE]
        y_batch = y_train_shuffled[i:i+BATCH_SIZE]
        
        # 1. Forward
        probs = model.forward(X_batch)
        
        # 2. Loss
        loss = cross_entropy_loss(probs, y_batch)
        epoch_loss += loss
        
        # 3. Backward (Tính Grads)
        grads = model.backward(y_batch, probs)
        
        # 4. Update Params (Adam)
        # Vì params trong optimizer trỏ tới cùng bộ nhớ với model.params, 
        # nên model sẽ tự động được update
        optimizer.step(grads)
        
        # Lưu kết quả để tính metrics
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(y_batch, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        # Cập nhật lại weight vào các layer (do dictionary chỉ lưu tham chiếu, 
        # nhưng tốt nhất gán ngược lại để chắc chắn nếu logic update thay đổi)
        for idx, layer in enumerate(model.layers):
            layer.W = model.params[f'lstm_{idx}_W']
            layer.b = model.params[f'lstm_{idx}_b']
        model.fc_W = model.params['fc_W']
        model.fc_b = model.params['fc_b']

    # Đánh giá cuối Epoch
    avg_loss = epoch_loss / (X_train.shape[0] // BATCH_SIZE)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | F1-Score: {f1:.4f} | Acc: {acc:.4f}")

print("Hoàn tất huấn luyện.")
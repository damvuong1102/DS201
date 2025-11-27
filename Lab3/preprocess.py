import json
import numpy as np
import os
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

class TextProcessor:
    """
    Class chịu trách nhiệm xử lý văn bản, xây dựng bộ từ điển và mã hóa dữ liệu.
    """
    def __init__(self, max_length: int = 50, embedding_dim: int = 300):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        # Khởi tạo vocab với 2 token đặc biệt: Padding (0) và Unknown (1)
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
        # Mapping nhãn cảm xúc
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.num_classes = 3

    def simple_tokenize(self, text: str) -> List[str]:
        """
        Tách từ đơn giản bằng khoảng trắng và chuyển về chữ thường.
        """
        if not isinstance(text, str):
            return []
        # Xóa các ký tự đặc biệt cơ bản nếu cần, ở đây giữ nguyên logic split whitespace
        return text.lower().strip().split()

    def build_vocab(self, sentences: List[str]):
        """
        Xây dựng bộ từ điển từ danh sách các câu.
        """
        for sentence in sentences:
            tokens = self.simple_tokenize(sentence)
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1
        print(f"[Info] Vocab size built: {self.vocab_size} words.")

    def load_embedding_matrix(self, embedding_path: str = None) -> np.ndarray:
        """
        Tải Pre-trained Word Vectors.
        Nếu không có file hoặc từ không tồn tại, khởi tạo ngẫu nhiên.
        """
        # Khởi tạo ma trận embedding ngẫu nhiên (phân phối chuẩn)
        # Scale nhỏ (0.01) để giúp mô hình hội tụ tốt hơn ban đầu
        embedding_matrix = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        
        # Gán vector 0 cho token <PAD>
        embedding_matrix[0] = np.zeros(self.embedding_dim)

        if embedding_path and os.path.exists(embedding_path):
            print(f"[Info] Loading embeddings from {embedding_path}...")
            hit = 0
            with open(embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in self.word2idx:
                        # Lấy vector từ file
                        vector = np.asarray(values[1:], dtype='float32')
                        if len(vector) == self.embedding_dim:
                            idx = self.word2idx[word]
                            embedding_matrix[idx] = vector
                            hit += 1
            print(f"[Info] Loaded {hit}/{self.vocab_size} words from pre-trained embeddings.")
        else:
            print("[Info] No pre-trained embedding file found. Using random initialization.")
            
        return embedding_matrix

    def texts_to_sequences(self, sentences: List[str]) -> List[List[int]]:
        """
        Chuyển câu văn thành danh sách các chỉ số (indices).
        """
        sequences = []
        for sentence in sentences:
            tokens = self.simple_tokenize(sentence)
            seq = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Padding: Chuyển danh sách index thành numpy array kích thước cố định.
        Cắt nếu dài hơn max_length, thêm 0 (PAD) nếu ngắn hơn.
        """
        num_samples = len(sequences)
        padded_data = np.zeros((num_samples, self.max_length), dtype=np.int32)

        for i, seq in enumerate(sequences):
            # Lấy độ dài thực tế, cắt nếu quá dài
            length = min(len(seq), self.max_length)
            # Điền vào mảng (Padding='post' - điền số 0 vào sau)
            padded_data[i, :length] = seq[:length]
            
        return padded_data

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Chuyển nhãn text sang One-hot vector.
        VD: 'positive' -> [0, 0, 1]
        """
        num_samples = len(labels)
        one_hot_labels = np.zeros((num_samples, self.num_classes), dtype=np.float32)
        
        for i, label in enumerate(labels):
            if label in self.label_map:
                idx = self.label_map[label]
                one_hot_labels[i, idx] = 1.0
            # Nếu nhãn lạ, có thể xử lý ngoại lệ hoặc để toàn 0
            
        return one_hot_labels


class DataLoader:
    """
    Class chịu trách nhiệm tải file, quản lý dữ liệu thô và chia tập.
    """
    def __init__(self, file_path: str, text_processor: TextProcessor):
        self.file_path = file_path
        self.processor = text_processor
        self.raw_data = []
        self.X_data = None
        self.y_data = None

    def load_data(self):
        """Đọc file JSON."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found.")
            
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Đọc toàn bộ nội dung
            data = json.load(f)
            # Xử lý trường hợp file chứa dict hay list
            if isinstance(data, dict) and 'fullContent' in data:
                self.raw_data = data['fullContent'] # Theo format bạn cung cấp
            elif isinstance(data, list):
                self.raw_data = data
            else:
                raise ValueError("JSON structure not recognized.")
        
        print(f"[Info] Loaded {len(self.raw_data)} samples.")

    def prepare_data(self):
        """
        Thực hiện toàn bộ quy trình tiền xử lý.
        """
        sentences = [item['sentence'] for item in self.raw_data]
        labels = [item['sentiment'] for item in self.raw_data]

        # 1. Xây dựng bộ từ điển (chỉ nên làm trên tập train, nhưng demo ta làm trên toàn bộ)
        self.processor.build_vocab(sentences)

        # 2. Chuyển text sang sequence
        sequences = self.processor.texts_to_sequences(sentences)

        # 3. Padding
        self.X_data = self.processor.pad_sequences(sequences)

        # 4. Encode Labels
        self.y_data = self.processor.encode_labels(labels)
        
        print(f"[Info] Data shape: X={self.X_data.shape}, y={self.y_data.shape}")

    def get_train_val_split(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu sử dụng sklearn.
        """
        if self.X_data is None or self.y_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
            
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_data, 
            self.y_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y_data # Giữ cân bằng tỷ lệ nhãn
        )
        
        print(f"[Info] Split done. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
        return X_train, X_val, y_train, y_val

# --- MAIN EXECUTION (Demo) ---
if __name__ == "__main__":
    # 1. Cấu hình
    FILE_PATH = 'UIT-VSFC-train.json'  # Đảm bảo file này nằm cùng thư mục
    MAX_LEN = 40  # Độ dài câu cố định
    EMBED_DIM = 100
    
    # 2. Khởi tạo Processor
    processor = TextProcessor(max_length=MAX_LEN, embedding_dim=EMBED_DIM)
    
    # 3. Khởi tạo DataLoader
    # Lưu ý: Giả định file json đã tồn tại. Nếu chạy demo, bạn cần tạo file dummy.
    try:
        loader = DataLoader(FILE_PATH, processor)
        loader.load_data()
        loader.prepare_data()
        
        # 4. Load Embeddings (Demo giả lập vì không có file vector thật)
        # Nếu bạn có file vector (ví dụ: word2vec_vi.vec), hãy điền đường dẫn vào.
        embedding_matrix = processor.load_embedding_matrix(embedding_path=None)
        print(f"[Info] Embedding Matrix Shape: {embedding_matrix.shape}")
        
        # 5. Chia tập dữ liệu
        X_train, X_val, y_train, y_val = loader.get_train_val_split()
        
        # Kiểm tra kết quả
        print("\n--- Sample Data Check ---")
        print("Original Sentence:", loader.raw_data[0]['sentence'])
        print("Tokenized Index:", X_train[0]) # Lưu ý: do shuffle nên index 0 này không nhất thiết là câu đầu tiên
        print("One-hot Label:", y_train[0])
        
    except Exception as e:
        print(f"Error: {e}")
        print("Vui lòng đảm bảo file 'UIT-VSFC-train.json' tồn tại đúng định dạng.")
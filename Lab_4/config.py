import torch

class Config:
    TRAIN_PATH = '/content/small-train.json'
    DEV_PATH = '/content/small-dev.json'
    TEST_PATH = '/content/small-test.json'
    
    # JSON Keys
    SRC_KEY = 'english' 
    TGT_KEY = 'vietnamese'
    
    # Model Params (Chỉnh sửa thoải mái tại đây)
    D_MODEL = 256
    N_ENC_LAYERS = 3
    N_DEC_LAYERS = 3
    DROPOUT = 0.5
    
    # Training Params
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15 # Tăng lên vì chạy GPU nhanh hơn
    CLIP = 1.0
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Special Tokens
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

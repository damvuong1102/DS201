
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
import utils
# Định nghĩa các hằng số 
NUM_CLASSES = 21
IMG_SIZE = (64, 64)  # LeNet dùng ảnh nhỏ
BATCH_SIZE = 32
EPOCHS = 30  

def build_lenet(input_shape, num_classes):
    model = models.Sequential([
        # C1: Convolutional Layer
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', 
                      input_shape=input_shape, padding='same'),
        # S2: Subsampling (Pooling) Layer
        layers.AveragePooling2D(pool_size=(2, 2)),
        
        # C3: Convolutional Layer
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        # S4: Subsampling (Pooling) Layer
        layers.AveragePooling2D(pool_size=(2, 2)),
        
        # Flatten
        layers.Flatten(),
        
        # C5: Fully Connected Layer
        layers.Dense(120, activation='relu'),
        
        # F6: Fully Connected Layer
        layers.Dense(84, activation='relu'),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def run_assignment_1():
    print("--- Bắt đầu Bài 1: LeNet ---")
    
    # 1. Tải dữ liệu 
    train_gen, val_gen, test_gen = utils.create_data_generators(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 2. Xây dựng mô hình
    input_shape = IMG_SIZE + (3,) 
    model = build_lenet(input_shape, NUM_CLASSES)
    model.summary()
    
    # 3. Biên dịch (Compile)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Định nghĩa Checkpoint
    checkpoint_dir = "/content/drive/My Drive/Colab_Checkpoints/Lab2"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = f"{checkpoint_dir}/best_model_bai_1.keras" 
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1 
    )

    # 5. Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Chờ 5 epochs
        verbose=1,
        mode='max'
    )
    
    # 6. Huấn luyện
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    print("--- Hoàn thành huấn luyện Bài 1 ---")
    
    # 7. Load lại model tốt nhất 
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    return best_model, history, test_gen

if __name__ == '__main__':
    model1, history1, test_data1 = run_assignment_1()
    
    # Đánh giá và in report
    utils.plot_history(history1, title="LeNet (Bài 1)")
    utils.evaluate_and_report(model1, test_data1, title="LeNet (Bài 1)")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, Input
import utils  
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
# Định nghĩa các hằng số 
NUM_CLASSES = 21
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40  

def inception_module(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    
   #Hàm định nghĩa một khối Inception Module CÓ Batch Normalization.
    
    
    # Nhánh 1: 1x1 Conv
    path1 = layers.Conv2D(f1, (1, 1), padding='same')(x)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.ReLU()(path1)

    # Nhánh 2: 1x1 Conv -> 3x3 Conv
    path2 = layers.Conv2D(f2_in, (1, 1), padding='same')(x)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.ReLU()(path2)
    path2 = layers.Conv2D(f2_out, (3, 3), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.ReLU()(path2)

    # Nhánh 3: 1x1 Conv -> 5x5 Conv
    path3 = layers.Conv2D(f3_in, (1, 1), padding='same')(x)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.ReLU()(path3)
    path3 = layers.Conv2D(f3_out, (5, 5), padding='same')(path3)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.ReLU()(path3)

    # Nhánh 4: 3x3 MaxPool -> 1x1 Conv
    path4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(f4_out, (1, 1), padding='same')(path4)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.ReLU()(path4)

    # Concatenate (ghép) 4 nhánh lại
    return layers.concatenate([path1, path2, path3, path4], axis=-1)

def build_googlenet_simple(input_shape, num_classes):
    """
    Xây dựng kiến trúc GoogLeNet (phiên bản cập nhật CÓ Batch Normalization).
    """
    input_tensor = Input(shape=input_shape)

    # 1. Stem (Phần thân đầu)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = layers.Conv2D(64, (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(192, (3, 3), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # 2. Các khối Inception
    # Inception 3a, 3b
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Inception 4a, 4b, 4c, 4d, 4e
    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Inception 5a, 5b
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    # 3. Phần Classifier (Phân loại)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    # Tạo mô hình
    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

def run_assignment_2():

    print("--- Bắt đầu Bài 2: GoogLeNet ---")
    
    # 1. Tải dữ liệu
    train_gen, val_gen, test_gen = utils.create_data_generators(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 2. Xây dựng mô hình
    input_shape = IMG_SIZE + (3,)  # (224, 224, 3)
    model = build_googlenet_simple(input_shape, NUM_CLASSES)
    
    model.summary()
    
    # 3. Biên dịch (Compile)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #Định nghĩa checkpoint
    checkpoint_dir = "/content/drive/My Drive/Colab_Checkpoints/Lab2"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = f"{checkpoint_dir}/best_model_bai_2.keras" 
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1 
    )

    
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Chờ 5 epochs
        verbose=1,
        mode='max',
        restore_best_weights=False 
    )
    
    # 5. Huấn luyện 
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint_callback, early_stopping_callback] 
    )
    
    print("--- Hoàn thành huấn luyện Bài 2 ---")
    
    
    print(f"Đang tải model tốt nhất từ: {checkpoint_path}")
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    return best_model, history, test_gen

if __name__ == '__main__':
    model2, history2, test_data2 = run_assignment_2()
    utils.plot_history(history2, title="GoogLeNet (Bài 2)")
    utils.evaluate_and_report(model2, test_data2, title="GoogLeNet (Bài 2)")
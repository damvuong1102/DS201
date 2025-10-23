import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam 
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
import utils

# Định nghĩa các hằng số
NUM_CLASSES = 21
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32
EPOCHS = 40  

def residual_block(x, filters, strides=1, use_projection=False):
    
    #Hàm định nghĩa một khối Residual Block.
    #x: input tensor
    #filters: số filter cho các lớp Conv2D
    #strides: bước nhảy (stride)
    #use_projection: True nếu cần dùng 1x1 Conv ở đường tắt (shortcut) 
                    #để điều chỉnh kích thước

    

    shortcut = x
    
    # Kiểm tra xem có cần dùng 1x1 Conv (projection) ở đường tắt không
    # Cần khi: 
    # 1. Bước nhảy (strides) > 1 (giảm kích thước không gian)
    # 2. Số filter của đầu vào và đầu ra khác nhau
    if use_projection:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Khối chính (main path)
    # Lớp Conv2D thứ nhất
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Lớp Conv2D thứ hai
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Cộng đường tắt (Add)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def build_resnet18(input_shape, num_classes):
    #Xây dựng kiến trúc ResNet-18.

    input_tensor = Input(shape=input_shape)

    # 1. Stem (Phần thân đầu)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # 2. Các khối Residual (ResNet-18 có 4 nhóm: 2, 2, 2, 2)
    
    # Nhóm 1 (conv2_x) - 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Nhóm 2 (conv3_x) - 128 filters
    x = residual_block(x, 128, strides=2, use_projection=True) # Block đầu tiên giảm kích thước
    x = residual_block(x, 128)

    # Nhóm 3 (conv4_x) - 256 filters
    x = residual_block(x, 256, strides=2, use_projection=True) # Block đầu tiên giảm kích thước
    x = residual_block(x, 256)

    # Nhóm 4 (conv5_x) - 512 filters
    x = residual_block(x, 512, strides=2, use_projection=True) # Block đầu tiên giảm kích thước
    x = residual_block(x, 512)

    # 3. Phần Classifier (Phân loại)
    x = layers.GlobalAveragePooling2D()(x)
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    # Tạo mô hình
    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

def run_assignment_3():
    print("--- Bắt đầu Bài 3: ResNet-18 ---")
    
    # 1. Tải dữ liệu 
    train_gen, val_gen, test_gen = utils.create_data_generators(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 2. Xây dựng mô hình
    input_shape = IMG_SIZE + (3,)
    model = build_resnet18(input_shape, NUM_CLASSES)
    model.summary()
    
    # 3. Biên dịch (Compile)
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Định nghĩa Checkpoint
    checkpoint_dir = "/content/drive/My Drive/Colab_Checkpoints/Lab2"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = f"{checkpoint_dir}/best_model_bai_3.keras" 
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
    
    print("--- Hoàn thành huấn luyện Bài 3 ---")
    
    # 7. Load lại model tốt nhất
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    return best_model, history, test_gen

if __name__ == '__main__':

    model3, history3, test_data3 = run_assignment_3()
    
    # Đánh giá và in report
    utils.plot_history(history3, title="ResNet-18 (Bài 3)")
    utils.evaluate_and_report(model3, test_data3, title="ResNet-18 (Bài 3)")
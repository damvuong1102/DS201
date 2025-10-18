import tensorflow as tf
from tensorflow.keras import layers, models, Input
import utils  

# Định nghĩa các hằng số
NUM_CLASSES = 21
IMG_SIZE = (224, 224)  # GoogLeNet/Inception dùng ảnh 224x224
BATCH_SIZE = 32
EPOCHS = 40  

def inception_module(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    
    #Hàm định nghĩa một khối Inception Module.
    #x: input tensor
    #f1: số filter cho nhánh 1x1 conv
    #f2_in, f2_out: số filter cho nhánh 1x1 -> 3x3 conv
    #f3_in, f3_out: số filter cho nhánh 1x1 -> 5x5 conv
    #f4_out: số filter cho nhánh 3x3 pool -> 1x1 conv
   
    
    # Nhánh 1: 1x1 Conv
    path1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    # Nhánh 2: 1x1 Conv -> 3x3 Conv
    path2 = layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu')(path2)

    # Nhánh 3: 1x1 Conv -> 5x5 Conv
    path3 = layers.Conv2D(f3_in, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu')(path3)

    # Nhánh 4: 3x3 MaxPool -> 1x1 Conv
    path4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu')(path4)

    # Concatenate (ghép) 4 nhánh lại
    return layers.concatenate([path1, path2, path3, path4], axis=-1)

def build_googlenet_simple(input_shape, num_classes):
    """
    Xây dựng kiến trúc GoogLeNet (phiên bản đơn giản hóa, không có auxiliary heads).
    """
    input_tensor = Input(shape=input_shape)

    # 1. Stem (Phần thân đầu)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Conv2D(64, (1, 1), strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(x)
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
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Huấn luyện
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )
    
    print("--- Hoàn thành huấn luyện Bài 2 ---")
    
    return model, history, test_gen

if __name__ == '__main__':
    model2, history2, test_data2 = run_assignment_2()
    
    # Đánh giá và in report
    utils.plot_history(history2, title="GoogLeNet (Bài 2)")
    utils.evaluate_and_report(model2, test_data2, title="GoogLeNet (Bài 2)")
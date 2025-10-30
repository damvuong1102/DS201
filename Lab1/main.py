
import tensorflow as tf
import numpy as np
import os
import struct
from sklearn.metrics import classification_report
import argparse
from models import create_mlp_1_layer, create_mlp_3_layer

# --- HÀM ĐỌC DỮ LIỆU  ---
def load_mnist_from_files(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

# Hàm gọi model
def get_model(model_name):
    if model_name == 'MLP_1_Layer':
        print("Đang xây dựng mô hình 1 lớp ẩn...")
        return create_mlp_1_layer()
    elif model_name == 'MLP_3_Layer':
        print("Đang xây dựng mô hình 3 lớp ẩn...")
        return create_mlp_3_layer()
    else:
        raise ValueError("Tên model không hợp lệ! Vui lòng chọn 'MLP_1_Layer' hoặc 'MLP_3_Layer'")

# --- HÀM MAIN ĐỂ CHẠY CHƯƠNG TRÌNH ---
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình MLP trên MNIST.")
    parser.add_argument('--model', type=str, required=True, help="Tên của mô hình để huấn luyện ('MLP_1_Layer' hoặc 'MLP_3_Layer')")
    args = parser.parse_args()
    model_name = args.model

    print("Đang đọc và tiền xử lý dữ liệu...")
    train_images_path = 'train-images-idx3-ubyte'
    train_labels_path = 'train-labels-idx1-ubyte'
    test_images_path = 't10k-images-idx3-ubyte'
    test_labels_path = 't10k-labels-idx1-ubyte'
    
    x_train, y_train = load_mnist_from_files(train_images_path, train_labels_path)
    x_test, y_test = load_mnist_from_files(test_images_path, test_labels_path)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
    
    model = get_model(model_name)
    model.summary()

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"\nBắt đầu huấn luyện mô hình {model_name}...")
    model.fit(x_train, y_train_one_hot, epochs=10, batch_size=128, verbose=2)
    print("Hoàn thành huấn luyện!")

    print("\nĐang đánh giá mô hình...")
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    target_names = [f"Chữ số {i}" for i in range(10)]
    
    report = classification_report(y_test, y_pred_classes, target_names=target_names)
    
    output_filename = f"report_{model_name}.txt"
    with open(output_filename, "w",encoding = 'utf-8') as f:
        f.write(f"BÁO CÁO KẾT QUẢ CHO MÔ HÌNH: {model_name}\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    print(f"Đã lưu báo cáo kết quả vào file: {output_filename}")

if __name__ == "__main__":
    main()
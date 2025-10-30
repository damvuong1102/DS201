
import tensorflow as tf

def create_mlp_1_layer():
    """Hàm này xây dựng và trả về mô hình 1 lớp ẩn."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def create_mlp_3_layer():
    """Hàm này xây dựng và trả về mô hình 3 lớp ẩn."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
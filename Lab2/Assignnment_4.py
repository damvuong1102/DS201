import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam 
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from transformers import TFAutoModel
import utils
import keras 

# Định nghĩa các hằng số 
NUM_CLASSES = 21
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  
HF_MODEL_NAME = 'microsoft/resnet-50'

# Custom layer
@keras.saving.register_keras_serializable()
class HuggingFaceLayer(layers.Layer):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hf_model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        self.hf_model.trainable = False 

    def call(self, inputs, training=False):
        outputs = self.hf_model(inputs, training=training)
        return outputs.last_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_hf_resnet50(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Permute((3, 1, 2)),
        HuggingFaceLayer(HF_MODEL_NAME), # <<< Dùng Custom Layer
        layers.Permute((2, 3, 1)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def run_assignment_4():
    print("--- Bắt đầu Bài 4: HuggingFace ResNet-50 ---")
    
    # 1. Tải dữ liệu 
    train_gen, val_gen, test_gen = utils.create_data_generators(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 2. Xây dựng mô hình
    input_shape = IMG_SIZE + (3,)
    model = build_hf_resnet50(input_shape, NUM_CLASSES)
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
    
    checkpoint_path = f"{checkpoint_dir}/best_model_bai_4.keras"
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
    
    print("--- Hoàn thành huấn luyện Bài 4 ---")

    best_model = tf.keras.models.load_model(checkpoint_path)
    
    return best_model, history, test_gen

if __name__ == '__main__':
    model4, history4, test_data4 = run_assignment_4()
    utils.plot_history(history4, title="HF ResNet-50 (Bài 4)")
    utils.evaluate_and_report(model4, test_data4, title="HF ResNet-50 (Bài 4)")
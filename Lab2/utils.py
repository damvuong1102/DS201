import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def create_data_generators(train_dir='train', 
                           test_dir='test', 
                           img_size=(224, 224), 
                           batch_size=32):
   
    
    # 1. Định nghĩa Data Augmentation cho tập train
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  
    )

    # 2. Định nghĩa cho tập Test/Validation 
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # 3. Tạo các generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  
    )
    
    return train_generator, validation_generator, test_generator

def plot_history(history, title):
    """
    Vẽ đồ thị accuracy và loss cho training và validation.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Đồ thị Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy\n{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Đồ thị Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'Training and Validation Loss\n{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_and_report(model, test_generator, title):
   
    print(f"--- Báo cáo đánh giá cho: {title} ---")
    
    # Lấy các nhãn thật (y_true)
    y_true = test_generator.classes
    
    # Dự đoán (y_pred)
    y_pred_proba = model.predict(test_generator)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Lấy tên các lớp
    class_names = list(test_generator.class_indices.keys())
    
    # In báo cáo (Precision, Recall, F1-score)
    print(classification_report(y_true, y_pred, target_names=class_names))
    

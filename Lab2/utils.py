
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# Định nghĩa các phép biến đổi (Augmentation)
def get_data_transforms(img_size=(224, 224)):
    """
    Định nghĩa các phép Augmentation và Normalize cho PyTorch.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_data_loaders(train_dir='train', 
                       test_dir='test', 
                       img_size=(224, 224), 
                       batch_size=32,
                       validation_split=0.2):
    """
    Tạo các DataLoaders cho PyTorch.
    """
    print("--- Bắt đầu tải dữ liệu ---")
    
    data_transforms = get_data_transforms(img_size)
    
    # 1. Tải toàn bộ tập train (train + val)
    full_train_dataset = torchvision.datasets.ImageFolder(
        train_dir, 
        transform=data_transforms['train']
    )
    
    # Lấy tên các lớp (class_names)
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"Tìm thấy {len(full_train_dataset)} ảnh trong thư mục train, chia thành {num_classes} lớp.")

    # 2. Chia tập train/validation
    num_train = len(full_train_dataset)
    num_val = int(num_train * validation_split)
    num_train = num_train - num_val
    
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])
    
    # Gán transform "val" (không augmentation) cho val_dataset
    val_dataset.dataset = torchvision.datasets.ImageFolder(
        train_dir, 
        transform=data_transforms['val']
    )
    val_dataset.indices = val_dataset.indices # Đảm bảo nó vẫn dùng đúng các index đã chia
    
    print(f"Đã chia: {num_train} ảnh train, {num_val} ảnh validation.")

    # 3. Tải tập test
    test_dataset = torchvision.datasets.ImageFolder(
        test_dir, 
        transform=data_transforms['val'] # Dùng transform của 'val' cho test
    )
    print(f"Tìm thấy {len(test_dataset)} ảnh trong thư mục test.")

    # 4. Tạo các DataLoaders
    # num_workers=2 để tăng tốc tải dữ liệu
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    print("--- Tải dữ liệu hoàn tất ---")
    return dataloaders, dataset_sizes, class_names

def get_device():
    """
    Kiểm tra xem có GPU (CUDA) không.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def plot_history(train_acc_history, val_acc_history, train_loss_history, val_loss_history, title):
    """
    Vẽ đồ thị accuracy và loss 
    """
    epochs = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(14, 5))

    # Đồ thị Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_history, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc_history, 'ro-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy\n{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Đồ thị Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    plt.title(f'Training and Validation Loss\n{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_and_report(model, test_loader, device, class_names, title):
    """
    Đánh giá model trên tập test và in báo cáo
    """
    print(f"\n--- Báo cáo đánh giá cho: {title} ---")
    
    model.eval()  # Chuyển model sang chế độ đánh giá (tắt Dropout,...)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Không tính gradient khi đánh giá
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # In báo cáo (Precision, Recall, F1-score)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
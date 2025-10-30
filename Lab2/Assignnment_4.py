# Trong file Assignnment_4.py (PHIÊN BẢN PYTORCH MỚI)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
import os
import utils
import pretrained_resnet 

# --- Định nghĩa các hằng số cho Bài 4 ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20      
NUM_CLASSES = 21
PATIENCE = 5         
LEARNING_RATE = 0.0001 

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL ---
# =file pretrained_resnet.py)


# --- 2. ĐỊNH NGHĨA VÒNG LẶP HUẤN LUYỆN  ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs, patience):
    start_time = time.time()
    
    # Lưu lại model tốt nhất
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Dùng cho Early Stopping
    epochs_no_improve = 0
    
    # Lưu lại lịch sử
    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': []
    }

    print("--- Bắt đầu huấn luyện  ---")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Mỗi epoch có 2 pha: train và val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Tính loss/acc trung bình của epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'[{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu lịch sử
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Kiểm tra Early Stopping và Best Model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        # Kiểm tra Early Stopping
        if epochs_no_improve >= patience:
            print(f"\n--- Dừng sớm (Early Stopping) tại epoch {epoch+1} ---")
            break

    time_elapsed = time.time() - start_time
    print(f'\nHuấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Val Acc tốt nhất: {best_acc:.4f}')

    # Load lại model tốt nhất
    model.load_state_dict(best_model_wts)
    return model, history

# --- 3. HÀM CHẠY CHÍNH CỦA BÀI 4 ---
def run_assignment_4():
    """
    Hàm chính để chạy Bài 4 
    """
    # 1. Lấy thiết bị (GPU hoặc CPU)
    device = utils.get_device()
    print(f"Đang sử dụng thiết bị: {device}")
    
    # 2. Tải dữ liệu
    dataloaders, dataset_sizes, class_names = utils.create_data_loaders(
        img_size=IMG_SIZE, # (224, 224)
        batch_size=BATCH_SIZE
    )
    
    # 3. Xây dựng mô hình 
    model = pretrained_resnet.PretrainedResnet().to(device)
    
    # 4. Định nghĩa Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # --- TRANSFER LEARNING ---

    print("--- Đóng băng thân (ResNet), chỉ huấn luyện lớp Classifier ---")
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # 5. Huấn luyện
    model, history = train_model(
        model, 
        dataloaders, 
        dataset_sizes,
        criterion, 
        optimizer, 
        device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )
    
    # 6. Lưu checkpoint 
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/best_model_bai_4.pth')
    
    return model, history, dataloaders['test'], class_names

# --- 4. KHỐI CHẠY KHI GỌI FILE TRỰC TIẾP ---
if __name__ == '__main__':
    # 1. Chạy huấn luyện
    model_4, history_4, test_loader_4, class_names_4 = run_assignment_4()
    
    # 2. Vẽ đồ thị
    utils.plot_history(
        history_4['train_acc'], history_4['val_acc'],
        history_4['train_loss'], history_4['val_loss'],
        title="Pretrained ResNet-50 (Bài 4)"
    )
    
    # 3. Đánh giá và in báo cáo
    utils.evaluate_and_report(
        model_4, 
        test_loader_4, 
        utils.get_device(), 
        class_names_4,
        title="Pretrained ResNet-50 (Bài 4)"
    )
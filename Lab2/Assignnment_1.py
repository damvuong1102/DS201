import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
import os
import utils

# --- Định nghĩa các hằng số cho Bài 1 ---
IMG_SIZE = (64, 64)  
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_CLASSES = 21
PATIENCE = 5 # Dùng cho Early Stopping
LEARNING_RATE = 0.001

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL ---
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # Input: (3, 64, 64)
        
        # Lớp Conv + Pool đầu tiên
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5) 
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) 
        
        # Lớp Conv + Pool thứ hai
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        
        # Phải tính toán kích thước
        in_features_fc1 = 16 * 13 * 13
        
        # Các lớp Fully Connected
        self.fc1 = nn.Linear(in_features_fc1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Định nghĩa luồng dữ liệu
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # "Flatten" tensor
        x = x.view(-1, 16 * 13 * 13) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

# --- 2. ĐỊNH NGHĨA VÒNG LẶP HUẤN LUYỆN ---
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

    print("--- Bắt đầu huấn luyện ---")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Mỗi epoch có 2 pha: train và val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Chuyển model sang chế độ train (bật Dropout,...)
            else:
                model.eval()   # Chuyển model sang chế độ eval (tắt Dropout,...)

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            for inputs, labels in dataloaders[phase]:
                # Chuyển dữ liệu lên GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Xóa gradient
                optimizer.zero_grad()

                # Chỉ tính toán gradient khi ở pha 'train'
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backpropagation + optimize (chỉ làm khi train)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Tính toán thống kê
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

# --- 3. HÀM CHẠY CHÍNH CỦA BÀI 1 ---
def run_assignment_1():
    """
    Hàm chính để chạy Bài 1
    """
    # 1. Lấy thiết bị 
    device = utils.get_device()
    print(f"Đang sử dụng thiết bị: {device}")
    
    # 2. Tải dữ liệu
    dataloaders, dataset_sizes, class_names = utils.create_data_loaders(
        img_size=IMG_SIZE, # (64, 64)
        batch_size=BATCH_SIZE
    )
    
    # 3. Xây dựng mô hình
    model = LeNet(num_classes=NUM_CLASSES).to(device)
    
    # 4. Định nghĩa Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
    
    # 6. Lưu checkpoint (tùy chọn)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/best_model_bai_1.pth')
    
    return model, history, dataloaders['test'], class_names

# --- 4. KHỐI CHẠY KHI GỌI FILE TRỰC TIẾP ---
if __name__ == '__main__':
    # 1. Chạy huấn luyện
    model_1, history_1, test_loader_1, class_names_1 = run_assignment_1()
    
    # 2. Vẽ đồ thị
    utils.plot_history(
        history_1['train_acc'], history_1['val_acc'],
        history_1['train_loss'], history_1['val_loss'],
        title="LeNet (Bài 1 - PyTorch)"
    )
    
    # 3. Đánh giá và in báo cáo
    utils.evaluate_and_report(
        model_1, 
        test_loader_1, 
        utils.get_device(), 
        class_names_1,
        title="LeNet (Bài 1)"
    )
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
import os
import utils

# --- Định nghĩa các hằng số cho Bài 2 ---
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32
NUM_EPOCHS = 40      
NUM_CLASSES = 21
PATIENCE = 5         
LEARNING_RATE = 0.0001 

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL---

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    """
    Kiến trúc Inception Module.
    """
    def __init__(self, in_channels, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        
        # Nhánh 1: 1x1 Conv
        self.branch1 = BasicConv2d(in_channels, f_1x1, kernel_size=1)

        # Nhánh 2: 1x1 Conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, f_3x3_reduce, kernel_size=1),
            BasicConv2d(f_3x3_reduce, f_3x3, kernel_size=3, padding=1)
        )

        # Nhánh 3: 1x1 Conv -> 5x5 Conv
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, f_5x5_reduce, kernel_size=1),
            BasicConv2d(f_5x5_reduce, f_5x5, kernel_size=5, padding=2)
        )

        # Nhánh 4: 3x3 MaxPool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, f_pool_proj, kernel_size=1)
        )

    def forward(self, x):
        # Chạy 4 nhánh song song
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concat (nối) 4 nhánh lại
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogLeNetSimple(nn.Module):
    """
    Kiến trúc GoogLeNet đơn giản.
    """
    def __init__(self, num_classes):
        super(GoogLeNetSimple, self).__init__()
        
        # Phần "thân" (Stem)
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2 Inception Modules 
        # Inception 1: in=192
        self.inception1 = InceptionModule(192, f_1x1=64, f_3x3_reduce=64, f_3x3=128, f_5x5_reduce=32, f_5x5=32, f_pool_proj=32)
        # Output: 64 + 128 + 32 + 32 = 256 channels
        
        # Inception 2: in=256
        self.inception2 = InceptionModule(256, f_1x1=64, f_3x3_reduce=64, f_3x3=128, f_5x5_reduce=32, f_5x5=32, f_pool_proj=32)
        # Output: 64 + 128 + 32 + 32 = 256 channels
        
        # Phần "đầu" (Classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        # Flatten
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
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

# --- 3. HÀM CHẠY CHÍNH CỦA BÀI 2 ---
def run_assignment_2():
    """
    Hàm chính để chạy Bài 2 (PyTorch)
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
    model = GoogLeNetSimple(num_classes=NUM_CLASSES).to(device)
    
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
    torch.save(model.state_dict(), 'checkpoints/best_model_bai_2.pth')
    
    return model, history, dataloaders['test'], class_names

# --- 4. KHỐI CHẠY KHI GỌI FILE TRỰC TIẾP ---
if __name__ == '__main__':
    # 1. Chạy huấn luyện
    model_2, history_2, test_loader_2, class_names_2 = run_assignment_2()
    
    # 2. Vẽ đồ thị
    utils.plot_history(
        history_2['train_acc'], history_2['val_acc'],
        history_2['train_loss'], history_2['val_loss'],
        title="GoogLeNet (Bài 2)"
    )
    
    # 3. Đánh giá và in báo cáo
    utils.evaluate_and_report(
        model_2, 
        test_loader_2, 
        utils.get_device(), 
        class_names_2,
        title="GoogLeNet (Bài 2)"
    )
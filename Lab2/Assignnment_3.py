import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
import os
import utils

# --- Định nghĩa các hằng số cho Bài 3 ---
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32
NUM_EPOCHS = 40
NUM_CLASSES = 21
PATIENCE = 5         # Dùng cho Early Stopping
LEARNING_RATE = 0.0001 

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL (PYTORCH) ---

class ResidualBlock(nn.Module):
    """
    Residual Block cơ bản của ResNet.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_projection=False):
        super(ResidualBlock, self).__init__()
        
        # Luồng chính (Conv -> BN -> ReLU -> Conv -> BN)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Luồng "shortcut" (đường tắt)
        self.shortcut = nn.Sequential() # Mặc định là một "identity" (không làm gì cả)
        if use_projection:
            # Nếu kích thước thay đổi (do stride=2 hoặc filter tăng)
            # Dùng 1x1 Conv để "chiếu" shortcut lên kích thước mới
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x) # Lấy đường tắt
        
        # Đi qua luồng chính
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Cộng luồng chính và đường tắt
        out += identity
        out = self.relu(out) # ReLU cuối cùng
        return out

class ResNet18(nn.Module):
    """
    Kiến trúc ResNet-18 .
    """
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        # 1. Phần "thân" (Stem) 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 2. Các khối (layer) ResNet
        # ResNet-18 có 4 layer, mỗi layer 2 block [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        
        # 3. Phần "đầu" (Classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Hàm tiện ích để "xếp chồng" các Residual Blocks.
        """
        # Block đầu tiên có thể thay đổi kích thước (stride=2, use_projection=True)
        use_projection = (stride != 1) or (self.in_channels != out_channels)
        strides = [stride] + [1] * (num_blocks - 1) # [stride, 1, 1, ...]
        layers = []
        
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=strides[0], use_projection=use_projection))
        self.in_channels = out_channels # Cập nhật in_channels cho block sau
        
        # Các block còn lại trong layer
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=strides[i], use_projection=False))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten
        x = self.fc(x)
        return x

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

# --- 3. HÀM CHẠY CHÍNH CỦA BÀI 3 ---
def run_assignment_3():
    """
    Hàm chính để chạy Bài 3
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
    model = ResNet18(num_classes=NUM_CLASSES).to(device)
    
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
    
    # 6. Lưu checkpoint 
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/best_model_bai_3.pth')
    
    return model, history, dataloaders['test'], class_names

# --- 4. KHỐI CHẠY KHI GỌI FILE TRỰC TIẾP ---
if __name__ == '__main__':
    # 1. Chạy huấn luyện
    model_3, history_3, test_loader_3, class_names_3 = run_assignment_3()
    
    # 2. Vẽ đồ thị
    utils.plot_history(
        history_3['train_acc'], history_3['val_acc'],
        history_3['train_loss'], history_3['val_loss'],
        title="ResNet-18 (Bài 3)"
    )
    
    # 3. Đánh giá và in báo cáo
    utils.evaluate_and_report(
        model_3, 
        test_loader_3, 
        utils.get_device(), 
        class_names_3,
        title="ResNet-18 (Bài 3)"
    )
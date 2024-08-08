import torch.utils.benchmark.utils.compare
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from utils.ImageLoader import PersistenceImageDataset
from torch.utils.data import DataLoader
from models.Resnet import resnet18

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # 调整学习率
        scheduler.step()

        # 计算测试集上的准确率
        test_accuracy = test_model(model, test_loader, device)
        if best_acc < test_accuracy:
            best_acc = test_accuracy
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        # 清空未使用的显存
        torch.cuda.empty_cache()

    print('Training complete, best acc is: ', best_acc)

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def train_model_binary(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # 调整学习率
        scheduler.step()

        # 计算测试集上的准确率
        test_accuracy = test_model_binary(model, test_loader, device)
        if best_acc < test_accuracy:
            best_acc = test_accuracy
        print(f'Test Accuracy: {test_accuracy:.2f}%')

    print('Training complete, best acc is: ', best_acc)

def test_model_binary(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
    else:
        raise ValueError("Invalid model name, exiting...")
    # 调整最后一层全连接层
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model_ft


def initialize_model_binary(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
    else:
        raise ValueError("Invalid model name, exiting...")

    # 修改第一个卷积层
    # if model_name.startswith("resnet"):
    #     model_ft.conv1 = nn.Conv2d(0, 64, kernel_size=7, stride=1, padding=3, bias=False)

    # 调整最后一层全连接层
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)  # 输出一个值用于二分类
    )

    return model_ft

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入通道改为4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(f'Shape after conv and pool: {x.shape}')  # 打印形状
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    train_root_dir = 'data/organmnist3d_pi/train'
    test_root_dir = 'data/organmnist3d_pi/test'
    lr_rate = 0.08
    epoch = 150
    num_classes = 11

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # my_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485], [0.229])
    # ])

    my_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ])

    train_dataset = PersistenceImageDataset(train_root_dir, None, my_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)

    test_dataset = PersistenceImageDataset(test_root_dir, None, my_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, pin_memory=True)

    model_name = "resnet18"
    if num_classes == 2:
        model_ft = initialize_model_binary(model_name, num_classes, feature_extract=False, use_pretrained=True)
    elif num_classes > 2:
        model_ft = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
    else:
        raise ValueError("Wrong Classes Number.")

    model = model_ft.to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch * len(train_loader), eta_min=1e-6)

    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
        train_model_binary(model, train_loader, test_loader, criterion, optimizer, epoch, device, scheduler)
    elif num_classes > 2:
        criterion = torch.nn.CrossEntropyLoss()
        train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device, scheduler)


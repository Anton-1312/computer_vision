import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

# ============================================
# 1. CEK DEVICE (CPU/GPU)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")

# ============================================
# 2. DEFINISIKAN ARSITEKTUR LeNet-5
# ============================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Layer Konvolusi 1: input 1 channel (grayscale) -> 6 channel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # output: 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # output: 14x14
        
        # Layer Konvolusi 2: 6 channel -> 16 channel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # output: 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # output: 5x5
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Aktivasi
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv1 -> ReLU -> Pool1
        x = self.pool1(self.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool2
        x = self.pool2(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # FC Layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # tidak pakai softmax karena pakai CrossEntropyLoss
        return x

# ============================================
# 3. PERSIAPAN DATA dan Preprocessing
# ============================================
# Transformasi: ubah ke tensor dan normalisasi
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # mean & std Fashion-MNIST
])

# Download dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Nama kelas Fashion-MNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Jumlah data training: {len(train_dataset)}")
print(f"Jumlah data testing: {len(test_dataset)}")

# ============================================
# 4. VISUALISASI SAMPLE DATA
# ============================================
def show_sample_images():
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        img, label = train_dataset[i]
        ax = axes[i // 5, i % 5]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'{classes[label]}')
        ax.axis('off')
    plt.suptitle('Sample Fashion-MNIST Images', fontsize=16)
    plt.tight_layout()
    plt.show()

show_sample_images()

# ============================================
# 5. INISIALISASI MODEL, LOSS, OPTIMIZER
# ============================================
model = LeNet5(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nTotal parameter model: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 6. FUNGSI TRAINING
# ============================================
def train_model(epochs=15):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Testing phase
        test_acc = evaluate_model(return_accuracy=True)
        test_accuracies.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Training selesai dalam {elapsed_time/60:.2f} menit")
    
    return train_losses, train_accuracies, test_accuracies

# ============================================
# 7. FUNGSI EVALUASI
# ============================================
def evaluate_model(return_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    if return_accuracy:
        return accuracy
    else:
        print(f"Akurasi pada test set: {accuracy:.2f}%")
        return accuracy

# ============================================
# 8. JALANKAN TRAINING
# ============================================
epochs = 15
train_losses, train_accuracies, test_accuracies = train_model(epochs)

# ============================================
# 9. VISUALISASI HASIL TRAINING
# ============================================
def plot_training_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(range(1, epochs+1), train_losses, marker='o', color='blue')
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(range(1, epochs+1), train_accuracies, marker='o', label='Train Acc', color='green')
    ax2.plot(range(1, epochs+1), test_accuracies, marker='s', label='Test Acc', color='red')
    ax2.set_title('Training & Testing Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_results()

# ============================================
# 10. CONFUSION MATRIX
# ============================================
def plot_confusion_matrix():
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Fashion-MNIST', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

plot_confusion_matrix()

# ============================================
# 11. PREDIKSI SAMPLE GAMBAR
# ============================================
def show_predictions():
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Ambil 10 gambar dari test set
    images, labels = next(iter(test_loader))
    images, labels = images[:10].to(device), labels[:10].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}', 
                     color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Prediction Results (Green = Correct, Red = Wrong)', fontsize=14)
    plt.tight_layout()
    plt.show()

show_predictions()

# ============================================
# 12. SIMPAN MODEL (OPSIONAL)
# ============================================
# torch.save(model.state_dict(), 'lenet5_fashion_mnist.pth')
# print("Model berhasil disimpan!")

print("\n🎉 Tugas Selesai!")

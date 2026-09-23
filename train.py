import time
import torch
import torch.nn as nn

def get_computing_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Hardware Acceleration Setup ===")
    print(f"Active Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Cores / MultiProcessors Available: {torch.cuda.get_device_properties(0).multi_processor_count}")
    else:
        print("CUDA Hardware Acceleration unavailable. Fallback to CPU.")
    return device


def train_svm_pipeline(train_loader, test_loader, epochs=5, lr=0.01, device=None):
    if device is None:
        device = get_computing_device()
        
    print(f"\n--- Processing Model 2: PyTorch GPU-Accelerated Linear SVM on {device} ---")
    start_time = time.time()

    
    svm_model = nn.Linear(3 * 32 * 32, 10).to(device)
    optimizer = torch.optim.SGD(svm_model.parameters(), lr=lr, weight_decay=1e-4)


    criterion = nn.MultiMarginLoss()

    for epoch in range(epochs):
        svm_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = svm_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"SVM Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    train_time = time.time() - start_time

    
    svm_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = svm_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = correct / total
    print(f"PyTorch SVM Training Time: {train_time:.2f}s | Test Accuracy: {test_acc*100:.2f}%")
    return test_acc, train_time


def train_deep_model(model, model_name, train_loader, val_loader, epochs=10, lr=0.001, device=None):
    if device is None:
        device = get_computing_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    print(f"\n--- Training {model_name} on {device} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    total_time = time.time() - start_time
    return model, history, total_time
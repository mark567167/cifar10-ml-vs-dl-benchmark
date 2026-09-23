import matplotlib.pyplot as plt
import torch

def evaluate_test_set(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total

def plot_training_curves(histories):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model_name, h in histories.items():
        plt.plot(h['train_loss'], label=f'{model_name} Train Loss')
        plt.plot(h['val_loss'], linestyle='--', label=f'{model_name} Val Loss')
    plt.title('Loss Trajectory Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for model_name, h in histories.items():
        plt.plot(h['val_acc'], label=f'{model_name} Val Accuracy')
    plt.title('Validation Accuracy Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('final_benchmark_curves.png')
    plt.show()
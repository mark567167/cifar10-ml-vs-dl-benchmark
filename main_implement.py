from dataset import get_data_loaders, visualize_sample_data
from models import MLPNetwork, CustomCNN, get_resnet50_model, count_parameters
from train import get_computing_device, train_svm_pipeline, train_deep_model
from evaluate import evaluate_test_set, plot_training_curves

def main():
    device = get_computing_device()

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    visualize_sample_data(train_loader)

    benchmark_results = {}
    histories = {}

    # 1. FNN / MLP
    mlp_model = MLPNetwork()
    mlp_params = count_parameters(mlp_model)
    mlp_trained, mlp_hist, mlp_time = train_deep_model(mlp_model, "FNN_MLP", train_loader, val_loader, epochs=10, device=device)
    mlp_test_acc = evaluate_test_set(mlp_trained, test_loader, device=device)
    histories['FNN'] = mlp_hist
    benchmark_results['FNN'] = {'params': mlp_params, 'time': mlp_time, 'val_acc': mlp_hist['val_acc'][-1], 'test_acc': mlp_test_acc}

    # 2. SVM
    svm_acc, svm_time = train_svm_pipeline(train_loader, test_loader)
    benchmark_results['SVM'] = {'params': 0, 'time': svm_time, 'val_acc': 0, 'test_acc': svm_acc}

    # 3. Custom CNN
    cnn_model = CustomCNN()
    cnn_params = count_parameters(cnn_model)
    cnn_trained, cnn_hist, cnn_time = train_deep_model(cnn_model, "Custom_CNN", train_loader, val_loader, epochs=10, device=device)
    cnn_test_acc = evaluate_test_set(cnn_trained, test_loader, device=device)
    histories['Custom_CNN'] = cnn_hist
    benchmark_results['Custom_CNN'] = {'params': cnn_params, 'time': cnn_time, 'val_acc': cnn_hist['val_acc'][-1], 'test_acc': cnn_test_acc}

    # 4. ResNet-50 Fine-Tuning
    resnet_model = get_resnet50_model()
    resnet_params = count_parameters(resnet_model)
    resnet_trained, resnet_hist, resnet_time = train_deep_model(resnet_model, "ResNet50", train_loader, val_loader, epochs=5, device=device)
    resnet_test_acc = evaluate_test_set(resnet_trained, test_loader, device=device)
    histories['ResNet50'] = resnet_hist
    benchmark_results['ResNet50'] = {'params': resnet_params, 'time': resnet_time, 'val_acc': resnet_hist['val_acc'][-1], 'test_acc': resnet_test_acc}

    # Plot & Summary
    plot_training_curves(histories)

    print("\n" + "="*70)
    print(f"{'Model Architecture':<20} | {'Params Count':<12} | {'Time (s)':<10} | {'Val Acc (%)':<12} | {'Test Acc (%)':<12}")
    print("="*70)
    for model_name, metrics in benchmark_results.items():
        print(f"{model_name:<20} | {metrics['params']:<12,} | {metrics['time']:<10.2f} | {metrics['val_acc']*100:<12.2f} | {metrics['test_acc']*100:<12.2f}")
    print("="*70)

if __name__ == '__main__':
    main()
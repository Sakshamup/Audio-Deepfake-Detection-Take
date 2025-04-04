# metrics_calculator.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from app import CRNN, ASVspoofDataset, find_all_datasets, load_dataset, DataLoader
import os

def calculate_metrics(model_path="crnn_unified_model.pth", root_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Find all datasets
    datasets = find_all_datasets(root_dir)
    
    # Aggregate all test files
    all_test_files, all_test_labels = [], []
    for dataset_path in datasets:
        test_files, test_labels = load_dataset(dataset_path, "testing")
        all_test_files.extend(test_files)
        all_test_labels.extend(test_labels)
    
    # Create test dataset
    test_dataset = ASVspoofDataset(all_test_files, all_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for mel_spec, label in tqdm(test_loader, desc="Calculating Metrics"):
            mel_spec, label = mel_spec.to(device), label.to(device)
            output = model(mel_spec)
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Calculate and print metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['REAL', 'FAKE']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    print("\nDetailed Metrics:")
    print(f"F1-Score: {f1_score(all_labels, all_predictions):.4f}")
    print(f"Precision: {precision_score(all_labels, all_predictions):.4f}")
    print(f"Recall: {recall_score(all_labels, all_predictions):.4f}")

if __name__ == "__main__":
    calculate_metrics()
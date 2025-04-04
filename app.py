import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # LSTM layer (input size = 128 channels * 25 time steps = 3200)
        self.lstm = nn.LSTM(
            input_size=3200,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(512, 128),  # 512 because bidirectional (256*2)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch, height, channels * width)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Dataset Loader
class ASVspoofDataset(Dataset):
    def __init__(self, file_list, labels, max_len=200):
        self.file_list = file_list
        self.labels = labels
        self.max_len = max_len
        self.n_mels = 128
        self.sr = 16000
        self.n_fft = 1024
        self.hop_length = 512
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        try:
            audio, sr = librosa.load(file_path, sr=self.sr)
            if len(audio) == 0:
                audio = np.zeros(self.sr)
        except:
            audio = np.zeros(self.sr)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        if mel_spec.shape[1] < self.max_len:
            pad_width = ((0, 0), (0, self.max_len - mel_spec.shape[1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
        else:
            mel_spec = mel_spec[:, :self.max_len]
            
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        return mel_spec.unsqueeze(0), torch.tensor(label, dtype=torch.long)

# Function to find all dataset folders
def find_all_datasets(base_dir):
    """Find all dataset folders (for-2sec, for-norm, etc.)"""
    datasets = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith("for-"):
            datasets.append(full_path)
    return datasets

# Function to load dataset from structure
def load_dataset(base_path, dataset_type):
    real_files = []
    fake_files = []
    
    real_path = os.path.join(base_path, dataset_type, "real")
    fake_path = os.path.join(base_path, dataset_type, "fake")
    
    if os.path.exists(real_path):
        real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) 
                     if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if os.path.exists(fake_path):
        fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) 
                     if f.endswith(('.wav', '.mp3', '.flac'))]
    
    file_list = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    
    return file_list, labels

# Training Function for all datasets
def train_all_datasets(root_dir=".", model_save_path="crnn_unified_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all datasets
    datasets = find_all_datasets(root_dir)
    if not datasets:
        print("No dataset folders found (expected folders like 'for-2sec', 'for-norm', etc.)")
        return
    
    print(f"Found {len(datasets)} datasets:")
    for ds in datasets:
        print(f"- {os.path.basename(ds)}")
    
    # Aggregate all training/validation files
    all_train_files, all_train_labels = [], []
    all_val_files, all_val_labels = [], []
    
    for dataset_path in datasets:
        # Load training data
        train_files, train_labels = load_dataset(dataset_path, "training")
        all_train_files.extend(train_files)
        all_train_labels.extend(train_labels)
        
        # Load validation data
        val_files, val_labels = load_dataset(dataset_path, "validation")
        all_val_files.extend(val_files)
        all_val_labels.extend(val_labels)
    
    print(f"\nTotal training samples: {len(all_train_files)}")
    print(f"Total validation samples: {len(all_val_files)}")
    
    # Create datasets
    train_dataset = ASVspoofDataset(all_train_files, all_train_labels)
    val_dataset = ASVspoofDataset(all_val_files, all_val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model
    model = CRNN().to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_accuracy = 0
    for epoch in range(6):  # Increased epochs for larger dataset
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for mel_spec, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            mel_spec, label = mel_spec.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(mel_spec)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += label.size(0)
            train_correct += predicted.eq(label).sum().item()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for mel_spec, label in val_loader:
                mel_spec, label = mel_spec.to(device), label.to(device)
                output = model(mel_spec)
                loss = criterion(output, label)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += label.size(0)
                val_correct += predicted.eq(label).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with accuracy {val_acc:.2f}%")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

def evaluate_all_datasets(root_dir=".", model_path="crnn_unified_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find all datasets
    datasets = find_all_datasets(root_dir)
    if not datasets:
        print("No dataset folders found")
        return
    
    # Load model
    model = CRNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found at {model_path}")
        return
    
    model.eval()
    
    # Aggregate all test files
    all_test_files, all_test_labels = [], []
    
    for dataset_path in datasets:
        test_files, test_labels = load_dataset(dataset_path, "testing")
        all_test_files.extend(test_files)
        all_test_labels.extend(test_labels)
    
    print(f"\nTotal test samples: {len(all_test_files)}")
    
    # Create test dataset
    test_dataset = ASVspoofDataset(all_test_files, all_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for mel_spec, label in tqdm(test_loader, desc="Testing"):
            mel_spec, label = mel_spec.to(device), label.to(device)
            output = model(mel_spec)
            _, predicted = output.max(1)
            test_total += label.size(0)
            test_correct += predicted.eq(label).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\nTest Accuracy: {test_acc:.2f}%")

def predict_audio(file_path, model_path="crnn_unified_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CRNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found at {model_path}")
        return None, None
    
    model.eval()
    
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) == 0:
            audio = np.zeros(16000)
    except:
        audio = np.zeros(16000)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_mels=128, n_fft=1024, hop_length=512
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    if mel_spec.shape[1] < 200:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, 200 - mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :200]
    
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(mel_spec)
        prob = torch.softmax(output, dim=1)
        fake_prob = prob[0][1].item()
    
    return "FAKE" if fake_prob > 0.5 else "REAL", fake_prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("1. To train on all datasets: python script.py train")
        print("2. To test on all datasets: python script.py test")
        print("3. To predict single file: python script.py predict <audio_file_path>")
        print("\nNote: Place all dataset folders (for-2sec, for-norm, etc.) in the same directory as this script")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "train":
        train_all_datasets()
    elif mode == "test":
        evaluate_all_datasets()
    elif mode == "predict":
        if len(sys.argv) < 3:
            print("Please provide audio file path")
            sys.exit(1)
        prediction, confidence = predict_audio(sys.argv[2])
        if prediction:
            print(f"\nPrediction: {prediction} (Confidence: {confidence:.2%})")
    else:
        print("Invalid mode. Use 'train', 'test', or 'predict'")
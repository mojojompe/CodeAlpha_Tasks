import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from model import CharCNN
import os

def evaluate(model_path='models/char_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    # usage batch_size=1000 for faster eval
    _, _, test_loader = get_dataloaders(batch_size=1000)
    
    # Model
    model = CharCNN(num_classes=62).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f"\nTest Set Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()

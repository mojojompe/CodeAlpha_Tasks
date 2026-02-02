import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from dataset import get_dataloaders
from model import CharCNN

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({'Loss': running_loss / (total/train_loader.batch_size), 'Acc': 100. * correct / total})
        
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    return val_loss / len(val_loader), 100. * correct / total

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # Model
    model = CharCNN(num_classes=62).to(DEVICE)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    best_val_acc = 0.0
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, DEVICE, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/char_cnn.pth')
            print("Saved Best Model!")

if __name__ == "__main__":
    main()

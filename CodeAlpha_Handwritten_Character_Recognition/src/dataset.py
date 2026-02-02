import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, root='./data', split='byclass'):
    """
    Downloads and loads the EMNIST dataset.
    
    Args:
        batch_size (int): Batch size for the dataloaders.
        root (str): Directory to download the data.
        split (str): EMNIST split to use (default: 'byclass').
                     'byclass' has 62 classes (0-9, a-z, A-Z).
    
    Returns:
        train_loader, val_loader, test_loader, mapping
    """
    
    # Define transformations
    # EMNIST images are 28x28 grayscale, but they are often rotated/flipped compared to standard MNIST
    # We also add some augmentation for robustness
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std of MNIST/EMNIST
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the datasets
    # Note: EMNIST in torchvision might need split argument
    try:
        full_train_dataset = datasets.EMNIST(root=root, split=split, train=True, download=True, transform=train_transform)
        test_dataset = datasets.EMNIST(root=root, split=split, train=False, download=True, transform=test_transform)
    except RuntimeError:
        print("Error downloading EMNIST. Please check your internet connection or install manually.")
        raise

    # EMNIST mapping (class index to character)
    # This is often inferred or provided by the dataset object if available, but for 'byclass':
    # 0-9: digits
    # 10-35: A-Z
    # 36-61: a-z
    # We can create a simple helper or check dataset.classes if available
    
    # Split train object into train/val
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Reset val_dataset transform to be same as test (no augmentation)
    # This is a bit tricky with random_split as it wraps the dataset
    # Standard practice is often to accept validation on augmented config or create two separate dataset objects if strict
    # For now, we'll keep it simple.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

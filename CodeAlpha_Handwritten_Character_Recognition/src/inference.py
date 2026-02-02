import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CharCNN
import argparse
import os

def get_mapping():
    """ Returns a dictionary mapping class index to character. """
    mapping = {}
    # EMNIST ByClass: 0-9 (digits), 10-35 (A-Z), 36-61 (a-z)
    for i in range(10):
        mapping[i] = str(i)
    for i in range(26):
        mapping[10 + i] = chr(65 + i) # A-Z
    for i in range(26):
        mapping[36 + i] = chr(97 + i) # a-z
    return mapping

def predict_image(image_path, model_path='models/char_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = CharCNN(num_classes=62).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. using random weights (prediction will be garbage).")
    
    model.eval()
    
    # Preprocess Image
    # Load as grayscale
    image = Image.open(image_path).convert('L')
    
    # Transform: Resize to 28x28, Tensor, Normalize
    # Note: user provided images might need Inversion if they are black on white
    # EMNIST is white character on black background.
    # We should add an option or check simple statistics to invert if needed.
    # Use standard transform first
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = output.max(1)
        
    mapping = get_mapping()
    predicted_char = mapping[predicted_idx.item()]
    
    print(f"Predicted Class Index: {predicted_idx.item()}")
    print(f"Predicted Character: {predicted_char}")
    
    # Vis
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {predicted_char}")
    plt.axis('off')
    plt.show()
    
    return predicted_char

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handwritten Character Recognition Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/char_cnn.pth', help='Path to trained model')
    
    args = parser.parse_args()
    predict_image(args.image, args.model)

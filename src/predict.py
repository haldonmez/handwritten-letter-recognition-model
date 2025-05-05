import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import numpy as np
from .model import LetterRecognizerModel
from otsu import findGreatestThreshold


def load_model(model_path: str, input_size=1, output_size=62, device=None):
    """Load a saved model from disk."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LetterRecognizerModel(input_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image_path: str, device=None):
    """Predict the class of a single image using a trained model and custom thresholding."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")
    transform = Compose([
        Resize((28, 28)),
        ToTensor()
    ])
    image_tensor = transform(image).squeeze().numpy()  # Shape: (28, 28), pixel range [0, 1]

    # Apply custom Otsu thresholding
    thresholded_image = findGreatestThreshold(image_tensor)

    # Prepare tensor for prediction
    input_tensor = torch.tensor(thresholded_image).unsqueeze(0).unsqueeze(0).float().to(device)

    # Predict
    with torch.inference_mode():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, 1).item()
        probability = F.softmax(output, dim=1)[0, predicted_class].item()

    return predicted_class, probability

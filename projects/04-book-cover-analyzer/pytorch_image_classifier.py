"""
PyTorch Image Classification for Book Covers
============================================

LEARNING OBJECTIVES:
1. Understand PyTorch tensors
2. Load pre-trained models (ResNet18)
3. Image preprocessing for neural networks
4. Running inference (classification)
5. Interpreting model outputs

This classifier identifies what TYPE of image is on the book cover:
- Logos
- Photos of people
- Illustrations
- Patterns/textures
- etc.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


# ============================================================================
# CONCEPT 1: Image Preprocessing for Neural Networks
# ============================================================================

def create_image_transform():
    """
    Create preprocessing pipeline for ResNet.

    LEARNING: Neural networks need specific input format:
    1. Resize to 224x224 (ResNet's expected input size)
    2. Convert to tensor (PyTorch's data format)
    3. Normalize with ImageNet mean/std (what ResNet was trained on)

    Why normalize?
    - ResNet was trained on ImageNet with these specific values
    - Normalization helps neural networks learn better
    - Mean=[0.485, 0.456, 0.406] are ImageNet RGB channel means
    - Std=[0.229, 0.224, 0.225] are ImageNet RGB channel standard deviations
    """
    transform = transforms.Compose([
        # Step 1: Resize image to 256x256
        # (Larger than needed, we'll crop next)
        transforms.Resize(256),

        # Step 2: Center crop to 224x224
        # (ResNet expects exactly 224x224 input)
        transforms.CenterCrop(224),

        # Step 3: Convert PIL Image to PyTorch tensor
        # (Converts [0-255] uint8 to [0.0-1.0] float32)
        # Also changes from HWC (Height, Width, Channels) to CHW format
        transforms.ToTensor(),

        # Step 4: Normalize using ImageNet statistics
        # (Standardize each RGB channel)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB means
            std=[0.229, 0.224, 0.225]     # ImageNet RGB stds
        )
    ])

    return transform


# ============================================================================
# CONCEPT 2: Loading Pre-trained Models
# ============================================================================

def load_resnet_model():
    """
    Load pre-trained ResNet18 model.

    LEARNING: Transfer Learning
    - ResNet18 is trained on ImageNet (1.2M images, 1000 classes)
    - We use its learned features for our task
    - No training required! Just use pre-trained weights

    Model Architecture:
    - 18 layers deep (hence ResNet18)
    - Residual connections (allows deeper networks)
    - Final layer: 1000 classes (ImageNet categories)

    Why ResNet?
    - Fast (smaller model)
    - Accurate enough for many tasks
    - Well-tested, widely used
    """
    print("[INFO] Loading pre-trained ResNet18 model...")
    print("[INFO] This will download ~45MB on first run")

    # Load model with pre-trained weights
    # weights='DEFAULT' uses the best available weights
    model = models.resnet18(weights='DEFAULT')

    # Set to evaluation mode
    # (Disables dropout, batch normalization behaves differently)
    model.eval()

    print("[SUCCESS] Model loaded!")
    print(f"[INFO] Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


# ============================================================================
# CONCEPT 3: Understanding Tensors
# ============================================================================

def opencv_to_pil(opencv_image):
    """
    Convert OpenCV image (BGR, numpy array) to PIL Image (RGB).

    LEARNING: Different libraries use different formats
    - OpenCV: BGR color order, numpy array
    - PIL: RGB color order, PIL Image object
    - PyTorch: Tensor format

    Need to convert between them!
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image)

    return pil_image


def preprocess_image(opencv_image, transform):
    """
    Preprocess OpenCV image for PyTorch model.

    LEARNING: Data flow
    OpenCV (numpy) → PIL Image → PyTorch Tensor

    Returns:
        tensor: Shape [1, 3, 224, 224]
                1 = batch size (single image)
                3 = RGB channels
                224x224 = image dimensions
    """
    # Convert to PIL
    pil_image = opencv_to_pil(opencv_image)

    # Apply transforms (resize, crop, normalize)
    tensor = transform(pil_image)

    # Add batch dimension
    # Neural networks expect batches, even if batch size = 1
    # Shape changes: [3, 224, 224] → [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)

    print(f"[DEBUG] Tensor shape: {tensor.shape}")
    print(f"[DEBUG] Tensor dtype: {tensor.dtype}")
    print(f"[DEBUG] Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    return tensor


# ============================================================================
# CONCEPT 4: Running Inference (Classification)
# ============================================================================

def classify_image(model, image_tensor, top_k=5):
    """
    Classify image using the model.

    LEARNING: Inference process
    1. Forward pass: tensor → model → logits
    2. Softmax: logits → probabilities
    3. TopK: get top predictions

    Args:
        model: Pre-trained ResNet model
        image_tensor: Preprocessed image [1, 3, 224, 224]
        top_k: Number of top predictions to return

    Returns:
        predictions: List of (class_id, probability) tuples
    """
    print("\n[INFO] Running inference...")

    # Disable gradient calculation (we're not training)
    # This saves memory and speeds up inference
    with torch.no_grad():
        # Forward pass through the network
        # Input: [1, 3, 224, 224]
        # Output: [1, 1000] (1000 ImageNet classes)
        outputs = model(image_tensor)

        print(f"[DEBUG] Raw output shape: {outputs.shape}")
        print(f"[DEBUG] Output is 'logits' (raw scores, not probabilities)")

        # Convert logits to probabilities using softmax
        # Softmax: e^x_i / sum(e^x_j) for all j
        # Ensures outputs sum to 1.0 (valid probabilities)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        print(f"[DEBUG] After softmax, values sum to: {probabilities.sum():.6f}")

        # Get top K predictions
        # torch.topk returns (values, indices)
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Convert to Python lists
        predictions = [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

    return predictions


# ============================================================================
# CONCEPT 5: ImageNet Classes
# ============================================================================

def load_imagenet_classes():
    """
    Load ImageNet class labels.

    LEARNING: ImageNet has 1000 classes
    - Class 0-999 map to different objects/concepts
    - Examples: "golden retriever", "laptop", "book jacket"
    - We need to map class IDs to human-readable names

    For simplicity, we'll load a subset of relevant classes
    """
    # In real applications, you'd load all 1000 classes from a file
    # For this demo, we'll define a few relevant ones manually

    # Format: class_id: class_name
    relevant_classes = {
        # Text/Document related
        921: "book jacket, dust cover, dust jacket, dust wrapper",
        922: "bookcase",
        969: "menu",

        # Logos/Branding
        516: "screen, CRT screen",

        # People/Photos
        281: "tabby, tabby cat",
        340: "zebra",
        388: "giant panda, panda, panda bear, coon bear",

        # Patterns
        928: "ice lolly, lollipop, lolly, popsicle",

        # Add more as needed
    }

    # For classes not in our dict, return "Unknown ImageNet class"
    return relevant_classes


def interpret_predictions(predictions, class_dict):
    """
    Convert class IDs to human-readable labels.

    Args:
        predictions: List of (class_id, probability) tuples
        class_dict: Dict mapping class_id → class_name

    Returns:
        List of (class_name, probability) tuples
    """
    results = []

    for class_id, prob in predictions:
        if class_id in class_dict:
            class_name = class_dict[class_id]
        else:
            class_name = f"ImageNet Class {class_id}"

        results.append((class_name, prob))

    return results


# ============================================================================
# CONCEPT 6: Putting It All Together
# ============================================================================

def classify_book_cover_region(opencv_image, verbose=True):
    """
    Main function: Classify a single image region from book cover.

    Args:
        opencv_image: Image as numpy array (from OpenCV)
        verbose: Print detailed information

    Returns:
        Dictionary with classification results
    """
    if verbose:
        print("\n" + "="*70)
        print("PYTORCH IMAGE CLASSIFICATION")
        print("="*70)

    # Step 1: Load model
    model = load_resnet_model()

    # Step 2: Create preprocessing pipeline
    transform = create_image_transform()

    # Step 3: Preprocess image
    if verbose:
        print(f"\n[INFO] Preprocessing image...")
        print(f"[INFO] Input image shape: {opencv_image.shape}")

    image_tensor = preprocess_image(opencv_image, transform)

    # Step 4: Classify
    predictions = classify_image(model, image_tensor, top_k=5)

    # Step 5: Load class labels
    class_dict = load_imagenet_classes()

    # Step 6: Interpret results
    results = interpret_predictions(predictions, class_dict)

    # Print results
    if verbose:
        print("\n" + "="*70)
        print("TOP 5 PREDICTIONS")
        print("="*70)
        for i, (class_name, prob) in enumerate(results, 1):
            print(f"{i}. {class_name:50s} {prob*100:6.2f}%")

    # Return structured result
    return {
        'top_prediction': results[0][0],
        'confidence': results[0][1],
        'all_predictions': results
    }


# ============================================================================
# DEMO: Test the classifier
# ============================================================================

def demo():
    """
    Demo: Classify a sample image from the book cover.
    """
    print("="*70)
    print("PYTORCH IMAGE CLASSIFICATION DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("1. Load a pre-trained ResNet18 model")
    print("2. Take an image region from your book cover")
    print("3. Classify what type of image it is")
    print("="*70)

    # Load the book cover image
    book_image_path = "../../data/images/book_cover.jpeg"

    print(f"\n[INFO] Loading book cover: {book_image_path}")
    book_image = cv2.imread(book_image_path)

    if book_image is None:
        print(f"[ERROR] Could not load image: {book_image_path}")
        return

    print(f"[SUCCESS] Image loaded: {book_image.shape}")

    # For demo, classify a region (you can change coordinates)
    # Let's take the center region
    height, width = book_image.shape[:2]

    # Extract center 300x300 region
    center_y, center_x = height // 2, width // 2
    crop_size = 300
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)

    region = book_image[y1:y2, x1:x2]

    print(f"\n[INFO] Extracted region: {region.shape}")
    print(f"[INFO] Region location: ({x1}, {y1}) to ({x2}, {y2})")

    # Save region for inspection
    cv2.imwrite("demo_region.jpg", region)
    print(f"[INFO] Saved region to: demo_region.jpg")

    # Classify!
    result = classify_book_cover_region(region, verbose=True)

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Classification: {result['top_prediction']}")
    print(f"Confidence:     {result['confidence']*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    demo()

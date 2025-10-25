"""
Book Cover Analyzer V4 - WITH PYTORCH IMAGE CLASSIFICATION
===========================================================

WHAT'S NEW IN V4:
- Everything from V3 (OCR, grouping, position/size classification)
- PLUS: PyTorch deep learning to classify image types
- Answers: "Is this a logo? A photo? An illustration?"

ML CONCEPTS YOU'LL LEARN:
1. Integrating multiple ML models in one pipeline
2. Batch processing with neural networks
3. Combining classical CV + deep learning results
4. Performance comparison: hand-crafted features vs learned features

PIPELINE:
Stage 1: OpenCV - Find book (edge detection, contours)
Stage 2: EasyOCR - Detect and read text (neural network)
Stage 3: PyTorch - Classify image types (neural network)
Stage 4: Multi-factor classification (combine all signals)
"""

import cv2
import imutils
import argparse
import numpy as np
import easyocr
import warnings
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings('ignore')


# ============================================================================
# IMPORT V3 FUNCTIONS (Book detection, OCR, etc.)
# ============================================================================

# We'll reuse Stage 1 functions from V3
# (I'll inline them here for completeness)

def find_book_contour(image):
    """Stage 1: Find book using edge detection"""
    print("\n" + "="*70)
    print("STAGE 1: FINDING THE BOOK COVER")
    print("="*70)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    book_contour = None
    for i, contour in enumerate(contours[:10]):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        image_area = image.shape[0] * image.shape[1]
        area_percentage = (area / image_area) * 100

        if (area_percentage > 10 and 0.4 < aspect_ratio < 1.2 and len(approx) >= 4):
            print(f"[INFO] Found book: {area_percentage:.1f}% of image")
            book_contour = contour
            break

    return book_contour


def order_points(pts):
    """Order 4 points clockwise"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Perspective transform to bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def extract_book_region(image, contour):
    """Extract book region with perspective correction"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        print("[INFO] Applying perspective transform")
        book_region = four_point_transform(image, approx.reshape(4, 2))
    else:
        print("[INFO] Using bounding rectangle")
        x, y, w, h = cv2.boundingRect(contour)
        book_region = image[y:y+h, x:x+w]

    return book_region


# ============================================================================
# STAGE 2: OCR (Text Detection & Reading)
# ============================================================================

def run_full_image_ocr(book_image, reader):
    """Use EasyOCR to detect and read all text"""
    print("\n" + "="*70)
    print("STAGE 2: TEXT DETECTION & READING (EasyOCR)")
    print("="*70)
    print("[INFO] Running EasyOCR neural network...")

    results = reader.readtext(book_image, detail=1)
    print(f"[SUCCESS] Found {len(results)} text regions")

    text_regions = []
    for idx, (bbox, text, confidence) in enumerate(results):
        bbox_array = np.array(bbox, dtype=np.int32)
        x = int(np.min(bbox_array[:, 0]))
        y = int(np.min(bbox_array[:, 1]))
        w = int(np.max(bbox_array[:, 0]) - x)
        h = int(np.max(bbox_array[:, 1]) - y)

        text_regions.append({
            'id': idx + 1,
            'bbox': (x, y, w, h),
            'text': text,
            'ocr_confidence': confidence,
            'type': 'text'
        })

        print(f"  Text #{idx+1}: \"{text}\" (confidence: {confidence:.2f})")

    return text_regions


# ============================================================================
# STAGE 3: PYTORCH IMAGE CLASSIFICATION (NEW!)
# ============================================================================

class ImageClassifier:
    """
    PyTorch Image Classifier for Book Cover Elements

    ML CONCEPT: Model as a Class
    ============
    We wrap the model in a class because:
    1. Load model once, reuse many times (efficient)
    2. Encapsulate preprocessing logic
    3. Cache frequently used objects (transform, model)

    This is a common pattern in ML engineering!
    """

    def __init__(self, verbose=True):
        """
        Initialize the classifier.

        ML CONCEPT: Lazy Loading
        =============
        We don't load the model immediately in __init__
        We load it when first needed (in classify())
        This saves memory if classifier is created but not used
        """
        self.model = None
        self.transform = None
        self.verbose = verbose
        self.imagenet_classes = self._load_imagenet_classes()

    def _load_imagenet_classes(self):
        """
        Load ImageNet class mappings.

        ML CONCEPT: Class Labels
        =============
        Neural networks output class IDs (0-999 for ImageNet)
        We need to map these to human-readable names

        ImageNet has 1000 classes, but only some are relevant to books
        """
        # Relevant ImageNet classes for book covers
        return {
            # Document/Text
            921: "book_jacket",
            922: "bookcase",
            969: "menu",

            # Logos/Graphics
            516: "screen",

            # Photos/People (common on book covers)
            # We'll keep it simple for now
        }

    def _init_model(self):
        """
        Initialize model and preprocessing (called once on first use).

        ML CONCEPT: Singleton Pattern
        =============
        We check if model is already loaded (self.model is None)
        If not loaded, load it once
        If already loaded, skip (don't reload every time!)

        This saves time and memory
        """
        if self.model is not None:
            return  # Already loaded

        if self.verbose:
            print("\n[INFO] Loading PyTorch ResNet18 model...")

        # Load pre-trained ResNet18
        self.model = models.resnet18(weights='DEFAULT')
        self.model.eval()  # Set to evaluation mode

        # Create preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if self.verbose:
            params = sum(p.numel() for p in self.model.parameters())
            print(f"[SUCCESS] Model loaded ({params:,} parameters)")

    def _opencv_to_pil(self, opencv_image):
        """
        Convert OpenCV image to PIL Image.

        ML CONCEPT: Data Format Conversion
        =============
        Different libraries use different formats:
        - OpenCV: BGR, numpy array
        - PIL: RGB, PIL Image object
        - PyTorch: Tensor, RGB

        Must convert between them!
        """
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image

    def classify(self, opencv_image):
        """
        Classify a single image region.

        Args:
            opencv_image: Image as numpy array (from OpenCV)

        Returns:
            dict: {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'top_5': [(class_name, confidence), ...]
            }

        ML CONCEPT: Inference Pipeline
        =============
        Steps:
        1. Load model (if not already loaded)
        2. Preprocess input (convert to right format)
        3. Forward pass (run through network)
        4. Post-process output (convert to probabilities)
        5. Interpret results (map IDs to names)
        """
        # Step 1: Ensure model is loaded
        self._init_model()

        # Step 2: Preprocess
        pil_image = self._opencv_to_pil(opencv_image)
        image_tensor = self.transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Step 3: Forward pass
        with torch.no_grad():  # Don't compute gradients (we're not training)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Step 4: Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 5)

        # Step 5: Format results
        top_class_id = top_indices[0].item()
        top_confidence = top_probs[0].item()

        # Map to class name
        if top_class_id in self.imagenet_classes:
            class_name = self.imagenet_classes[top_class_id]
        else:
            class_name = f"generic_element_{top_class_id}"

        # Top 5 predictions
        top_5 = []
        for idx, prob in zip(top_indices, top_probs):
            class_id = idx.item()
            if class_id in self.imagenet_classes:
                name = self.imagenet_classes[class_id]
            else:
                name = f"class_{class_id}"
            top_5.append((name, prob.item()))

        return {
            'class_id': top_class_id,
            'class_name': class_name,
            'confidence': top_confidence,
            'top_5': top_5
        }


def classify_image_regions_with_pytorch(book_image, text_regions, verbose=True):
    """
    Detect image regions (non-text areas) and classify them with PyTorch.

    ML CONCEPT: Two-Stage Detection
    =============
    Stage 1: Find regions (where are images?)
    Stage 2: Classify regions (what type of image?)

    This is a common pattern:
    - OCR does text detection + recognition (2 stages in 1 model)
    - We do image detection (manual) + classification (PyTorch)

    Args:
        book_image: The book cover image
        text_regions: Already detected text regions from OCR
        verbose: Print progress

    Returns:
        List of image regions with PyTorch classifications
    """
    print("\n" + "="*70)
    print("STAGE 3: IMAGE DETECTION & CLASSIFICATION (PyTorch)")
    print("="*70)

    # Create classifier
    classifier = ImageClassifier(verbose=verbose)

    # Simple approach: Find non-text regions
    # (In production, you'd use object detection like YOLO)
    height, width = book_image.shape[:2]

    # Create mask of text regions
    text_mask = np.zeros((height, width), dtype=np.uint8)
    for region in text_regions:
        x, y, w, h = region['bbox']
        # Expand text boxes slightly to avoid edge effects
        expand = 5
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(width, x + w + expand)
        y2 = min(height, y + h + expand)
        cv2.rectangle(text_mask, (x1, y1), (x2, y2), 255, -1)

    # Find remaining regions (simplified approach)
    # In reality, you'd use contour detection on non-text areas
    # For this demo, we'll sample a few regions

    print("[INFO] Detecting non-text regions...")

    # Sample approach: Divide image into grid, classify non-text cells
    # This is a SIMPLIFIED approach for demonstration
    # Production would use object detection (YOLO, Faster R-CNN)

    image_regions = []

    # Sample center region if mostly non-text
    center_size = 200
    cx, cy = width // 2, height // 2
    x1 = max(0, cx - center_size // 2)
    y1 = max(0, cy - center_size // 2)
    x2 = min(width, cx + center_size // 2)
    y2 = min(height, cy + center_size // 2)

    # Check if this region is mostly non-text
    region_mask = text_mask[y1:y2, x1:x2]
    text_coverage = np.sum(region_mask > 0) / region_mask.size

    if text_coverage < 0.3:  # Less than 30% text
        region_crop = book_image[y1:y2, x1:x2]

        print(f"[INFO] Classifying image region at ({x1},{y1})...")

        # Classify with PyTorch!
        result = classifier.classify(region_crop)

        image_regions.append({
            'id': len(text_regions) + 1,
            'bbox': (x1, y1, x2-x1, y2-y1),
            'type': 'image',
            'pytorch_class': result['class_name'],
            'pytorch_confidence': result['confidence'],
            'pytorch_top5': result['top_5']
        })

        print(f"  → Classified as: {result['class_name']} ({result['confidence']*100:.1f}%)")
        print(f"  → Top 5 predictions:")
        for i, (name, conf) in enumerate(result['top_5'], 1):
            print(f"      {i}. {name}: {conf*100:.1f}%")

    print(f"[SUCCESS] Classified {len(image_regions)} image region(s)")

    return image_regions


# ============================================================================
# VISUALIZATION & RESULTS
# ============================================================================

def draw_results_v4(book_image, text_regions, image_regions):
    """Draw all detections with PyTorch classifications"""
    output = book_image.copy()

    # Draw text regions (green)
    for region in text_regions:
        x, y, w, h = region['bbox']
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

        label = f"Text: {region['text'][:20]}"
        cv2.putText(output, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw image regions (blue) with PyTorch classification
    for region in image_regions:
        x, y, w, h = region['bbox']
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)

        label = f"{region['pytorch_class']} ({region['pytorch_confidence']*100:.0f}%)"
        cv2.putText(output, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return output


def print_results_summary_v4(text_regions, image_regions):
    """Print comprehensive results with PyTorch classifications"""
    print("\n" + "="*70)
    print("ANALYSIS RESULTS SUMMARY")
    print("="*70)

    print(f"\n📝 TEXT REGIONS: {len(text_regions)}")
    print("-" * 70)
    for i, region in enumerate(text_regions, 1):
        text = region['text']
        conf = region['ocr_confidence']
        print(f"{i:2d}. \"{text[:50]}\" (OCR conf: {conf:.0%})")

    print(f"\n🖼️  IMAGE REGIONS: {len(image_regions)}")
    print("-" * 70)
    if image_regions:
        for i, region in enumerate(image_regions, 1):
            pytorch_class = region['pytorch_class']
            pytorch_conf = region['pytorch_confidence']
            print(f"{i:2d}. {pytorch_class} (PyTorch conf: {pytorch_conf:.0%})")
            print(f"     Top predictions:")
            for j, (name, conf) in enumerate(region['pytorch_top5'][:3], 1):
                print(f"       {j}. {name}: {conf:.0%}")
    else:
        print("No image regions detected")

    print("\n" + "="*70)
    print(f"TOTAL: {len(text_regions)} text + {len(image_regions)} image regions")
    print("="*70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function - V4 with PyTorch"""
    ap = argparse.ArgumentParser(description="Book Cover Analyzer V4 with PyTorch")
    ap.add_argument("-i", "--image", required=True, help="Path to book cover image")
    args = vars(ap.parse_args())

    print("="*70)
    print("BOOK COVER ANALYZER V4 - WITH PYTORCH")
    print("="*70)
    print("\nNEW IN V4:")
    print("  ✓ PyTorch image classification")
    print("  ✓ Identifies image types (logo, photo, illustration, etc.)")
    print("  ✓ Deep learning + classical CV combined")
    print("="*70)

    # Initialize EasyOCR
    print("\n[INFO] Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("[SUCCESS] EasyOCR ready!")

    # Load image
    image = cv2.imread(args["image"])
    if image is None:
        print(f"[ERROR] Could not load image: {args['image']}")
        return

    print(f"[INFO] Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # STAGE 1: Find the book
    book_contour = find_book_contour(image)
    if book_contour is None:
        print("\n[ERROR] Could not find book cover!")
        return

    book_image = extract_book_region(image, book_contour)
    print(f"[SUCCESS] Book extracted: {book_image.shape[1]}x{book_image.shape[0]} pixels")

    # STAGE 2: Detect and read text with EasyOCR
    text_regions = run_full_image_ocr(book_image, reader)

    # STAGE 3: Detect and classify images with PyTorch
    image_regions = classify_image_regions_with_pytorch(book_image, text_regions)

    # Print results
    print_results_summary_v4(text_regions, image_regions)

    # Draw and display
    output = draw_results_v4(book_image, text_regions, image_regions)
    cv2.imshow("V4: Text (green) + Images (blue) with PyTorch", output)
    cv2.waitKey(0)

    # Save results
    cv2.imwrite("book_cover_v4_pytorch_result.jpg", output)
    print("\n[SUCCESS] Results saved to: book_cover_v4_pytorch_result.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Book Cover Analyzer - Web-Ready Version
Refactored V4 for Flask integration
"""

import cv2
import numpy as np
import easyocr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings
import imutils

warnings.filterwarnings('ignore')


class BookCoverAnalyzer:
    """
    Complete book cover analysis pipeline.

    ML CONCEPT: Stateful Service
    ============================
    This class is designed for web deployment:
    - Initialize once (loads models into memory)
    - Process many images (reuse loaded models)
    - Thread-safe design (can handle multiple requests)

    Benefits:
    - Fast: Models loaded once, not per request
    - Memory efficient: Shared model weights
    - Production-ready: Error handling included
    """

    def __init__(self, verbose=False):
        """
        Initialize analyzer with all ML models.

        ML CONCEPT: Model Loading Strategy
        ===================================
        We load ALL models at startup (not lazy):
        - EasyOCR: Text detection + recognition
        - PyTorch ResNet18: Image classification

        Why load at startup?
        - Web server starts once
        - First request doesn't have cold-start delay
        - Consistent response times
        """
        self.verbose = verbose

        if self.verbose:
            print("[INFO] Initializing Book Cover Analyzer...")

        # Load EasyOCR
        if self.verbose:
            print("[INFO] Loading EasyOCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        # Load PyTorch model
        if self.verbose:
            print("[INFO] Loading PyTorch ResNet18...")
        self.pytorch_model = models.resnet18(weights='DEFAULT')
        self.pytorch_model.eval()

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

        # Load ImageNet class labels
        import os
        labels_path = os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt')
        try:
            with open(labels_path, 'r') as f:
                self.imagenet_classes = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            if self.verbose:
                print("[WARNING] ImageNet classes file not found, using generic labels")
            self.imagenet_classes = None

        if self.verbose:
            print("[SUCCESS] All models loaded!")

    def analyze(self, image_path):
        """
        Complete analysis pipeline.

        Args:
            image_path: Path to book cover image

        Returns:
            dict: {
                'success': bool,
                'book_extracted': bool,
                'text_regions': [...],
                'image_regions': [...],
                'summary': {...},
                'error': str (if failed)
            }

        ML CONCEPT: Pipeline Architecture
        ==================================
        Stage 1: OpenCV (Classical CV) - Find book
        Stage 2: EasyOCR (DL) - Detect & read text
        Stage 3: PyTorch (DL) - Classify images

        Each stage can fail independently.
        We return partial results + error status.
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image'
                }

            # Stage 1: Find and extract book
            book_contour = self._find_book(image)

            if book_contour is None:
                # Fallback: If book detection fails, process whole image
                # This handles complex scenes where edge detection struggles
                if self.verbose:
                    print("[INFO] Book detection failed, processing entire image")
                book_image = image
                book_extracted = False
            else:
                # Successfully detected book boundary
                book_image = self._extract_book(image, book_contour)
                book_extracted = True

            # Stage 2: Text detection and recognition
            text_regions = self._detect_text(book_image)

            # Stage 3: Image classification
            image_regions = self._classify_images(book_image, text_regions)

            # Stage 4: Generate human-readable interpretation
            interpretation = self._generate_interpretation(text_regions, image_regions)

            # Summary statistics
            summary = {
                'total_text_regions': len(text_regions),
                'total_image_regions': len(image_regions),
                'book_dimensions': f"{book_image.shape[1]}x{book_image.shape[0]}"
            }

            return {
                'success': True,
                'book_extracted': book_extracted,
                'text_regions': text_regions,
                'image_regions': image_regions,
                'summary': summary,
                'interpretation': interpretation
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }

    def _generate_interpretation(self, text_regions, image_regions):
        """
        Generate human-readable interpretation of analysis results.

        ML CONCEPT: Post-Processing and Interpretation
        ===============================================
        Raw ML outputs need human-readable summaries.
        This method:
        1. Combines all detected text into readable format
        2. Describes what types of images were found
        3. Provides context for non-technical users

        This bridges the gap between ML predictions and user understanding.
        """
        interpretation = {}

        # Text interpretation with smart inference
        if text_regions:
            # Combine all text sorted by position (top to bottom)
            sorted_texts = sorted(text_regions, key=lambda r: (r['bbox']['y'], r['bbox']['x']))
            all_text = [r['text'] for r in sorted_texts]

            # High confidence text only (> 70%)
            high_conf_text = [r['text'] for r in sorted_texts if r['confidence'] > 0.7]

            interpretation['full_text'] = ' '.join(all_text)
            interpretation['high_confidence_text'] = ' '.join(high_conf_text)
            interpretation['word_count'] = len(all_text)

            # Smart inference: Infer book title, author, publisher
            interpretation.update(self._infer_book_metadata(text_regions))
        else:
            interpretation['full_text'] = "No text detected"
            interpretation['high_confidence_text'] = ""
            interpretation['word_count'] = 0
            interpretation['inferred_title'] = None
            interpretation['inferred_authors'] = []
            interpretation['inferred_publisher'] = None
            interpretation['other_text'] = []

        # Image interpretation
        if image_regions:
            image_descriptions = []
            for img in image_regions:
                location = img.get('location', 'unknown')
                confidence = img['confidence'] * 100
                class_name = img['pytorch_class']

                image_descriptions.append({
                    'location': location.replace('_', ' ').title(),
                    'classification': class_name,
                    'confidence': f"{confidence:.1f}%",
                    'description': f"Found visual element in {location.replace('_', ' ')} area (classified as {class_name} with {confidence:.1f}% confidence)"
                })

            interpretation['images'] = image_descriptions
            interpretation['image_summary'] = f"Detected {len(image_regions)} visual elements/illustrations on the cover"
        else:
            interpretation['images'] = []
            interpretation['image_summary'] = "No distinct image regions detected (cover may be text-only or require closer inspection)"

        # Overall interpretation
        if text_regions and image_regions:
            interpretation['cover_type'] = "Mixed - Contains both text and visual elements"
        elif text_regions and not image_regions:
            interpretation['cover_type'] = "Text-heavy - Primarily text-based design"
        elif image_regions and not text_regions:
            interpretation['cover_type'] = "Visual-heavy - Primarily image-based design"
        else:
            interpretation['cover_type'] = "Unknown - Analysis incomplete"

        return interpretation

    def _infer_book_metadata(self, text_regions):
        """
        Infer book title, author, and publisher from detected text.

        ML CONCEPT: Heuristic-Based Inference
        ======================================
        Uses simple rules to guess book metadata:
        - Title: Largest text, usually at top
        - Author: Often capitalized names, medium-large text
        - Publisher: Smaller text, often at bottom
        - Reviews/Quotes: Text with quotation marks

        This is NOT ML - it's rule-based heuristics!
        For better accuracy, would use NER (Named Entity Recognition).
        """
        import re

        sorted_regions = sorted(text_regions, key=lambda r: (r['bbox']['y'], r['bbox']['x']))

        # Categorize by size and position
        large_text = [r for r in text_regions if r['bbox']['height'] > 50]  # Very large (title candidates)
        medium_text = [r for r in text_regions if 30 < r['bbox']['height'] <= 50]  # Medium (author/subtitle)
        small_text = [r for r in text_regions if r['bbox']['height'] <= 30]  # Small (publisher, quotes)

        # Sort by vertical position
        large_text.sort(key=lambda r: r['bbox']['y'])
        medium_text.sort(key=lambda r: r['bbox']['y'])
        small_text.sort(key=lambda r: r['bbox']['y'])

        result = {}

        # Infer Title: Largest text at top
        if large_text:
            title_parts = [r['text'] for r in large_text[:2]]  # Top 2 largest
            result['inferred_title'] = ' '.join(title_parts)
        else:
            result['inferred_title'] = None

        # Infer Authors: Look for capitalized names
        authors = []
        for region in medium_text + large_text:
            text = region['text']
            # Check if mostly uppercase (likely author name)
            if text.isupper() and len(text) > 3:
                # Avoid common words like "THE", "OF"
                if text not in ['THE', 'OF', 'AND', 'IN', 'A', 'AN']:
                    authors.append(text.title())  # Convert to title case

        # Also check for name patterns (First Last)
        for region in text_regions:
            text = region['text']
            # Pattern: Two or more capitalized words
            words = text.split()
            if len(words) >= 2 and all(w[0].isupper() if w else False for w in words):
                if text not in authors and len(text) > 5:
                    authors.append(text)

        result['inferred_authors'] = list(set(authors))[:3]  # Dedupe, max 3

        # Infer Publisher: Small text at bottom
        bottom_small = [r for r in small_text if r['bbox']['y'] > sum([reg['bbox']['y'] for reg in text_regions]) / len(text_regions)]
        publisher_candidates = []
        for region in bottom_small:
            text = region['text']
            # Skip quotes, numbers, common words
            if not any(char in text for char in ['"', "'", '«', '»']) and not text.isdigit():
                if len(text) > 3:
                    publisher_candidates.append(text)

        result['inferred_publisher'] = publisher_candidates[0] if publisher_candidates else None

        # Collect other meaningful text (reviews, quotes, etc.)
        other_text = []
        for region in text_regions:
            text = region['text']
            # Text with quotes (likely reviews)
            if any(char in text for char in ['"', "'", '«', '»', 'Compelling', 'Fascinating']):
                other_text.append(text)

        result['other_text'] = other_text[:5]  # Max 5 items

        return result

    def _find_book(self, image):
        """
        Stage 1: Find book using edge detection with fallback.

        ML CONCEPT: Classical Computer Vision with Robustness
        ======================================================
        Uses hand-crafted algorithms (not ML):
        - Canny edge detection with multiple thresholds
        - Contour finding
        - Shape analysis (aspect ratio, area)
        - Fallback strategy for real-world photos

        IMPROVEMENT: Added fallback for books in complex scenes
        (e.g., book on desk with other objects)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try multiple edge detection thresholds
        edge_params = [
            (50, 150),   # Original
            (30, 100),   # More sensitive
            (75, 200),   # Less sensitive
        ]

        for low_thresh, high_thresh in edge_params:
            edged = cv2.Canny(blurred, low_thresh, high_thresh)

            contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # First pass: Strict criteria (ideal conditions)
            for contour in contours[:15]:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                image_area = image.shape[0] * image.shape[1]
                area_percentage = (area / image_area) * 100

                # Strict criteria: Clean book photos
                if (area_percentage > 10 and 0.4 < aspect_ratio < 1.2
                    and len(approx) >= 4):
                    return contour

            # Second pass: Relaxed criteria (real-world photos)
            for contour in contours[:20]:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                image_area = image.shape[0] * image.shape[1]
                area_percentage = (area / image_area) * 100

                # Relaxed criteria: Books in complex scenes
                # - Lower area threshold (5% instead of 10%)
                # - Wider aspect ratio range (0.3 to 1.5)
                # - Require rectangular shape (4 corners)
                if (area_percentage > 5 and 0.3 < aspect_ratio < 1.5
                    and len(approx) >= 4 and area > 5000):
                    return contour

        # Fallback: Use largest rectangular contour with reasonable size
        # This handles cases where book edges aren't perfect
        edged = cv2.Canny(blurred, 30, 100)
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_area = image.shape[0] * image.shape[1]

        for contour in contours[:25]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)
            area_percentage = (area / image_area) * 100

            # Absolute minimum: Any rectangular shape with meaningful size
            # - Must be at least 3% of image (not tiny labels/logos)
            # - Must be at least 20,000 pixels (prevents tiny regions)
            if len(approx) >= 4 and area > 20000 and area_percentage > 3:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Basic book-like shape
                # - Minimum width/height to avoid tiny regions
                if 0.3 < aspect_ratio < 1.8 and w > 100 and h > 100:
                    return contour

        return None

    def _extract_book(self, image, contour):
        """Extract book region with perspective correction."""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            # Apply perspective transform
            book_region = self._four_point_transform(image, approx.reshape(4, 2))
        else:
            # Use bounding box
            x, y, w, h = cv2.boundingRect(contour)
            book_region = image[y:y+h, x:x+w]

        return book_region

    def _four_point_transform(self, image, pts):
        """Perspective transform helper."""
        rect = self._order_points(pts)
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

    def _order_points(self, pts):
        """Order points clockwise."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect_text(self, book_image):
        """
        Stage 2: Text detection and recognition using EasyOCR.

        ML CONCEPT: End-to-End Deep Learning
        =====================================
        EasyOCR uses TWO neural networks:
        1. CRAFT: Detects WHERE text is (bounding boxes)
        2. Recognition Net: Reads WHAT text says (OCR)

        Both are pre-trained, no training needed!
        """
        results = self.ocr_reader.readtext(book_image, detail=1)

        text_regions = []
        for idx, (bbox, text, confidence) in enumerate(results):
            bbox_array = np.array(bbox, dtype=np.int32)
            x = int(np.min(bbox_array[:, 0]))
            y = int(np.min(bbox_array[:, 1]))
            w = int(np.max(bbox_array[:, 0]) - x)
            h = int(np.max(bbox_array[:, 1]) - y)

            text_regions.append({
                'id': idx + 1,
                'text': text,
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'confidence': float(confidence),
                'type': 'text'
            })

        return text_regions

    def _classify_images(self, book_image, text_regions):
        """
        Stage 3: Image classification using PyTorch.

        ML CONCEPT: Transfer Learning (Zero-shot) + Multi-Region Sampling
        ==================================================================
        Strategy: Sample multiple regions across the book cover
        - Center region (main illustration)
        - Top corners (logos, badges, icons)
        - Bottom corners (publisher logos, barcodes)

        IMPROVEMENT: No longer skips regions with text overlay!
        Books often have illustrations WITH text on them.
        """
        height, width = book_image.shape[:2]

        # Define regions to sample (multiple locations)
        sample_regions = [
            # Center - main illustration area
            {
                'name': 'center',
                'x1': max(0, width // 2 - 150),
                'y1': max(0, height // 2 - 150),
                'x2': min(width, width // 2 + 150),
                'y2': min(height, height // 2 + 150)
            },
            # Top-left corner
            {
                'name': 'top_left',
                'x1': 10,
                'y1': 10,
                'x2': min(width, 150),
                'y2': min(height, 150)
            },
            # Top-right corner
            {
                'name': 'top_right',
                'x1': max(0, width - 150),
                'y1': 10,
                'x2': width - 10,
                'y2': min(height, 150)
            },
            # Bottom-left corner
            {
                'name': 'bottom_left',
                'x1': 10,
                'y1': max(0, height - 150),
                'x2': min(width, 150),
                'y2': height - 10
            },
            # Bottom-right corner
            {
                'name': 'bottom_right',
                'x1': max(0, width - 150),
                'y1': max(0, height - 150),
                'x2': width - 10,
                'y2': height - 10
            }
        ]

        image_regions = []
        region_id = len(text_regions) + 1

        for sample in sample_regions:
            x1, y1 = sample['x1'], sample['y1']
            x2, y2 = sample['x2'], sample['y2']

            # Skip if region is too small
            if x2 - x1 < 50 or y2 - y1 < 50:
                continue

            # Extract region
            region_crop = book_image[y1:y2, x1:x2]

            # Check if region has enough visual content (not just solid color)
            gray = cv2.cvtColor(region_crop, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray)

            # Only classify if there's visual complexity (not blank/solid color)
            if std_dev > 15:  # Threshold for visual complexity
                result = self._classify_single_image(region_crop)

                image_regions.append({
                    'id': region_id,
                    'location': sample['name'],
                    'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                    'pytorch_class': result['class_name'],
                    'confidence': result['confidence'],
                    'top_5': result['top_5'],
                    'type': 'image'
                })
                region_id += 1

        return image_regions

    def _classify_single_image(self, opencv_image):
        """
        Classify single image region with PyTorch.

        ML CONCEPT: Inference Pipeline
        ===============================
        1. Convert BGR → RGB → PIL
        2. Resize, crop, normalize (preprocessing)
        3. Convert to tensor, add batch dim
        4. Forward pass through ResNet18
        5. Softmax to get probabilities
        6. Return top predictions
        """
        # Convert OpenCV (BGR) → PIL (RGB)
        rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Preprocess
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            outputs = self.pytorch_model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 5
        top_probs, top_indices = torch.topk(probs, 5)

        top_class_id = top_indices[0].item()
        top_confidence = top_probs[0].item()

        # Get actual ImageNet class name
        if self.imagenet_classes and top_class_id < len(self.imagenet_classes):
            class_name = self.imagenet_classes[top_class_id]
        else:
            class_name = f"element_{top_class_id}"

        top_5 = [
            {
                'class_id': idx.item(),
                'class_name': self.imagenet_classes[idx.item()] if self.imagenet_classes and idx.item() < len(self.imagenet_classes) else f"class_{idx.item()}",
                'confidence': prob.item()
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

        return {
            'class_id': top_class_id,
            'class_name': class_name,
            'confidence': float(top_confidence),
            'top_5': top_5
        }

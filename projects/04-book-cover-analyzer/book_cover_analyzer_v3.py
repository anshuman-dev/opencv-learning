"""
Book Cover Analyzer V3 - IMPROVED WITH FULL-IMAGE OCR
======================================================
WHAT'S NEW IN V3:
- Uses full-image OCR (no manual region detection!)
- Groups nearby text boxes (fixes "T,E,A,M" problem)
- Position-based classification (top=title, middle=author, bottom=publisher)
- Size-based classification (largest=most important)
- Multi-factor scoring system
- Pattern matching for author names and ISBN
- Confidence scores for all classifications

LEARNING OBJECTIVES:
1. End-to-end OCR approach
2. Text region grouping algorithms
3. Spatial heuristics (position, size, centrality)
4. Multi-factor scoring
5. Pattern matching with regex
6. Decision fusion

COMPARISON TO V2:
- V2: Edge detection → Contour detection → OCR (3 stages, splits words)
- V3: Edge detection → Full OCR (2 stages, keeps words together)
"""

import cv2
import imutils
import argparse
import numpy as np
import easyocr
import warnings
import re
warnings.filterwarnings('ignore')


# ============================================================================
# STAGE 1: FIND THE BOOK (SAME AS BEFORE)
# ============================================================================

def find_book_contour(image):
    """
    Find the book cover in the image by detecting the largest rectangular region.
    This is the same as V2 - no changes needed here.
    """
    print("\n" + "="*70)
    print("STAGE 1: FINDING THE BOOK COVER")
    print("="*70)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cv2.imshow("Stage 1: Edge Detection", edged)
    cv2.waitKey(0)

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    print(f"[INFO] Found {len(contours)} contours")

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
            print(f"  → Found book! Contour #{i+1}, area={area_percentage:.1f}%")
            book_contour = contour
            break

    return book_contour


def order_points(pts):
    """Order points clockwise: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Apply perspective transform to get top-down view"""
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
    """Extract the book region using perspective transform or bounding box"""
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
# STAGE 2: FULL-IMAGE OCR (NEW APPROACH!)
# ============================================================================

def run_full_image_ocr(book_image, reader):
    """
    NEW APPROACH: Run OCR on the entire book cover at once.
    EasyOCR will find text regions AND read them in one step.

    This is simpler and better than manual region detection because:
    - EasyOCR's neural network is trained on millions of images
    - It naturally groups letters into words
    - No "T,E,A,M" splitting problem!

    LEARNING: End-to-end deep learning vs pipeline approaches
    """
    print("\n" + "="*70)
    print("STAGE 2: FULL-IMAGE OCR (V3 NEW APPROACH)")
    print("="*70)
    print("[INFO] Running EasyOCR on entire book cover...")
    print("[INFO] EasyOCR will detect AND read text in one step")

    # Run OCR on whole image
    # Returns: [[bbox, text, confidence], ...]
    # bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    results = reader.readtext(book_image, detail=1)

    print(f"[SUCCESS] Found {len(results)} text regions")

    # Convert to our format for easier processing
    text_regions = []
    for idx, (bbox, text, confidence) in enumerate(results):
        # Convert bbox to x, y, w, h format
        bbox_array = np.array(bbox, dtype=np.int32)
        x = int(np.min(bbox_array[:, 0]))
        y = int(np.min(bbox_array[:, 1]))
        w = int(np.max(bbox_array[:, 0]) - x)
        h = int(np.max(bbox_array[:, 1]) - y)

        text_regions.append({
            'id': idx + 1,
            'bbox': (x, y, w, h),
            'bbox_polygon': bbox,
            'text': text,
            'ocr_confidence': confidence,
            'area': w * h,
            'center_x': x + w // 2,
            'center_y': y + h // 2,
            'aspect_ratio': w / h if h > 0 else 0
        })

        print(f"  Region #{idx+1}: \"{text}\" (confidence: {confidence:.2f})")

    return text_regions


# ============================================================================
# STEP 2: TEXT REGION GROUPING (FIX T,E,A,M PROBLEM)
# ============================================================================

def group_nearby_text_regions(text_regions, image_dims):
    """
    Group nearby text boxes that likely belong together.

    Criteria for grouping:
    1. On same horizontal line (similar Y coordinate)
    2. Close together horizontally (small gap)
    3. Similar height (same font size)

    LEARNING: Spatial proximity algorithms, clustering basics

    Example:
    Before: ["T", "E", "A", "M"] as 4 separate regions
    After: ["TEAM"] as 1 merged region
    """
    print("\n" + "="*70)
    print("STEP 2: GROUPING NEARBY TEXT REGIONS")
    print("="*70)
    print("[INFO] Merging text boxes that belong together...")

    if not text_regions:
        return []

    # Sort by Y position (top to bottom), then X position (left to right)
    sorted_regions = sorted(text_regions, key=lambda r: (r['center_y'], r['center_x']))

    grouped_regions = []
    current_group = [sorted_regions[0]]

    for i in range(1, len(sorted_regions)):
        current = sorted_regions[i]
        previous = sorted_regions[i-1]

        # Check if current region belongs to the same group
        y_distance = abs(current['center_y'] - previous['center_y'])
        x_distance = current['bbox'][0] - (previous['bbox'][0] + previous['bbox'][2])
        height_diff = abs(current['bbox'][3] - previous['bbox'][3])

        # Thresholds for grouping
        same_line = y_distance < previous['bbox'][3] * 0.5  # Within 50% of height
        close_together = x_distance < previous['bbox'][2] * 1.5  # Gap < 1.5x width
        similar_height = height_diff < previous['bbox'][3] * 0.3  # Height diff < 30%

        if same_line and close_together and similar_height:
            # Add to current group
            current_group.append(current)
        else:
            # Start new group
            grouped_regions.append(current_group)
            current_group = [current]

    # Don't forget the last group
    grouped_regions.append(current_group)

    # Merge regions in each group
    merged_regions = []
    for group in grouped_regions:
        if len(group) == 1:
            # Single region, no merging needed
            merged_regions.append(group[0])
        else:
            # Merge multiple regions
            merged = merge_text_group(group)
            merged_regions.append(merged)

            original_texts = [r['text'] for r in group]
            print(f"  Merged: {original_texts} → \"{merged['text']}\"")

    print(f"[SUCCESS] Reduced from {len(text_regions)} to {len(merged_regions)} regions")

    return merged_regions


def merge_text_group(group):
    """
    Merge a group of text regions into one.

    LEARNING: Bounding box union, text concatenation
    """
    # Find bounding box that contains all regions
    min_x = min(r['bbox'][0] for r in group)
    min_y = min(r['bbox'][1] for r in group)
    max_x = max(r['bbox'][0] + r['bbox'][2] for r in group)
    max_y = max(r['bbox'][1] + r['bbox'][3] for r in group)

    w = max_x - min_x
    h = max_y - min_y

    # Concatenate text (left to right order)
    sorted_group = sorted(group, key=lambda r: r['bbox'][0])
    merged_text = ' '.join(r['text'] for r in sorted_group)

    # Average confidence
    avg_confidence = sum(r['ocr_confidence'] for r in group) / len(group)

    return {
        'id': group[0]['id'],  # Keep first ID
        'bbox': (min_x, min_y, w, h),
        'text': merged_text,
        'ocr_confidence': avg_confidence,
        'area': w * h,
        'center_x': min_x + w // 2,
        'center_y': min_y + h // 2,
        'aspect_ratio': w / h if h > 0 else 0,
        'merged_from': [r['text'] for r in group]
    }


# ============================================================================
# STEP 3: POSITION-BASED CLASSIFICATION
# ============================================================================

def classify_by_position(region, image_height):
    """
    Classify text based on WHERE it's located on the book cover.

    Layout zones:
    - Top 30%: Usually title, subtitle
    - Middle 40%: Usually author, series info
    - Bottom 30%: Usually publisher, ISBN, edition info

    LEARNING: Layout analysis, zone-based classification
    """
    y_ratio = region['center_y'] / image_height

    if y_ratio < 0.3:
        return "top_zone", 0.6
    elif y_ratio < 0.7:
        return "middle_zone", 0.5
    else:
        return "bottom_zone", 0.5


def classify_by_size(region, all_regions):
    """
    Classify based on text size (area).
    Larger text is usually more important.

    LEARNING: Visual hierarchy, relative sizing
    """
    # Sort all regions by area
    sorted_by_area = sorted(all_regions, key=lambda r: r['area'], reverse=True)

    # Find rank of current region
    rank = next(i for i, r in enumerate(sorted_by_area) if r['id'] == region['id'])

    # Convert rank to score (0.0 - 1.0)
    total_regions = len(all_regions)
    size_score = 1.0 - (rank / total_regions)

    return size_score


def calculate_centrality_score(region, image_width):
    """
    Calculate how centered the text is horizontally.
    Titles are often centered.

    LEARNING: Center alignment detection, distance normalization
    """
    region_center_x = region['center_x']
    image_center_x = image_width / 2

    distance_from_center = abs(region_center_x - image_center_x)
    max_distance = image_width / 2

    # Normalize: 1.0 = perfectly centered, 0.0 = at edge
    centrality_score = 1.0 - (distance_from_center / max_distance)

    return centrality_score


# ============================================================================
# STEP 4: PATTERN MATCHING
# ============================================================================

def detect_author_pattern(text):
    """
    Detect author name patterns using regex.

    Common patterns:
    - "by FirstName LastName"
    - "FirstName LastName"
    - "FirstName MiddleInitial. LastName"
    - "F. LastName"

    LEARNING: Regular expressions, text pattern matching
    """
    # Clean text
    text_clean = text.strip()

    # Pattern 1: "by John Smith"
    if re.match(r'^by\s+[A-Z][a-z]+\s+[A-Z][a-z]+', text_clean, re.IGNORECASE):
        return True, 0.95

    # Pattern 2: "John Smith" (capitalized words, 2-4 words)
    if re.match(r'^[A-Z][a-z]+(\s+[A-Z]\.)?(\s+[A-Z][a-z]+){1,2}$', text_clean):
        # But exclude common title words
        title_words = ['The', 'A', 'An', 'And', 'Of', 'In', 'On', 'To', 'For']
        words = text_clean.split()
        if not any(word in title_words for word in words):
            return True, 0.75

    # Pattern 3: "J. Smith" or "John Q. Smith"
    if re.match(r'^[A-Z]\.\s+[A-Z][a-z]+$', text_clean):
        return True, 0.80

    return False, 0.0


def detect_metadata_pattern(text):
    """
    Detect ISBN, year, edition, publisher patterns.

    LEARNING: Structured data extraction, metadata patterns
    """
    text_clean = text.strip()

    # ISBN-13: 978-0-123-45678-9 or variations
    if re.search(r'ISBN[-:]?\s*978[-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d', text_clean, re.IGNORECASE):
        return "ISBN-13", 0.95

    # ISBN-10
    if re.search(r'ISBN[-:]?\s*\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d', text_clean, re.IGNORECASE):
        return "ISBN-10", 0.95

    # Year patterns
    if re.search(r'(Copyright|©|Published)?\s*(19|20)\d{2}', text_clean):
        return "Year", 0.85

    # Edition
    if re.search(r'\d+(st|nd|rd|th)\s+Edition', text_clean, re.IGNORECASE):
        return "Edition", 0.90

    # Publisher indicators
    if re.search(r'(Press|Publishing|Publisher|Books)', text_clean, re.IGNORECASE):
        return "Publisher", 0.70

    return None, 0.0


# ============================================================================
# STEP 5: MULTI-FACTOR SCORING & FINAL CLASSIFICATION
# ============================================================================

def calculate_title_score(region, all_regions, image_dims):
    """
    Calculate likelihood that this region is the TITLE.

    Combines multiple signals:
    - Position (top of page)
    - Size (largest text)
    - Centrality (centered horizontally)

    LEARNING: Multi-factor scoring, weighted averaging
    """
    # Individual scores
    position_zone, pos_conf = classify_by_position(region, image_dims[0])
    size_score = classify_by_size(region, all_regions)
    centrality_score = calculate_centrality_score(region, image_dims[1])

    # Weights (must sum to 1.0)
    position_weight = 0.35
    size_weight = 0.40
    centrality_weight = 0.25

    # Position score (top zone = higher score)
    if position_zone == "top_zone":
        position_score = 0.9
    elif position_zone == "middle_zone":
        position_score = 0.3
    else:
        position_score = 0.1

    # Weighted combination
    title_score = (
        position_score * position_weight +
        size_score * size_weight +
        centrality_score * centrality_weight
    )

    return title_score


def final_classification(region, all_regions, image_dims):
    """
    Make final decision on what this text region represents.

    Priority order:
    1. Strong pattern match (ISBN, author name)
    2. Multi-factor score for title
    3. Position-based classification

    LEARNING: Decision fusion, priority-based classification
    """
    # Check for strong pattern matches first
    is_author, author_conf = detect_author_pattern(region['text'])
    metadata_type, meta_conf = detect_metadata_pattern(region['text'])

    # Priority 1: Metadata (ISBN, year, etc.) - very high confidence
    if meta_conf > 0.85:
        return metadata_type, meta_conf

    # Priority 2: Author name - high confidence
    if author_conf > 0.75:
        return "Author", author_conf

    # Priority 3: Title detection using multi-factor score
    title_score = calculate_title_score(region, all_regions, image_dims)

    # Priority 4: Position-based classification
    position_zone, pos_conf = classify_by_position(region, image_dims[0])

    # Decision logic
    if title_score > 0.65:
        return "Title", title_score

    if author_conf > 0.5:  # Weaker author pattern
        return "Author (low confidence)", author_conf

    if position_zone == "top_zone" and title_score > 0.4:
        return "Subtitle", 0.6

    if position_zone == "middle_zone":
        return "Series/Info", 0.5

    if position_zone == "bottom_zone":
        return "Publisher/Metadata", 0.5

    return "Other", 0.3


def classify_all_regions(text_regions, image_dims):
    """
    Classify all text regions with confidence scores.

    LEARNING: Applying classification pipeline to dataset
    """
    print("\n" + "="*70)
    print("STEP 3-5: MULTI-FACTOR CLASSIFICATION")
    print("="*70)
    print("[INFO] Classifying text regions...")

    for region in text_regions:
        classification, confidence = final_classification(region, text_regions, image_dims)
        region['classification'] = classification
        region['classification_confidence'] = confidence

        print(f"  \"{region['text'][:40]}\" → {classification} (conf: {confidence:.2f})")

    return text_regions


# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def draw_results_v3(book_image, classified_regions):
    """Draw bounding boxes with classifications and confidence scores"""
    output = book_image.copy()

    # Color map for different classifications
    color_map = {
        'Title': (0, 255, 0),           # Green
        'Subtitle': (0, 200, 0),        # Dark green
        'Author': (255, 0, 0),          # Blue
        'ISBN-13': (0, 0, 255),         # Red
        'ISBN-10': (0, 0, 255),         # Red
        'Year': (255, 0, 255),          # Magenta
        'Edition': (255, 100, 0),       # Orange
        'Publisher': (200, 200, 0),     # Cyan
        'Series/Info': (150, 150, 150), # Gray
        'Other': (100, 100, 100)        # Dark gray
    }

    for region in classified_regions:
        x, y, w, h = region['bbox']
        classification = region['classification']
        confidence = region['classification_confidence']

        # Get color (default to white if not in map)
        color = color_map.get(classification, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        label = f"{classification} ({confidence:.0%})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x, y - label_h - 8), (x + label_w + 4, y), color, -1)

        # Draw label text
        cv2.putText(output, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output


def print_results_summary(classified_regions):
    """Print organized summary of detected text"""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    # Group by classification
    by_class = {}
    for region in classified_regions:
        classification = region['classification']
        if classification not in by_class:
            by_class[classification] = []
        by_class[classification].append(region)

    # Print each category
    for classification in ['Title', 'Subtitle', 'Author', 'ISBN-13', 'ISBN-10',
                          'Year', 'Edition', 'Publisher', 'Series/Info', 'Other']:
        if classification in by_class:
            print(f"\n📌 {classification.upper()}:")
            for region in by_class[classification]:
                conf = region['classification_confidence']
                text = region['text']
                print(f"   \"{text}\" (confidence: {conf:.0%})")


def main():
    """Main function - Book Cover Analyzer V3"""
    ap = argparse.ArgumentParser(description="Book Cover Analyzer V3 - Improved")
    ap.add_argument("-i", "--image", required=True, help="Path to book cover image")
    args = vars(ap.parse_args())

    print("="*70)
    print("BOOK COVER ANALYZER V3 - IMPROVED")
    print("="*70)
    print("\nIMPROVEMENTS OVER V2:")
    print("  ✓ Full-image OCR (no manual region detection)")
    print("  ✓ Text grouping (fixes 'T,E,A,M' problem)")
    print("  ✓ Position + size + centrality scoring")
    print("  ✓ Pattern matching (author names, ISBN)")
    print("  ✓ Confidence scores for all classifications")
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
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # STAGE 1: Find the book
    book_contour = find_book_contour(image)
    if book_contour is None:
        print("\n[ERROR] Could not find book cover!")
        cv2.destroyAllWindows()
        return

    book_image = extract_book_region(image, book_contour)
    print(f"[SUCCESS] Book extracted: {book_image.shape[1]}x{book_image.shape[0]} pixels")

    cv2.imshow("Extracted Book Cover", book_image)
    cv2.waitKey(0)

    # STAGE 2: Full-image OCR
    text_regions = run_full_image_ocr(book_image, reader)

    # STEP 2: Group nearby text
    grouped_regions = group_nearby_text_regions(text_regions,
                                                 (book_image.shape[0], book_image.shape[1]))

    # STEPS 3-5: Classify regions
    classified_regions = classify_all_regions(grouped_regions,
                                              (book_image.shape[0], book_image.shape[1]))

    # Print results
    print_results_summary(classified_regions)

    # Draw and display results
    output = draw_results_v3(book_image, classified_regions)
    cv2.imshow("V3: Classified Book Cover", output)
    cv2.waitKey(0)

    # Save results
    cv2.imwrite("book_cover_v3_result.jpg", output)
    print("\n[SUCCESS] Results saved to: book_cover_v3_result.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

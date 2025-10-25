"""
Book Cover Analyzer - PRODUCTION VERSION
=========================================
Two-stage detection for real-world scenarios:
Stage 1: Find and isolate the book cover from background
Stage 2: Detect text and image elements ON the book cover

This approach works regardless of background (table, desk, etc.)

CONCEPTS USED:
- Edge detection (Canny)
- Contour detection and filtering
- Shape analysis (aspect ratio, area)
- Perspective transformation (optional)
- Adaptive thresholding
- Morphological operations
- Color space analysis
"""

import cv2
import imutils
import argparse
import numpy as np


def find_book_contour(image):
    """
    Stage 1: Find the book cover in the image by detecting the largest
    light-colored rectangular region.

    Args:
        image: Original BGR image

    Returns:
        Contour of the book, or None if not found
    """
    print("\n" + "="*60)
    print("STAGE 1: FINDING THE BOOK COVER")
    print("="*60)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    # We want to find the book's outline against the background
    edged = cv2.Canny(blurred, 50, 150)

    # Show edge detection result
    cv2.imshow("Stage 1a: Edge Detection (Finding Book Outline)", edged)
    cv2.waitKey(0)

    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    print(f"[INFO] Found {len(contours)} contours")

    # Try to find a book-shaped contour
    book_contour = None

    # Visualize candidate contours
    debug_image = image.copy()

    for i, contour in enumerate(contours[:10]):  # Check top 10 largest
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        area = cv2.contourArea(contour)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Book characteristics:
        # 1. Should be a large region (at least 10% of image)
        # 2. Aspect ratio between 0.5 and 1.5 (typical book cover)
        # 3. Should have 4 corners (approximately rectangular)

        image_area = image.shape[0] * image.shape[1]
        area_percentage = (area / image_area) * 100

        print(f"  Contour #{i+1}: area={area_percentage:.1f}%, aspect_ratio={aspect_ratio:.2f}, corners={len(approx)}")

        # Draw this contour for debugging
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        cv2.drawContours(debug_image, [contour], -1, color, 3)

        # Check if this looks like a book
        if (area_percentage > 10 and
            0.4 < aspect_ratio < 1.2 and
            len(approx) >= 4):

            print(f"  → This looks like a book! Using contour #{i+1}")
            book_contour = contour
            break

    cv2.imshow("Stage 1b: Candidate Contours (Green=Selected)", debug_image)
    cv2.waitKey(0)

    return book_contour


def order_points(pts):
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left

    Args:
        pts: Array of 4 points

    Returns:
        Ordered array of points
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left point has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transform to get a top-down view of the book.

    Args:
        image: Original image
        pts: 4 corner points of the book

    Returns:
        Warped image (bird's eye view)
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Calculate perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def extract_book_region(image, contour):
    """
    Extract the book region from the image using the contour.
    Applies perspective transform if needed.

    Args:
        image: Original image
        contour: Book contour

    Returns:
        Cropped/warped book image
    """
    # Approximate contour to polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # If we have 4 points, apply perspective transform
    if len(approx) == 4:
        print("[INFO] Applying perspective transform (4 corners detected)")
        book_region = four_point_transform(image, approx.reshape(4, 2))
    else:
        # Otherwise, just use bounding rectangle
        print("[INFO] Using bounding rectangle (non-rectangular contour)")
        x, y, w, h = cv2.boundingRect(contour)
        book_region = image[y:y+h, x:x+w]

    return book_region


def analyze_book_elements(book_image):
    """
    Stage 2: Analyze elements ON the book cover.
    Detects text regions and images.

    Args:
        book_image: Cropped image of just the book cover

    Returns:
        List of detected regions with classifications
    """
    print("\n" + "="*60)
    print("STAGE 2: ANALYZING ELEMENTS ON THE BOOK COVER")
    print("="*60)

    height, width = book_image.shape[:2]
    total_area = height * width

    # Convert to grayscale
    gray = cv2.cvtColor(book_image, cv2.COLOR_BGR2GRAY)

    # Apply slight blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    cv2.imshow("Stage 2a: Book Cover (Isolated)", book_image)
    cv2.waitKey(0)

    # Use adaptive thresholding to find dark elements (text/images)
    print("[INFO] Applying adaptive thresholding to detect text and images...")
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,  # Block size (larger for book covers)
        3    # Constant
    )

    cv2.imshow("Stage 2b: Adaptive Threshold", thresh)
    cv2.waitKey(0)

    # Morphological operations to connect text and clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.dilate(thresh, kernel, iterations=2)
    morph = cv2.erode(morph, kernel, iterations=1)

    cv2.imshow("Stage 2c: Morphological Cleanup", morph)
    cv2.waitKey(0)

    # ========================================
    # NEW: Find contours WITH hierarchy
    # ========================================
    # RETR_TREE captures parent-child relationships (holes in letters)
    # This helps distinguish text (has holes) from images (no holes)
    print("[INFO] Finding contours with hierarchy analysis...")
    contours, hierarchy = cv2.findContours(morph.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"[INFO] Found {len(contours)} contours on book cover")

    # Filter and classify regions
    regions = []
    min_area = total_area * 0.001  # 0.1% of book area
    max_area = total_area * 0.5    # 50% of book area

    # Process each contour
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        percentage = (area / total_area) * 100

        # ========================================
        # NEW: Analyze contour hierarchy
        # ========================================
        # Hierarchy format: [Next, Previous, First_Child, Parent]
        # If First_Child != -1, this contour has holes inside it
        # Text often has holes (O, A, D, etc.), images usually don't

        has_holes = False
        hole_count = 0

        if hierarchy is not None:
            # Get hierarchy info for this contour
            h_info = hierarchy[0][idx]
            first_child = h_info[2]  # Index of first child contour

            # Count how many children (holes) this contour has
            if first_child != -1:
                has_holes = True
                child_idx = first_child
                while child_idx != -1:
                    hole_count += 1
                    child_idx = hierarchy[0][child_idx][0]  # Next sibling

        # ========================================
        # NEW: Analyze color variance
        # ========================================
        # Extract the region from the original book image
        region_crop = book_image[y:y+h, x:x+w]

        # Calculate color statistics
        # 1. Standard deviation of pixel intensities (how varied the colors are)
        color_std = np.std(region_crop)

        # 2. Convert to HSV and check saturation (colorfulness)
        region_hsv = cv2.cvtColor(region_crop, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(region_hsv[:, :, 1])  # S channel

        # 3. Check if it's monochrome (text) or colorful (image)
        # Text: low saturation (<30), low color variance (<40)
        # Images: high saturation (>30) or high color variance (>40)
        is_monochrome = (saturation_mean < 30 and color_std < 40)

        # Calculate "text likelihood score" based on multiple factors
        # More holes + monochrome = more likely to be text
        text_score = hole_count * 2  # Holes are strong indicator
        if is_monochrome:
            text_score += 3  # Monochrome adds to text score
        if aspect_ratio > 2:
            text_score += 2  # Wide rectangles are usually text

        # ========================================
        # IMPROVED: Classification with hierarchy + color
        # ========================================
        # Now we use: aspect_ratio + has_holes + hole_count + color_variance

        # Use text_score for better classification
        # Higher score = more likely text
        # Score components:
        #   holes: 2 points each
        #   monochrome: 3 points
        #   wide rectangle: 2 points

        if percentage > 10:
            # Large regions - use text score
            if text_score >= 5 or aspect_ratio > 3:
                classification = "Large Text Block"
                color = (0, 0, 255)  # Red
            else:
                classification = "Large Image"
                color = (200, 0, 200)  # Purple
        elif percentage > 2:
            # Medium regions - use text score + aspect ratio
            if text_score >= 5:
                classification = "Text Block (high text score)"
                color = (0, 255, 0)  # Green
            elif aspect_ratio > 3:
                classification = "Text Line (Horizontal)"
                color = (255, 165, 0)  # Orange
            elif is_monochrome and aspect_ratio > 1.5:
                # Monochrome and somewhat horizontal = likely text
                classification = "Text (monochrome)"
                color = (0, 200, 0)  # Dark Green
            elif 0.8 < aspect_ratio < 1.2 and not is_monochrome:
                # Square and colorful = likely image
                classification = "Image/Icon (colorful)"
                color = (0, 255, 255)  # Cyan
            else:
                classification = "Text/Image (ambiguous)"
                color = (128, 128, 128)  # Gray
        else:
            # Small regions - use text score
            if text_score >= 3:
                classification = "Small Text (text score)"
                color = (255, 0, 255)  # Magenta
            elif aspect_ratio > 2:
                classification = "Small Text (wide)"
                color = (255, 100, 255)  # Light Magenta
            elif is_monochrome:
                classification = "Small Text/Icon (mono)"
                color = (200, 0, 200)  # Purple
            else:
                classification = "Small Icon/Element"
                color = (255, 0, 0)  # Blue

        regions.append({
            'bbox': (x, y, w, h),
            'area': area,
            'percentage': percentage,
            'aspect_ratio': aspect_ratio,
            'has_holes': has_holes,
            'hole_count': hole_count,
            'text_score': text_score,
            'color_std': color_std,
            'saturation': saturation_mean,
            'is_monochrome': is_monochrome,
            'classification': classification,
            'color': color
        })

    # Sort by area
    regions.sort(key=lambda x: x['area'], reverse=True)

    print(f"[INFO] {len(regions)} significant regions detected")

    return regions


def save_detected_regions(book_image, regions):
    """
    Save each detected region as a separate image file so you can see
    exactly what the algorithm detected (not read, just cropped).

    Args:
        book_image: The book cover image
        regions: List of detected regions
    """
    import os

    # Create output directory
    output_dir = "detected_regions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n[INFO] Saving cropped regions to '{output_dir}/' folder...")

    for idx, region in enumerate(regions, 1):
        x, y, w, h = region['bbox']
        classification = region['classification']

        # Crop the region
        cropped = book_image[y:y+h, x:x+w]

        # Determine if text or image
        region_type = "text" if ("Text" in classification or "Line" in classification) else "image"

        # Save with descriptive filename
        filename = f"{output_dir}/region_{idx:02d}_{region_type}_{w}x{h}.jpg"
        cv2.imwrite(filename, cropped)

    print(f"[SUCCESS] Saved {len(regions)} region images to '{output_dir}/'")


def print_detection_summary(regions, book_image):
    """
    Print a clear summary of what was detected, grouped by type.
    Also saves cropped images of each region.

    IMPORTANT: This does NOT read text (no OCR). It only detects regions
    and classifies them by shape/size. We cannot tell you WHAT the text says.

    Args:
        regions: List of detected regions
        book_image: The book cover image for cropping regions
    """
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print("\n⚠️  IMPORTANT: This is NOT OCR (Optical Character Recognition)")
    print("   We detect REGIONS where text/images exist, but we DON'T read")
    print("   what the text says or identify what images show.")
    print("\n💡 TIP: Check the 'detected_regions/' folder to see what each")
    print("   region actually contains (cropped image files).")
    print("="*60)

    # Save cropped regions first
    save_detected_regions(book_image, regions)

    # Group regions by type
    text_regions = []
    image_regions = []

    for idx, region in enumerate(regions, 1):
        classification = region['classification']

        # Classify as text or image based on classification
        if "Text" in classification or "Line" in classification:
            text_regions.append((idx, region))
        else:
            image_regions.append((idx, region))

    # Print text regions in table format with hierarchy and color info
    print(f"\n📝 TEXT REGIONS DETECTED: {len(text_regions)}")
    print("="*95)
    print(f"{'#':<4} {'Type':<28} {'Size':<12} {'Holes':<8} {'Mono':<6} {'Score':<7} {'File':<28}")
    print("-" * 95)
    if text_regions:
        for idx, region in text_regions:
            x, y, w, h = region['bbox']
            classification = region['classification']
            has_holes = region.get('has_holes', False)
            hole_count = region.get('hole_count', 0)
            is_monochrome = region.get('is_monochrome', False)
            text_score = region.get('text_score', 0)
            region_type = "text"
            filename = f"region_{idx:02d}_{region_type}_{w}x{h}.jpg"

            holes_str = f"✓({hole_count})" if has_holes else "✗"
            mono_str = "✓" if is_monochrome else "✗"
            print(f"{idx:<4} {classification:<28} {w}×{h:<10} {holes_str:<8} {mono_str:<6} {text_score:<7} {filename:<28}")
    else:
        print("  No text regions detected.")

    # Print image regions in table format
    print(f"\n🖼️  IMAGE/ICON REGIONS DETECTED: {len(image_regions)}")
    print("="*80)
    print(f"{'#':<4} {'Type':<28} {'Size':<12} {'Holes':<8} {'File':<28}")
    print("-" * 80)
    if image_regions:
        for idx, region in image_regions:
            x, y, w, h = region['bbox']
            classification = region['classification']
            has_holes = region.get('has_holes', False)
            hole_count = region.get('hole_count', 0)
            region_type = "image"
            filename = f"region_{idx:02d}_{region_type}_{w}x{h}.jpg"

            holes_str = f"✓ ({hole_count})" if has_holes else "✗"
            print(f"{idx:<4} {classification:<28} {w}×{h:<10} {holes_str:<8} {filename:<28}")
    else:
        print("  No image regions detected.")

    print("\n" + "="*60)
    print(f"TOTAL ELEMENTS DETECTED: {len(regions)}")
    print(f"  - Text regions: {len(text_regions)}")
    print(f"  - Image regions: {len(image_regions)}")
    print(f"\n💾 All regions saved to: detected_regions/")
    print("   Open those files to see what was detected!")
    print("="*60)


def draw_results(book_image, regions):
    """
    Draw bounding boxes and labels on the book image.

    Args:
        book_image: Book cover image
        regions: List of detected regions

    Returns:
        Annotated image
    """
    output = book_image.copy()
    height, width = book_image.shape[:2]

    # Draw each region
    for idx, region in enumerate(regions, 1):
        x, y, w, h = region['bbox']
        color = region['color']
        classification = region['classification']

        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Draw region number
        cv2.circle(output, (x, y), 10, color, -1)
        cv2.putText(output, str(idx), (x - 7, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Add statistics
    stats_y = 30
    cv2.putText(output, f"Elements Detected: {len(regions)}",
                (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add simplified legend
    legend_y = height - 180
    cv2.putText(output, "Classifications:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output, "Red = Large Text/Image", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, "Orange = Text Line", (10, legend_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
    cv2.putText(output, "Green = Text Block", (10, legend_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(output, "Cyan = Image/Icon", (10, legend_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output, "Magenta = Small Text", (10, legend_y + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(output, "Blue = Small Element", (10, legend_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return output


def main():
    """
    Main function - Production quality book cover analyzer.
    """
    ap = argparse.ArgumentParser(description="Production Book Cover Analyzer")
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the book cover image")
    args = vars(ap.parse_args())

    print("="*60)
    print("PRODUCTION BOOK COVER ANALYZER")
    print("Two-Stage Detection for Real-World Scenarios")
    print("="*60)

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
        print("\n[ERROR] Could not find book cover in image!")
        print("Tips:")
        print("  - Ensure book is clearly visible")
        print("  - Use good lighting")
        print("  - Place book on contrasting background")
        cv2.destroyAllWindows()
        return

    # Extract book region
    book_image = extract_book_region(image, book_contour)

    print(f"[SUCCESS] Book extracted: {book_image.shape[1]}x{book_image.shape[0]} pixels")

    # STAGE 2: Analyze elements on the book
    regions = analyze_book_elements(book_image)

    # Print clear summary with cropped images
    print_detection_summary(regions, book_image)

    # Draw results
    output = draw_results(book_image, regions)

    cv2.imshow("FINAL: Book Cover Analysis", output)
    cv2.waitKey(0)

    # Save results
    cv2.imwrite("book_cover_production_result.jpg", output)
    cv2.imwrite("book_cover_extracted.jpg", book_image)

    print("\n[SUCCESS] Analysis complete!")
    print(f"  - Annotated result: book_cover_production_result.jpg")
    print(f"  - Extracted book: book_cover_extracted.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

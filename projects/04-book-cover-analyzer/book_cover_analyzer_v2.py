"""
Book Cover Analyzer V2 - Improved with Adaptive Thresholding
=============================================================
This improved version uses adaptive thresholding to better detect
text and images ON the book cover, not just the book outline.

NEW IMPROVEMENTS:
- Adaptive thresholding (better for text detection)
- Inverted threshold (finds dark elements on light backgrounds)
- Better contour filtering

TUTORIAL CONCEPTS USED:
1. Loading and displaying images (cv2.imread, cv2.imshow)
2. Converting to grayscale (cv2.cvtColor)
3. Blurring for noise reduction (cv2.GaussianBlur)
4. Thresholding - UPGRADED to adaptive (cv2.adaptiveThreshold)
5. Finding contours (cv2.findContours)
6. Drawing shapes (cv2.rectangle, cv2.circle)
7. Drawing text (cv2.putText)
8. Morphological operations (cv2.dilate, cv2.erode)
"""

import cv2
import imutils
import argparse
import numpy as np


def classify_region_by_size(area, total_image_area):
    """
    Classify a region based on its size relative to the total image.

    Args:
        area: The area of the contour/region
        total_image_area: Total area of the image (width * height)

    Returns:
        tuple: (classification_name, color_for_box)
    """
    percentage = (area / total_image_area) * 100

    if percentage > 20:
        return ("Large Region (Main Image/Large Text)", (0, 0, 255))  # Red
    elif percentage > 2:
        return ("Medium Region (Text Block/Image)", (0, 255, 0))  # Green
    else:
        return ("Small Region (Small Text/Icon)", (255, 0, 0))  # Blue


def analyze_book_cover(image_path):
    """
    Main function to analyze a book cover image using adaptive thresholding.

    Args:
        image_path: Path to the book cover image file
    """

    # ========================================
    # STEP 1: LOAD THE IMAGE
    # ========================================
    print("[INFO] Loading image...")
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not load image from {image_path}")
        print("Please make sure the file exists and the path is correct.")
        return

    # Get image dimensions
    height, width = image.shape[:2]
    total_area = height * width
    print(f"[INFO] Image size: {width}x{height} pixels ({total_area} total pixels)")

    # Display original image
    cv2.imshow("1. Original Book Cover", image)
    cv2.waitKey(0)

    # ========================================
    # STEP 2: CONVERT TO GRAYSCALE
    # ========================================
    print("[INFO] Converting to grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("2. Grayscale", gray)
    cv2.waitKey(0)

    # ========================================
    # STEP 3: BLUR THE IMAGE
    # ========================================
    # Reduce noise before thresholding
    print("[INFO] Applying Gaussian blur to reduce noise...")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("3. Blurred", blurred)
    cv2.waitKey(0)

    # ========================================
    # STEP 4: ADAPTIVE THRESHOLDING (NEW!)
    # ========================================
    # This is the KEY improvement!
    # Instead of one threshold for whole image, it calculates threshold
    # for small neighborhoods. Perfect for text on white backgrounds.

    print("[INFO] Applying ADAPTIVE thresholding...")
    print("  - This calculates different thresholds for different regions")
    print("  - Much better for detecting text and images on white paper")

    # cv2.adaptiveThreshold parameters:
    # - src: input grayscale image
    # - maxValue: value to assign to pixels that pass threshold (255 = white)
    # - adaptiveMethod: ADAPTIVE_THRESH_GAUSSIAN_C uses weighted sum of neighborhood
    # - thresholdType: THRESH_BINARY_INV means dark objects become white (inverted)
    # - blockSize: size of neighborhood (must be odd number) - 11 works well for text
    # - C: constant subtracted from weighted mean (fine-tuning parameter)

    thresh = cv2.adaptiveThreshold(
        blurred,                           # Input image
        255,                               # Max value (white)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Use gaussian weighted neighborhood
        cv2.THRESH_BINARY_INV,            # Invert: dark text becomes white
        11,                                # Block size (neighborhood size)
        2                                  # Constant C (subtract from mean)
    )

    cv2.imshow("4. Adaptive Threshold (Dark elements now white)", thresh)
    cv2.waitKey(0)

    # ========================================
    # STEP 5: MORPHOLOGICAL OPERATIONS (NEW!)
    # ========================================
    # Clean up the threshold image using morphology
    # This helps connect nearby text and remove small noise

    print("[INFO] Applying morphological operations...")
    print("  - Dilation: connects nearby text/elements")
    print("  - Erosion: removes small noise")

    # Create a rectangular kernel for morphological operations
    # Horizontal kernel helps connect letters in words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Dilate to connect nearby elements (like letters in a word)
    morph = cv2.dilate(thresh, kernel, iterations=2)

    # Optional: erode slightly to separate distinct regions
    morph = cv2.erode(morph, kernel, iterations=1)

    cv2.imshow("5. Morphological Cleanup", morph)
    cv2.waitKey(0)

    # ========================================
    # STEP 6: FIND CONTOURS
    # ========================================
    print("[INFO] Finding contours in thresholded image...")
    contours = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    print(f"[INFO] Found {len(contours)} total contours")

    # ========================================
    # STEP 7: FILTER AND CLASSIFY CONTOURS
    # ========================================
    # Filter out very tiny contours (noise) and very large ones (background)

    min_area = 100      # Minimum area (smaller than before to catch text)
    max_area = total_area * 0.8  # Maximum area (ignore if too big - likely background)

    significant_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore very small contours (noise) and very large ones (background)
        if area < min_area or area > max_area:
            continue

        # Get bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Classify the region
        classification, color = classify_region_by_size(area, total_area)

        # Store information about this region
        significant_regions.append({
            'area': area,
            'bbox': (x, y, w, h),
            'classification': classification,
            'color': color
        })

    print(f"[INFO] Found {len(significant_regions)} significant regions after filtering")

    # ========================================
    # STEP 8: DRAW RESULTS
    # ========================================
    # Create a copy of original image to draw on
    output = image.copy()

    # Sort regions by area (largest first)
    significant_regions.sort(key=lambda x: x['area'], reverse=True)

    # Draw bounding boxes and labels for each region
    for idx, region in enumerate(significant_regions, 1):
        x, y, w, h = region['bbox']
        area = region['area']
        classification = region['classification']
        color = region['color']

        # Draw rectangle around the region
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Draw a small circle at the top-left corner with the region number
        cv2.circle(output, (x, y), 8, color, -1)
        cv2.putText(output, str(idx), (x - 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Print region info to console
        percentage = (area / total_area) * 100
        print(f"  Region #{idx}: {classification}")
        print(f"    - Area: {area} pixels ({percentage:.2f}% of image)")
        print(f"    - Position: ({x}, {y}), Size: {w}x{h}")

    # ========================================
    # STEP 9: ADD STATISTICS OVERLAY
    # ========================================
    stats_y = 30
    cv2.putText(output, f"Regions Detected: {len(significant_regions)}",
                (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if significant_regions:
        largest_area = significant_regions[0]['area']
        largest_percentage = (largest_area / total_area) * 100
        cv2.putText(output, f"Largest: {largest_percentage:.1f}%",
                    (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add legend
    legend_y = height - 100
    cv2.putText(output, "Legend:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output, "Red = Large", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, "Green = Medium", (10, legend_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(output, "Blue = Small", (10, legend_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ========================================
    # STEP 10: DISPLAY FINAL RESULT
    # ========================================
    cv2.imshow("6. FINAL RESULT - Detected Regions", output)
    cv2.waitKey(0)

    # Save the result
    output_path = "book_cover_analysis_v2_result.jpg"
    cv2.imwrite(output_path, output)
    print(f"\n[SUCCESS] Analysis complete! Result saved to: {output_path}")

    # Also save the threshold image for comparison
    cv2.imwrite("adaptive_threshold_debug.jpg", thresh)
    print(f"[DEBUG] Threshold image saved to: adaptive_threshold_debug.jpg")

    # Clean up windows
    cv2.destroyAllWindows()


def main():
    """
    Entry point for the script.
    """
    ap = argparse.ArgumentParser(description="Analyze book cover - V2 with adaptive thresholding")
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the book cover image file")
    args = vars(ap.parse_args())

    print("="*60)
    print("BOOK COVER ANALYZER V2 - WITH ADAPTIVE THRESHOLDING")
    print("="*60)
    print("\nIMPROVEMENTS:")
    print("✓ Adaptive thresholding (better text detection)")
    print("✓ Morphological operations (cleaner regions)")
    print("✓ Better filtering (detects smaller elements)")
    print("="*60 + "\n")

    analyze_book_cover(args["image"])


if __name__ == "__main__":
    main()

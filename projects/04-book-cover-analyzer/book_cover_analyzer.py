"""
Book Cover Analyzer - Beginner OpenCV Project
==============================================
This script analyzes a book cover image and detects different regions/elements.

TUTORIAL CONCEPTS USED:
1. Loading and displaying images (cv2.imread, cv2.imshow)
2. Converting to grayscale (cv2.cvtColor)
3. Blurring for noise reduction (cv2.GaussianBlur)
4. Edge detection (cv2.Canny)
5. Thresholding (cv2.threshold)
6. Finding contours (cv2.findContours)
7. Drawing shapes (cv2.rectangle, cv2.circle)
8. Drawing text (cv2.putText)

WHAT IT DOES:
- Detects rectangular regions on book cover
- Classifies regions by size (large/medium/small)
- Draws color-coded bounding boxes
- Shows statistics about detected regions
"""

import cv2
import imutils
import argparse

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

    if percentage > 30:
        return ("Large Region (Likely Main Image)", (0, 0, 255))  # Red
    elif percentage > 5:
        return ("Medium Region (Text/Secondary Image)", (0, 255, 0))  # Green
    else:
        return ("Small Region (Decorative Element)", (255, 0, 0))  # Blue


def analyze_book_cover(image_path):
    """
    Main function to analyze a book cover image.

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
    cv2.imshow("Original Book Cover", image)
    cv2.waitKey(0)

    # ========================================
    # STEP 2: CONVERT TO GRAYSCALE
    # ========================================
    # Why? Most image processing operations work better on grayscale images
    # Grayscale has 1 channel vs 3 (BGR), making it simpler to process
    print("[INFO] Converting to grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(0)

    # ========================================
    # STEP 3: BLUR THE IMAGE
    # ========================================
    # Why? Reduces high-frequency noise and helps edge detection work better
    # GaussianBlur smooths the image using a Gaussian kernel
    print("[INFO] Applying Gaussian blur to reduce noise...")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(0)

    # ========================================
    # STEP 4: EDGE DETECTION
    # ========================================
    # Canny edge detection finds the boundaries/outlines of objects
    # Parameters: (image, min_threshold, max_threshold)
    print("[INFO] Detecting edges using Canny algorithm...")
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Edges Detected", edged)
    cv2.waitKey(0)

    # ========================================
    # STEP 5: FIND CONTOURS
    # ========================================
    # Contours are continuous curves that follow the boundaries of objects
    # RETR_EXTERNAL: only retrieves external/outer contours
    # CHAIN_APPROX_SIMPLE: compresses contours to save memory
    print("[INFO] Finding contours...")
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    print(f"[INFO] Found {len(contours)} total contours")

    # ========================================
    # STEP 6: FILTER AND CLASSIFY CONTOURS
    # ========================================
    # We'll filter out very tiny contours (likely noise)
    # and classify the remaining ones by size

    min_area = 500  # Minimum area to consider (ignore tiny noise)
    significant_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore very small contours (noise)
        if area < min_area:
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
    # STEP 7: DRAW RESULTS
    # ========================================
    # Create a copy of original image to draw on
    output = image.copy()

    # Sort regions by area (largest first) for better visualization
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
    # STEP 8: ADD STATISTICS OVERLAY
    # ========================================
    # Add text overlay with summary statistics
    stats_y = 30
    cv2.putText(output, f"Total Regions Found: {len(significant_regions)}",
                (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if significant_regions:
        largest_area = significant_regions[0]['area']
        largest_percentage = (largest_area / total_area) * 100
        cv2.putText(output, f"Largest Region: {largest_percentage:.1f}% of image",
                    (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add legend
    legend_y = height - 100
    cv2.putText(output, "Legend:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output, "Red = Large Region", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, "Green = Medium Region", (10, legend_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(output, "Blue = Small Region", (10, legend_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ========================================
    # STEP 9: DISPLAY FINAL RESULT
    # ========================================
    cv2.imshow("Book Cover Analysis - Final Result", output)
    cv2.waitKey(0)

    # Save the result
    output_path = "book_cover_analysis_result.jpg"
    cv2.imwrite(output_path, output)
    print(f"\n[SUCCESS] Analysis complete! Result saved to: {output_path}")

    # Clean up windows
    cv2.destroyAllWindows()


def main():
    """
    Entry point for the script.
    Uses command line arguments to get the image path.
    """
    # Set up command line argument parser
    ap = argparse.ArgumentParser(description="Analyze a book cover image")
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the book cover image file")
    args = vars(ap.parse_args())

    # Run the analysis
    analyze_book_cover(args["image"])


if __name__ == "__main__":
    main()

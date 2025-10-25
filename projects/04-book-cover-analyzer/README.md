# Book Cover Analyzer

A beginner-friendly OpenCV project that analyzes book cover images and detects different regions/elements.

## What This Project Does

1. Loads a book cover image
2. Detects rectangular regions (potential text boxes, images, design elements)
3. Classifies regions by size:
   - **Large regions** (>30% of image) - Likely main images
   - **Medium regions** (5-30% of image) - Text blocks or secondary images
   - **Small regions** (<5% of image) - Decorative elements
4. Draws color-coded bounding boxes around detected regions
5. Displays statistics about the book cover

## OpenCV Concepts Used

This project uses concepts from the beginner tutorial:
- Loading and displaying images (`cv2.imread`, `cv2.imshow`)
- Converting to grayscale (`cv2.cvtColor`)
- Blurring for noise reduction (`cv2.GaussianBlur`)
- Edge detection (`cv2.Canny`)
- Finding contours (`cv2.findContours`)
- Drawing shapes (`cv2.rectangle`, `cv2.circle`)
- Drawing text (`cv2.putText`)

## How to Use

### 1. Prepare Your Image
Place your book cover image in the `data/images/` folder. For example:
```
data/images/my_book_cover.jpg
```

### 2. Run the Script
Activate your virtual environment and run:
```bash
source venv/bin/activate
cd projects/04-book-cover-analyzer
python book_cover_analyzer.py --image ../../data/images/my_book_cover.jpg
```

### 3. View Results
- Press any key to cycle through the processing steps
- Each step shows you what's happening:
  1. Original image
  2. Grayscale conversion
  3. Blurred image
  4. Edge detection
  5. Final result with detected regions

### 4. Check Output
The final analyzed image is saved as `book_cover_analysis_result.jpg` in the current directory.

## Understanding the Output

### Color-Coded Boxes
- **Red boxes** = Large regions (likely the main cover image)
- **Green boxes** = Medium regions (text blocks or secondary images)
- **Blue boxes** = Small regions (decorative elements)

### Console Output
The script prints detailed information about each detected region:
- Region number
- Classification
- Area in pixels and percentage of total image
- Position and size

## Example Output
```
[INFO] Loading image...
[INFO] Image size: 800x1200 pixels (960000 total pixels)
[INFO] Converting to grayscale...
[INFO] Applying Gaussian blur to reduce noise...
[INFO] Detecting edges using Canny algorithm...
[INFO] Finding contours...
[INFO] Found 45 total contours
[INFO] Found 8 significant regions after filtering
  Region #1: Large Region (Likely Main Image)
    - Area: 384000 pixels (40.00% of image)
    - Position: (50, 100), Size: 700x800
  Region #2: Medium Region (Text/Secondary Image)
    - Area: 96000 pixels (10.00% of image)
    - Position: (100, 50), Size: 600x160
  ...
```

## Tips for Best Results

1. **Use clear images** - High-resolution book cover scans work best
2. **Good lighting** - Avoid shadows or glare on the book cover
3. **Straight angle** - Take photo from directly above (not at an angle)
4. **Clean background** - Place book on a plain, contrasting background

## Adjusting Detection Sensitivity

You can modify these values in the script for better results:

```python
# Line 97: Canny edge detection thresholds
edged = cv2.Canny(blurred, 30, 150)  # Try (50, 150) or (20, 100)

# Line 115: Minimum area to filter noise
min_area = 500  # Increase to ignore more small details
```

## Next Steps

After mastering this project, try:
1. Adding perspective correction for angled photos
2. Implementing color analysis of detected regions
3. Counting specific shapes (circles, triangles, etc.)
4. Creating a batch processor for multiple book covers

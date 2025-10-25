"""
Generate HTML Table with OCR Results
=====================================
This script creates an interactive HTML table showing:
1. Region images (cropped from the book cover)
2. OCR text that was read from each region
3. Classification and confidence scores

Run this after running book_cover_analyzer_with_ocr.py to visualize results.
"""

import os
import json
import base64
import cv2
from pathlib import Path


def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_table(regions_dir, output_html="ocr_results_table.html"):
    """
    Generate an HTML table with images and OCR results.

    Since we don't have the OCR results stored, we'll run OCR again
    on the saved region images.
    """
    import easyocr
    import warnings
    warnings.filterwarnings('ignore')

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR ready!")

    # Get all region files
    region_files = sorted([f for f in os.listdir(regions_dir) if f.endswith('.jpg')])

    print(f"\nFound {len(region_files)} region images")

    # Separate text and image regions
    text_regions = []
    image_regions = []

    for filename in region_files:
        filepath = os.path.join(regions_dir, filename)

        # Parse filename: region_XX_type_WxH.jpg
        parts = filename.replace('.jpg', '').split('_')
        region_num = int(parts[1])
        region_type = parts[2]  # 'text' or 'image'
        size = parts[3]  # 'WxH'

        # Read the image
        img = cv2.imread(filepath)
        if img is None:
            continue

        # Convert to base64 for HTML embedding
        img_base64 = image_to_base64(filepath)

        region_data = {
            'number': region_num,
            'filename': filename,
            'type': region_type,
            'size': size,
            'image_base64': img_base64,
            'ocr_text': None,
            'confidence': 0.0
        }

        # Run OCR only on text regions
        if region_type == 'text':
            print(f"  Running OCR on region {region_num}...")
            try:
                results = reader.readtext(img, detail=1)
                if results:
                    texts = [r[1] for r in results]
                    confidences = [r[2] for r in results]
                    region_data['ocr_text'] = ' '.join(texts)
                    region_data['confidence'] = sum(confidences) / len(confidences)
                else:
                    region_data['ocr_text'] = '[empty]'
            except Exception as e:
                region_data['ocr_text'] = f'[error: {str(e)[:30]}]'

        if region_type == 'text':
            text_regions.append(region_data)
        else:
            image_regions.append(region_data)

    # Sort by region number
    text_regions.sort(key=lambda x: x['number'])
    image_regions.sort(key=lambda x: x['number'])

    print(f"\nProcessed {len(text_regions)} text regions and {len(image_regions)} image regions")

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Book Cover OCR Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            background-color: #e0e0e0;
            padding: 10px;
            border-radius: 5px;
        }}
        .summary {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .summary-number {{
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .summary-label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .region-img {{
            max-width: 200px;
            max-height: 150px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .region-img:hover {{
            transform: scale(1.5);
            z-index: 1000;
            position: relative;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        .ocr-text {{
            font-family: monospace;
            background-color: #f8f8f8;
            padding: 8px;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
        }}
        .confidence {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .confidence-high {{
            background-color: #4CAF50;
            color: white;
        }}
        .confidence-medium {{
            background-color: #FF9800;
            color: white;
        }}
        .confidence-low {{
            background-color: #f44336;
            color: white;
        }}
        .region-number {{
            font-weight: bold;
            color: #4CAF50;
            font-size: 16px;
        }}
        .type-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .type-text {{
            background-color: #2196F3;
            color: white;
        }}
        .type-image {{
            background-color: #9C27B0;
            color: white;
        }}
    </style>
</head>
<body>
    <h1>📖 Book Cover OCR Analysis Results</h1>

    <div class="summary">
        <h3>Summary</h3>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-number">{len(text_regions) + len(image_regions)}</div>
                <div class="summary-label">Total Regions</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len(text_regions)}</div>
                <div class="summary-label">Text Regions</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len(image_regions)}</div>
                <div class="summary-label">Image Regions</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len([r for r in text_regions if r['ocr_text'] and r['ocr_text'] not in ['[empty]', '[error]']])}</div>
                <div class="summary-label">Readable Text</div>
            </div>
        </div>
    </div>
"""

    # Text regions table
    html += """
    <h2>📝 Text Regions (with OCR Results)</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Preview</th>
                <th>OCR Text</th>
                <th>Confidence</th>
                <th>Size</th>
            </tr>
        </thead>
        <tbody>
"""

    for region in text_regions:
        conf = region['confidence']
        if conf > 0.7:
            conf_class = 'confidence-high'
        elif conf > 0.4:
            conf_class = 'confidence-medium'
        else:
            conf_class = 'confidence-low'

        conf_display = f"{conf*100:.0f}%" if conf > 0 else "N/A"
        ocr_text = region['ocr_text'] if region['ocr_text'] else '[empty]'

        html += f"""
            <tr>
                <td><span class="region-number">#{region['number']}</span></td>
                <td><img src="data:image/jpeg;base64,{region['image_base64']}" class="region-img" alt="Region {region['number']}"></td>
                <td><div class="ocr-text">{ocr_text}</div></td>
                <td><span class="confidence {conf_class}">{conf_display}</span></td>
                <td>{region['size']}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
"""

    # Image regions table
    html += """
    <h2>🖼️ Image/Icon Regions</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Preview</th>
                <th>Type</th>
                <th>Size</th>
                <th>Filename</th>
            </tr>
        </thead>
        <tbody>
"""

    for region in image_regions:
        html += f"""
            <tr>
                <td><span class="region-number">#{region['number']}</span></td>
                <td><img src="data:image/jpeg;base64,{region['image_base64']}" class="region-img" alt="Region {region['number']}"></td>
                <td><span class="type-badge type-image">Image</span></td>
                <td>{region['size']}</td>
                <td style="font-family: monospace; font-size: 11px;">{region['filename']}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>

    <div style="margin-top: 40px; padding: 20px; background-color: #fff; border-radius: 8px;">
        <h3>💡 How to Use This Table</h3>
        <ul>
            <li><strong>Hover over images</strong> to enlarge them for better viewing</li>
            <li><strong>Text regions</strong> show the OCR-detected text with confidence scores</li>
            <li><strong>Confidence scores:</strong>
                <span class="confidence confidence-high">70%+</span> = High accuracy,
                <span class="confidence confidence-medium">40-70%</span> = Medium accuracy,
                <span class="confidence confidence-low">&lt;40%</span> = Low accuracy
            </li>
            <li><strong>Image regions</strong> are graphical elements detected on the book cover</li>
        </ul>
    </div>

</body>
</html>
"""

    # Write to file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HTML table generated: {output_html}")
    print(f"   Open this file in your browser to view the results!")

    return output_html


if __name__ == "__main__":
    regions_dir = "detected_regions"

    if not os.path.exists(regions_dir):
        print(f"Error: '{regions_dir}' directory not found!")
        print("Run book_cover_analyzer_with_ocr.py first to generate regions.")
        exit(1)

    print("="*60)
    print("GENERATING OCR RESULTS TABLE")
    print("="*60)

    output_file = generate_html_table(regions_dir)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nTo view results, open: {output_file}")
    print("\nThe table shows:")
    print("  ✓ All detected text with OCR readings")
    print("  ✓ All detected images/icons")
    print("  ✓ Confidence scores for text detection")
    print("  ✓ Visual previews (hover to zoom)")

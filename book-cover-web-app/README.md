# Book Cover Analyzer - ML Web Application

A production-ready web application that analyzes book covers using a three-stage ML pipeline combining classical computer vision with modern deep learning.

## Features

- **Book Detection**: Automatically finds and extracts book covers from images using OpenCV edge detection
- **Text Recognition**: Detects and reads all text on book covers using EasyOCR neural networks
- **Image Classification**: Classifies non-text visual elements using PyTorch ResNet18
- **Interactive Web UI**: Drag-and-drop interface with real-time results display
- **Educational**: Includes detailed methodology page explaining how each ML model works

## ML Pipeline Architecture

```
Stage 1: OpenCV (Classical CV)
  └─> Find book boundary using edge detection and contours
      └─> Apply perspective transform

Stage 2: EasyOCR (Deep Learning)
  └─> CRAFT network: Detect WHERE text is located
      └─> Recognition network: Read WHAT the text says

Stage 3: PyTorch ResNet18 (Deep Learning)
  └─> Sample non-text regions
      └─> Classify image types using transfer learning
```

## Technology Stack

**Backend:**
- Python 3.x
- Flask 3.1.2
- OpenCV 4.x (classical computer vision)
- EasyOCR (text detection and recognition)
- PyTorch 2.x (image classification)
- imutils (image processing utilities)

**Frontend:**
- HTML5, CSS3, JavaScript (vanilla)
- Responsive design
- Drag-and-drop file upload

**ML Models:**
- **CRAFT** - Character Region Awareness for Text detection
- **ResNet18** - Pre-trained on ImageNet for image classification

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
```bash
cd opencv-learning/book-cover-web-app
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install flask opencv-python-headless easyocr torch torchvision imutils
```

4. **Run the application:**
```bash
python3 app.py
```

5. **Open in browser:**
```
http://127.0.0.1:5001
```

## Usage

### Web Interface

1. **Upload Image**: Drag and drop a book cover image or click to browse
2. **Analyze**: Click "Analyze Book Cover" to run the ML pipeline
3. **View Results**: See detected text and image regions in interactive tables
4. **Learn**: Click "How does this work?" to understand the ML methodology

### API Endpoints

**POST /upload**
- Upload and analyze book cover image
- Content-Type: multipart/form-data
- Accepts: PNG, JPG, JPEG (max 16MB)
- Returns: JSON with analysis results

```bash
curl -X POST -F "file=@book_cover.jpg" http://127.0.0.1:5001/upload
```

**GET /methodology**
- Educational page explaining the ML pipeline

**GET /health**
- Health check endpoint for monitoring

## Project Structure

```
book-cover-web-app/
├── app.py                    # Flask application (routes, config)
├── analyzer/
│   ├── __init__.py          # Module exports
│   └── book_analyzer.py     # ML pipeline (OpenCV + EasyOCR + PyTorch)
├── templates/
│   ├── index.html           # Upload page with drag-drop UI
│   └── methodology.html     # Educational explanation page
├── static/
│   ├── css/                 # Stylesheets (inline for now)
│   ├── js/                  # JavaScript (inline for now)
│   └── images/              # Static assets
├── uploads/                 # Temporary uploaded files
└── README.md               # This file
```

## ML Concepts Demonstrated

### 1. Transfer Learning
- Uses ResNet18 pre-trained on ImageNet (1.2M images)
- Zero-shot inference - no fine-tuning on book covers
- Leverages learned features (edges, textures, shapes)

### 2. Multi-Model Pipeline
- Combines classical CV (OpenCV) with deep learning (PyTorch)
- Each stage specialized for specific task
- Models work together in sequence

### 3. Stateful Service Pattern
- Models loaded once at startup (not per request)
- Fast inference (milliseconds)
- Memory-efficient model reuse

### 4. Inference vs Training
- Application does INFERENCE only
- Uses pre-trained models
- No GPU required (CPU inference)

## Sample Output

**Analysis Results:**
```json
{
  "success": true,
  "book_extracted": true,
  "text_regions": [
    {
      "id": 1,
      "text": "SCALING",
      "confidence": 1.0,
      "bbox": {"x": 120, "y": 45, "width": 510, "height": 85}
    },
    {
      "id": 2,
      "text": "How Small Teams",
      "confidence": 1.0,
      "bbox": {"x": 85, "y": 185, "width": 580, "height": 70}
    }
  ],
  "image_regions": [
    {
      "id": 8,
      "pytorch_class": "element_78",
      "confidence": 0.547,
      "bbox": {"x": 275, "y": 407, "width": 200, "height": 200}
    }
  ],
  "summary": {
    "total_text_regions": 7,
    "total_image_regions": 1,
    "book_dimensions": "750x1014"
  }
}
```

## Performance

**Model Loading (One-time):**
- EasyOCR: ~3-4 seconds
- PyTorch ResNet18: ~1-2 seconds
- Total startup: ~5-8 seconds

**Inference (Per Request):**
- Book detection (OpenCV): ~100-200ms
- Text recognition (EasyOCR): ~2-3 seconds
- Image classification (PyTorch): ~50-100ms
- Total per image: ~3-5 seconds

## Current Limitations

1. **Zero-Shot Classification**: ResNet18 uses generic ImageNet classes, not book-specific categories
2. **Simple Region Sampling**: Only analyzes center region for images
3. **No Fine-Tuning**: Model not trained on book cover dataset

## Future Enhancements

### Short-term
- [ ] Add more image region sampling (corners, edges)
- [ ] Improve text grouping (combine title words)
- [ ] Add confidence threshold filtering

### Medium-term
- [ ] Fine-tune ResNet18 on book cover dataset
- [ ] Add semantic labeling (title, author, publisher)
- [ ] Implement layout analysis

### Long-term
- [ ] Replace with end-to-end object detection (YOLO)
- [ ] Add book metadata extraction (ISBN, price)
- [ ] Build recommendation system using visual features

## Development

**Run in debug mode:**
```bash
python3 app.py  # Debug mode enabled by default
```

**Production deployment:**
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Testing

**Test the API:**
```bash
# Upload image
curl -X POST -F "file=@test.jpg" http://127.0.0.1:5001/upload

# Health check
curl http://127.0.0.1:5001/health
```

## Troubleshooting

**Port 5000 already in use:**
- macOS AirPlay uses port 5000
- App configured to use port 5001 instead
- Or disable AirPlay in System Preferences

**Models downloading:**
- First run downloads EasyOCR models (~100MB)
- PyTorch downloads ResNet18 weights (~45MB)
- Stored in ~/.EasyOCR and ~/.cache/torch

**Memory usage:**
- EasyOCR: ~500MB RAM
- PyTorch ResNet18: ~200MB RAM
- Total: ~1GB RAM minimum recommended

## Learning Resources

- **Classical CV**: OpenCV edge detection, contours, perspective transforms
- **OCR**: EasyOCR CRAFT architecture, text recognition networks
- **Deep Learning**: PyTorch tensors, ResNet architecture, transfer learning
- **Web**: Flask routing, file uploads, RESTful APIs

## License

Educational project - part of opencv-learning repository.

## Credits

**ML Models:**
- EasyOCR by JaidedAI
- ResNet18 by Microsoft Research (ImageNet pre-trained weights)
- OpenCV by OpenCV Foundation

**Built as part of:** opencv-learning journey - exploring computer vision and machine learning through practical projects.

---

**Repository:** anshuman-dev/opencv-learning
**Project:** Book Cover Analyzer Web Application
**Version:** 1.0 (Production-ready)

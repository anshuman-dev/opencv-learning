"""
Flask Web Application for Book Cover Analyzer
Integrates ML backend with web interface
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from analyzer import BookCoverAnalyzer

# ML CONCEPT: Web Service Architecture
# =====================================
# Pattern: Initialize expensive resources (ML models) once at startup
# Benefit: Fast response times - models loaded in memory, not per request
# Trade-off: Higher memory usage, but much faster inference

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize analyzer ONCE at startup
# ML CONCEPT: Stateful Service
# =============================
# This loads EasyOCR + PyTorch ResNet18 into memory
# Takes 5-10 seconds at startup, but then ALL requests are fast
print("[INFO] Loading ML models...")
analyzer = BookCoverAnalyzer(verbose=True)
print("[SUCCESS] ML models ready!")


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """
    Main upload page.

    WEB CONCEPT: Single Page Application (SPA) Pattern
    ===================================================
    - User uploads image via drag-drop or file picker
    - JavaScript sends POST to /upload
    - Results displayed on same page (no reload)
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and trigger ML analysis.

    ML CONCEPT: Inference API Pattern
    ==================================
    1. Validate input (file type, size)
    2. Save file temporarily
    3. Run ML pipeline (OpenCV → EasyOCR → PyTorch)
    4. Return JSON results
    5. Frontend displays results

    This is the standard pattern for ML APIs:
    - RESTful endpoint
    - JSON in/out
    - Synchronous inference (< 5 seconds)
    """
    # Validation
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run ML analysis
        # ML CONCEPT: Three-Stage Pipeline
        # =================================
        # Stage 1: OpenCV (Classical CV) - Find book boundary
        # Stage 2: EasyOCR (Neural Net) - Detect & read text
        # Stage 3: PyTorch (Neural Net) - Classify images
        results = analyzer.analyze(filepath)

        # Add filename to results for frontend
        results['filename'] = filename
        results['image_url'] = f'/uploads/{filename}'

        return jsonify(results), 200

    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'success': False
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded images.

    WEB CONCEPT: Static File Serving
    =================================
    Allows frontend to display the uploaded image
    alongside analysis results.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/methodology')
def methodology():
    """
    Educational page explaining how the analyzer works.

    EDUCATIONAL CONCEPT: Transparency
    =================================
    Show users:
    - What ML models are used (EasyOCR, ResNet18)
    - How the pipeline works (3 stages)
    - What each model does
    - Limitations (zero-shot, not fine-tuned)
    """
    return render_template('methodology.html')


@app.route('/health')
def health():
    """
    Health check endpoint.

    WEB CONCEPT: Health Check
    =========================
    Used by:
    - Load balancers (is server alive?)
    - Monitoring systems (is service healthy?)
    - DevOps tools (deployment readiness)
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'analyzer_ready': analyzer is not None
    }), 200


if __name__ == '__main__':
    """
    Development server.

    IMPORTANT: For production, use gunicorn or uWSGI:
    $ gunicorn -w 4 -b 0.0.0.0:8000 app:app

    ML CONCEPT: Multi-Worker Deployment
    ===================================
    - Each worker loads models once
    - Workers share CPU/GPU resources
    - Handle concurrent requests
    - Graceful restarts without downtime
    """
    PORT = 5001
    print("\n" + "="*60)
    print("Book Cover Analyzer Web App")
    print("="*60)
    print("ML Pipeline:")
    print("  1. OpenCV - Find book boundary")
    print("  2. EasyOCR - Detect & read text")
    print("  3. PyTorch ResNet18 - Classify images")
    print("="*60)
    print(f"Open: http://127.0.0.1:{PORT}")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=PORT)

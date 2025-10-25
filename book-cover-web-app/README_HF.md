---
title: Book Cover Analyzer
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.2
app_file: app_gradio.py
pinned: false
license: mit
---

# 📖 Book Cover Analyzer

An AI-powered book cover analysis tool that automatically detects:

- **📚 Book Title** - Inferred from detected text
- **✍️ Author(s)** - Extracted using heuristics
- **🏢 Publisher** - Identified from text positioning
- **🖼️ Visual Elements** - Classified using ResNet18 (ImageNet)
- **📝 All Text** - Extracted using EasyOCR

## Technologies

- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch ResNet18
- **OCR**: EasyOCR
- **Interface**: Gradio

## How It Works

1. **Text Detection**: EasyOCR extracts all text regions
2. **Text Interpretation**: Heuristics infer title/author/publisher based on position and formatting
3. **Image Classification**: ResNet18 classifies visual elements (book jacket, person, etc.)
4. **Smart Analysis**: Combines OCR + computer vision for comprehensive results

## Deploy Your Own

This app can run on:
- **Hugging Face Spaces** (Free - 16GB RAM)
- Local machine with Python 3.9+

Built with ❤️ using OpenCV and PyTorch.

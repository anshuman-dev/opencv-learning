"""
Gradio interface for Book Cover Analyzer - optimized for Hugging Face Spaces
"""
import gradio as gr
from analyzer.book_analyzer import BookCoverAnalyzer
from PIL import Image
import json

# Initialize analyzer once (loads models)
print("Loading models... This may take 30-60 seconds on first run.")
analyzer = BookCoverAnalyzer(verbose=True)
print("Models loaded successfully!")

def analyze_book_cover(image):
    """Analyze uploaded book cover image"""
    if image is None:
        return "Please upload an image.", "", "", "", ""

    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Analyze
        result = analyzer.analyze(image)

        # Extract interpretation
        interp = result.get('interpretation', {})

        # Format outputs
        title = interp.get('inferred_title') or "Unable to determine"
        authors = ", ".join(interp.get('inferred_authors', [])) or "Unable to determine"
        publisher = interp.get('inferred_publisher') or "Unable to determine"

        # Format detected elements
        elements_text = f"**Detected {len(result['detected_elements'])} visual elements:**\n"
        for elem in result['detected_elements'][:5]:  # Top 5
            elements_text += f"- {elem['class_name']}: {elem['confidence']:.1%}\n"

        # Format detected text
        text_regions = result.get('text_regions', [])
        text_output = f"**Detected {len(text_regions)} text regions:**\n"
        for region in text_regions[:10]:  # Top 10
            text_output += f"- {region['text']}\n"

        # Format other text (reviews/quotes)
        other = interp.get('other_text', [])
        if other:
            text_output += f"\n**Reviews/Quotes:**\n"
            for text in other[:3]:
                text_output += f"- {text}\n"

        return title, authors, publisher, elements_text, text_output

    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        return error_msg, "", "", "", ""

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_book_cover,
    inputs=gr.Image(type="pil", label="Upload Book Cover"),
    outputs=[
        gr.Textbox(label="📚 Book Title", lines=2),
        gr.Textbox(label="✍️ Author(s)", lines=2),
        gr.Textbox(label="🏢 Publisher", lines=1),
        gr.Textbox(label="🖼️ Visual Elements (ImageNet Classification)", lines=8),
        gr.Textbox(label="📝 Detected Text (EasyOCR)", lines=15)
    ],
    title="📖 Book Cover Analyzer",
    description="""
    Upload a book cover image to automatically detect:
    - **Book title, author, publisher** (inferred from text)
    - **Visual elements** (classified using ResNet18)
    - **All text regions** (extracted using EasyOCR)

    Powered by OpenCV, PyTorch, and EasyOCR.
    """,
    examples=[
        # You can add example images here if you want
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

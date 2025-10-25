"""
Load and Display Image
Learn how to load, display, and save images using OpenCV.
"""

import cv2

def main():
    # Note: Replace 'sample.jpg' with the path to your image
    # You can add images to the data/images/ folder
    image_path = '../../data/images/sample.jpg'

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please add an image to data/images/ folder and update the path")
        return

    # Get image dimensions
    height, width, channels = img.shape
    print(f"Image dimensions: {width}x{height}")
    print(f"Number of channels: {channels}")

    # Display the image
    cv2.imshow('Original Image', img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)

    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the grayscale image
    cv2.imwrite('output_gray.png', gray)
    print("Grayscale image saved as 'output_gray.png'")

if __name__ == "__main__":
    main()

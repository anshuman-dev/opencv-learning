"""
Hello OpenCV - Test Installation
This script verifies that OpenCV is properly installed.
"""

import cv2
import numpy as np

def main():
    # Check OpenCV version
    print(f"OpenCV Version: {cv2.__version__}")

    # Create a simple image (black background)
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add text
    text = "Hello OpenCV!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50, 200), font, 2, (255, 255, 255), 3)

    # Draw a rectangle
    cv2.rectangle(img, (50, 250), (550, 350), (0, 255, 0), 3)

    # Draw a circle
    cv2.circle(img, (300, 100), 50, (0, 0, 255), -1)

    # Save the image
    cv2.imwrite('output.png', img)
    print("Image saved as 'output.png'")

    # Display the image (will open in a window)
    cv2.imshow('Hello OpenCV', img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

-

import cv2
import numpy as np

def grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size=5):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_threshold, high_threshold):
    """Apply Canny edge detector to the image."""
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image):
    """Apply a region of interest mask to the image."""
    height, width = image.shape
    mask = np.zeros_like(image)
    
    # Define a triangle to mask the region of interest
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, height // 2)
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    """Use Hough Transform to detect lines."""
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def draw_lines(image, lines):
    """Draw detected lines on the image."""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

def process_image(image):
    """Process the image to detect lane lines."""
    gray = grayscale(image)
    blur = gaussian_blur(gray)
    edges = canny(blur, 50, 150)
    roi = region_of_interest(edges)
    lines = hough_lines(roi, 1, np.pi / 180, 15, 40, 20)
    result = draw_lines(image, lines)
    return result

# Read the input image
image = cv2.imread('image.png')
result = process_image(image)

# Display the result
cv2.imshow('Lane Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
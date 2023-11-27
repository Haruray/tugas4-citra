import cv2
import numpy as np


def segmentation(image, size):
    image = cv2.resize(image, (size, size))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    dilated = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 2000
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
    ]

    detected_objects = []

    # Create a mask for each object and extract it
    for i, cnt in enumerate(filtered_contours):
        object_mask = np.zeros_like(gray_img)
        cv2.drawContours(object_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original image
        object_image = cv2.bitwise_and(image, image, mask=object_mask)
        
        

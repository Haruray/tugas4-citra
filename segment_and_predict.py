import cv2
import numpy as np
from ml_classification import svm_classification


def segment_and_predict(image, size, model):
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
    classifications = []
    # Create a mask for each object and extract it
    for i, cnt in enumerate(filtered_contours):
        object_mask = np.zeros_like(gray_img)
        cv2.drawContours(object_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original image
        object_image = cv2.bitwise_and(image, image, mask=object_mask)

        # predict image
        probability, prediction = svm_classification(object_image, model)

        classifications.append(prediction)

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img=image,
            text=f"{prediction} {probability:.2f}",
            org=(x, y - 10),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_4,
        )
    return image, classifications

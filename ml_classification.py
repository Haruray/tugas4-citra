import joblib
import cv2

IMG_SIZE = 220


def preprocess(image):
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))


def svm_classification(image, model):
    image = preprocess(image)
    input = [image.flatten()]

    categories = [
        "SUV",
        "bus",
        "family sedan",
        "fire engine",
        "heavy truck",
        "jeep",
        "minibus",
        "racing car",
        "taxi",
        "truck",
    ]

    prediction = model.predict(input)[0]
    probability = model.predict_proba(input)[0][prediction]
    prediction = categories[prediction]

    return probability, prediction

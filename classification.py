import joblib
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE_ML = 220
IMG_SIZE_DL = 240


def preprocess(image, size=IMG_SIZE_ML):
    return cv2.resize(image, (size, size))


def svm_classification(image, model):
    image = preprocess(image)
    input = [image.flatten()]

    categories = [
        "Bus",
        "Car",
        "Truck",
        "motorcycle",
    ]

    prediction = model.predict(input)[0]
    probability = model.predict_proba(input)[0][prediction]
    prediction = categories[prediction]

    return probability, prediction


def cnn_classification(image, model):
    # classify image from tensorflow keras model
    image = preprocess(image, size=IMG_SIZE_DL)
    image = np.array(image, dtype="float32")
    image = np.expand_dims(image, axis=0)
    categories = [
        "Bus",
        "Car",
        "Truck",
        "motorcycle",
    ]
    prediction = model.predict(image)[0]
    predicted_class = np.argmax(prediction)
    probability = prediction[predicted_class]
    return probability, categories[predicted_class]

    # read image and print output of cnn clasifcation

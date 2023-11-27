from gui import GUI
import joblib
import tensorflow as tf


if __name__ == "__main__":
    model_svm = joblib.load("../model/svc_img_clas_vehicle.sav")
    model_cnn = tf.keras.saving.load_model("../model/model.keras")
    gui = GUI(model_svm, model_cnn)
    gui.show_gui()

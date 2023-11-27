from gui import GUI
import joblib
import tensorflow as tf


if __name__ == "__main__":
    model = joblib.load("model/svc_img_clas_vehicle.sav")
    loaded_model = tf.keras.saving.load_model("model/model.keras")
    gui = GUI(model, loaded_model)
    gui.show_gui()

from gui import GUI
import joblib

if __name__ == "__main__":
    model = joblib.load("model/svc_img_clas_vehicle.sav")
    gui = GUI(model)
    gui.show_gui()

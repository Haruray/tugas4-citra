# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
from segment_and_predict import segment_and_predict
import cv2

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter.filedialog import askopenfilename


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("../assets/frame0")
IMG_SIZE_FOR_GUI = 300
IMG_SIZE_FOR_ML = 220
IMG_SIZE_FOR_DL = 240


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class GUI:
    def __init__(self, model_ml, model_dl) -> None:
        self.canvas = None
        self.original_image = None
        self.processed_image = None
        self.original_image_gui = None
        self.processed_image_gui = None
        self.classification_result = None
        self.model_ml = model_ml
        self.model_dl = model_dl

    def open_file(self):
        file_path = askopenfilename(
            filetypes=[
                ("Image files", "*.jpeg .jpg .png .bmp"),
            ],
        )
        if file_path is not None:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (IMG_SIZE_FOR_GUI, IMG_SIZE_FOR_GUI))
            cv2.imwrite("../output/original.png", image)
            self.original_image = PhotoImage(file="../output/original.png")
            self.canvas.itemconfig(self.original_image_gui, image=self.original_image)

    def process(self, ml, dl):
        if ml:
            if self.original_image is not None:
                image = cv2.imread("../output/original.png")
                image, classifications = segment_and_predict(
                    image, IMG_SIZE_FOR_ML, self.model_ml, is_ml=True, is_dl=False
                )
                image = cv2.resize(image, (IMG_SIZE_FOR_GUI, IMG_SIZE_FOR_GUI))
                cv2.imwrite("../output/processed.png", image)
                self.processed_image = PhotoImage(file="../output/processed.png")
                self.canvas.itemconfig(
                    self.processed_image_gui, image=self.processed_image
                )
                self.canvas.itemconfig(
                    self.classification_result, text="\n".join(classifications)
                )
        else:
            # if is_dl = True
            if self.original_image is not None:
                image = cv2.imread("../output/original.png")
                image, classifications = segment_and_predict(
                    image, IMG_SIZE_FOR_DL, self.model_dl, is_ml=False, is_dl=True
                )
                image = cv2.resize(image, (IMG_SIZE_FOR_GUI, IMG_SIZE_FOR_GUI))
                cv2.imwrite("../output/processed.png", image)
                self.processed_image = PhotoImage(file="../output/processed.png")
                self.canvas.itemconfig(
                    self.processed_image_gui, image=self.processed_image
                )
                self.canvas.itemconfig(
                    self.classification_result, text="\n".join(classifications)
                )

    def show_gui(self):
        window = Tk(screenName="Citra Vehicle Classification")

        window.geometry("1202x638")
        window.configure(bg="#FFFFFF")

        self.canvas = Canvas(
            window,
            bg="#FFFFFF",
            height=638,
            width=1202,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )

        self.canvas.place(x=0, y=0)
        image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        image_1 = self.canvas.create_image(601.0, 319.0, image=image_image_1)

        image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        image_2 = self.canvas.create_image(142.0, 369.0, image=image_image_2)

        button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.open_file(),
            relief="flat",
        )
        button_1.place(
            x=68.0, y=73.0, width=155.4285125732422, height=32.72178649902344
        )

        button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.process(ml=True, dl=False),
            relief="flat",
        )
        button_2.place(x=63.0, y=416.0, width=158.0, height=33.263153076171875)

        button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.process(ml=False, dl=True),
            relief="flat",
        )
        button_3.place(
            x=63.0, y=481.6947326660156, width=158.0, height=33.263153076171875
        )

        image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        self.original_image_gui = self.canvas.create_image(
            537.0, 338.0, image=image_image_3
        )

        image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        self.processed_image_gui = self.canvas.create_image(
            946.0, 338.0, image=image_image_4
        )

        self.canvas.create_text(
            455.0,
            130.0,
            anchor="nw",
            text="Original Image",
            fill="#000000",
            font=("Inter Bold", 23 * -1),
        )

        self.canvas.create_text(
            910.0,
            130.0,
            anchor="nw",
            text="Result",
            fill="#000000",
            font=("Inter Bold", 23 * -1),
        )
        self.classification_result = self.canvas.create_text(
            910.0,
            510.0,
            anchor="nw",
            text="",
            fill="#000000",
            font=("Inter Bold", 20 * -1),
        )
        window.resizable(False, False)
        window.mainloop()

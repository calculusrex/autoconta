import tkinter as tk
# import tk
# import pytesseract as tesseract
# import os
# import numpy as np

from im import tkim_from_cv

from im import rgb2hex
from constants import *

# state_seed['win_w'] = root.winfo_screenwidth()
# state_seed['win_h'] = root.winfo_screenheight() - 50    


class ValidationCanvas(tk.Canvas):
    def __init__(self, master, ocr_box_im, width=256, bg=GRAY):
        self.im = ocr_box_im
        self.im_h, self.im_w = self.im.shape[:2]
        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight() - 100
        if self.im_h > self.im_w:
            self.canv_h = int(round(
                1/3 * screen_h))
            self.canv_w = int(round(
                (self.canv_h * self.im_w) / self.im_h))
        else:
            self.canv_w = int(round(1/3 * screen_w))
            self.canv_h = int(round(
                (self.canv_w * self.im_h) / self.im_w))

        self.scale_factor = self.canv_w / self.im_w
        
        super().__init__(
            master, width=self.canv_w, height=self.canv_h, bg=bg)

        self.tkim = tkim_from_cv(
            self.im, scale_factor=self.scale_factor)
        self.create_image(0, 0, image=self.tkim, anchor=tk.NW)


class DocumentCanvas(tk.Canvas):
    def __init__(self, master, cvim, bg=GRAY, screen_width_percentage=1/3):
        self.im = cvim
        self.im_h, self.im_w = self.im.shape[:2]
        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight() - 100
        if self.im_h > self.im_w * 1.25:
            self.canv_h = screen_h
            self.scale_factor = self.canv_h / self.im_h
            self.canv_w = int(self.im_w * self.scale_factor)
        else:
            self.canv_w = int(round(screen_w * screen_width_percentage))
            self.scale_factor = self.canv_w / self.im_w
            self.canv_h = int(self.im_h * self.scale_factor)

        super().__init__(
            master, width=self.canv_w, height=self.canv_h, bg=bg)

        self.tkim = tkim_from_cv(
            self.im, scale_factor=self.scale_factor)
        self.create_image(
            0, 0, image=self.tkim, anchor=tk.NW)

    def install_grid(self, offset):
        for i in range(1, self.canv_h // offset + 1): # horizontal
            self.create_line(
                0, i*offset, self.canv_w, i*offset,
                fill=GRAY, width=1)
        for i in range(1, self.canv_w // offset + 1): # vertical
            self.create_line(
                i*offset, 0, i*offset, self.canv_h,
                fill=GRAY, width=1)

    def install_crosshairs(self, fill=GRAY, width=1):
        self.horizontal_hair = self.create_line(0, 0, 0, 0, fill=fill, width=width)
        self.vertical_hair = self.create_line(0, 0, 0, 0, fill=fill, width=width)

    def update_crosshairs(self, event):
        self.coords(
            self.horizontal_hair,
            0, event.y, self.canv_w, event.y)
        self.coords(
            self.vertical_hair,
            event.x, 0, event.x, self.canv_h)


def canvas_from_im(global_data, root, im, grid=None):
    h, w = im.shape[:2]
    if h > w * 1.25:
        canv_h = global_data['win_h']
        scale_factor = canv_h / h
        canv_w = int(w * scale_factor)
    else:
        canv_w = global_data['win_w'] // 3
        scale_factor = canv_w / w
        canv_h = int(h * scale_factor)
    canvas = tk.Canvas(
        root,
        width=canv_w,
        height=canv_h,
        bg=GRAY,
    )
    tkim = tkim_from_cv(
        global_data, im, scale_factor=scale_factor)
    canvas.create_image(0, 0, image=tkim, anchor=tk.NW)

    if grid:
        w_offset = canv_w / grid
        h_offset = canv_h / grid
        for i in range(1, grid):
            canvas.create_line(
                0, i*h_offset, canv_w, i*h_offset,
                fill=GRAY, width=1
            )
            canvas.create_line(
                i*w_offset, 0, i*w_offset, canv_h,
                fill=GRAY, width=1
            )            
    
    return canvas, scale_factor

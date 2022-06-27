import os
import cv2 as cv
import numpy as np
from PIL import Image

from pdf2image import convert_from_path, convert_from_bytes

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


project_folder = os.getcwd()
subfolders = ['2022_06_26', '2022_06_27']
input_folder = f'{project_folder}/test_data/levike_facturi_test/{subfolders[0]}'

def compl_fname(fname):
    return f'{input_folder}/{fname}'

full_filenames = list(map(compl_fname,
                          os.listdir(input_folder)))

def read_im(filepath):
    extension = filepath.split('.')[-1].lower()
    if extension == 'pdf':
        im = np.array(
            convert_from_path(filepath)[0])
    else:
        im = cv.imread(filepath)
    return im

def image_data__from_filepaths(filepaths):
    print("reading images ...")
    data = []
    for filepath in filepaths:
        im = read_im(filepath)
        data.append(
            {'og_im': im}
        )
        print("read: ", filepath)
    return data

image_data = image_data__from_filepaths(
    full_filenames)


# ---------------------------------------------------------------------------------
# PIPELINE

# --------------------------------
# WARPING

gui_data = {}
gui_data['workflow'] = 'batch'

def warp__(image):
    


def warping_pipeline__from_image_data(image_data):




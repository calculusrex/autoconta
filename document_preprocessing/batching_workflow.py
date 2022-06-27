import tkinter as tk

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

from frames import WarpingEditor, OrthogonalRotationEditor, FineRotationEditor, RescaleEditor, CropEditor, DenoiseEditor, DilateErodeEditor, ThresholdEditor
from constants import *

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

images = list(map(read_im, full_filenames[:10]))

# def image_data__from_filepaths(filepaths):
#     print("reading images ...")
#     data = []
#     for filepath in filepaths:
#         im = read_im(filepath)
#         data.append(
#             {'og_im': im}
#         )
#         print("read: ", filepath)
#     return data

# image_data = image_data__from_filepaths(
#     full_filenames)


# ---------------------------------------------------------------------------------
# PIPELINE

# --------------------------------
# WARPING

def destruction_procedure(frame, pipeline_data):
    progress_data = {
        'param_data': frame.param_data,
        'processed_image': frame.proc_im,
    }
    pipeline_data['progress_sequence'].append(
        progress_data)
    parent = frame.master
    frame.destroy()
    return parent, pipeline_data

def construction_procedure(parent, pipeline_data):
    if len(pipeline_data['construction_sequence']) == 0:
        print(
            pipeline_data['progress_sequence'])
    else:
        data = pipeline_data['construction_sequence'].pop()
        frame = pipeline_data['frame_constructor'](
            parent, data['original_image'], pipeline_data, data_directed=True)
        frame.grid(
            row=0, column=0, rowspan=FRAME_ROWSPAN)


pipeline_data = {
    'frame_constructor': WarpingEditor,
    'construction_sequence': [],
    'progress_sequence': [],
    'construction_procedure': construction_procedure,
    'destruction_procedure': destruction_procedure,
}

for im in images:
    pipeline_data['construction_sequence'].append({
        'original_image': im,
    })

root = tk.Tk()
root.bind('<Control-q>', lambda event: root.destroy())
construction_procedure(root, pipeline_data)


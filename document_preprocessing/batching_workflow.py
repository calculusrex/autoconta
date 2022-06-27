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
from im import display_cv

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
    data = {
        'fname': filepath.split('/')[-1],
        'im': im
    }
    return data

images_data = list(map(read_im, full_filenames[:2]))

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
    # display_cv(frame.proc_im)
    progress_data = {
        'param_data': frame.param_data,
        'processed_image': frame.proc_im,
        'original_image_fname': frame.og_im_fname,
    }
    pipeline_data['progress_sequence'].append(
        progress_data)
    parent = frame.master
    frame.destroy()
    return parent, pipeline_data

def construction_procedure(parent, pipeline_data):
    if len(pipeline_data['construction_sequence']) == 0 and len(pipeline_data['pipeline_constructor_sequence']) == 0:
        for document_data in pipeline_data['progress_sequence']:
            out_fpath = "/".join([
                OUTPUT_FOLDER,
                ".".join([
                    document_data['original_image_fname'].split(
                        ".")[0],
                    "png"])])
            cv.imwrite(
                out_fpath,
                document_data['processed_image'])
            print(f"written: {out_fpath}")

    else:
        if len(pipeline_data['pipeline_constructor_sequence']) == 0:
            pipeline_sequence_data = pipeline_data['construction_sequence'].pop()
            pipeline_data['pipeline_constructor_sequence'] = pipeline_sequence_data[
                'pipeline_constructors']
            image_data = current_sequence_data['image_data']
            pipeline_data['current_original_image_data'] = image_data
            pipeline_data['current_improc_progress'] = []

        image_data = pipeline_data['current_improc_progress'][-1]
        constructor = pipeline_data['pipeline_constructor_sequence'].pop()
        frame = constructor(
            parent, image_data, pipeline_data, data_directed=True)
        frame.grid(
            row=0, column=0, rowspan=FRAME_ROWSPAN)
        
        

frame_constructors =  [
        WarpingEditor, OrthogonalRotationEditor,
        FineRotationEditor, RescaleEditor, CropEditor
]
frame_constructors.reverse()

pipeline_data = {
    'frame_constructors': frame_constructors,
    'construction_sequence': []
    'progress_sequence': [],
    'construction_procedure': construction_procedure,
    'destruction_procedure': destruction_procedure,
}

for image_data in images_data:
    pipeline_data['construction_sequence'].append(
        pipeline_data.append({
            'image_data': image_data,
            'pipeline_constructors': frame_constructors.copy()
        }))

# pipeline_data = {
#     'frame_constructor': WarpingEditor,
#     'construction_sequence': images_data,
#     'progress_sequence': [],
#     'construction_procedure': construction_procedure,
#     'destruction_procedure': destruction_procedure,
# }


root = tk.Tk()
root.bind('<Control-q>', lambda event: root.destroy())
construction_procedure(root, pipeline_data)


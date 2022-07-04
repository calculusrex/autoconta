import tkinter as tk

import os
import cv2 as cv
import numpy as np
from PIL import Image
from copy import deepcopy
import functools as ft
import json

from frames import OCR, OCRROI, DocumentDescriber
from constants import *
from im import display_cv

from document_preprocessing import folder_data__
from human import human_guided_im_transform, human_guided_im_filter

def init_gui():    
    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())
    return root

# :: (TK, string) -> [im]
def preprocess_documents(input_folder):
    cvims = load_cv_images(
        input_folder)
    transformed_cvims = list(
        map(human_guided_im_transform,
            cvims))
    filtered_cvims = list(
        map(human_guided_im_filter,
            transformed_cvims))
    return filtered_cvims

def data_entry__extraction_phase(input_folder):
    root = init_gui()
    improc_doc_data = preprocess_documents(
        root, input_folder)
    extracted_data = extract_data(
        root, improc_doc_data)
    export_to_pag_interface(
        extracted_data)

if __name__ == '__main__':
    print('top_down_functional.py\n')

    project_folder = os.getcwd()

    subfolders = ['2022_06_26', '2022_06_27']
    out_subfolders = list(map(
        lambda cs: "__".join([cs, "out"]),
        subfolders))

    subfolder_path = "/".join([
        'test_data',
        'levike_facturi_test',
        out_subfolders[0]])

    input_folder = "/".join([
        project_folder, subfolder_path])

    data_entry__extraction_phase(input_folder)

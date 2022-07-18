import tkinter as tk

import os
import cv2 as cv
import numpy as np
from PIL import Image
from copy import deepcopy
import functools as ft
import json

from functional_frames import WarpingEditor, OrthogonalRotationEditor, FineRotationEditor, RescaleEditor, CropEditor, DenoiseEditor, DilateErodeEditor, ThresholdEditor, OCRROI, OCR, OrthogonalLineEraser, OCRValidation
from history import load_historical_data


from constants import *

stage_keys_x_constructors = {
    'warping': WarpingEditor,
    'orthogonal_rotation': OrthogonalRotationEditor,
    'fine_rotation': FineRotationEditor,
    'rescale': RescaleEditor,
    'crop': CropEditor,
    'denoise': DenoiseEditor,
    'dilate_erode': DilateErodeEditor,
    'threshold': ThresholdEditor,
}

transform_stage_sequence = [
#    'warping',
#    'orthogonal_rotation',
    'fine_rotation',
    'rescale',
    'crop',
]

transform_stage_data = {
    # 'warping': {'cnstrctr_params': {}},
    # 'orthogonal_rotation': {'cnstrctr_params': {}},
    'fine_rotation': {'cnstrctr_params': {}},
    'rescale': {'cnstrctr_params': {}},
    'crop': {'cnstrctr_params': {}},
}

transform_pipeline_data = {
    'sequence': transform_stage_sequence,
    'data': transform_stage_data,
}
    
filter_stage_sequence = [
    'denoise',
    'threshold',
    'dilate_erode',
]

filter_stage_data = {
    'denoise': {'cnstrctr_params': {}},
    'threshold': {'cnstrctr_params': {}},
    'dilate_erode': {'cnstrctr_params': {}},
}

filter_pipeline_data = {
    'sequence': filter_stage_sequence,
    'data': filter_stage_data,
}

def init_gui():    
    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())
    return root

def deploy_(frame):
    frame.grid(
        row=0, column=0, rowspan=FRAME_ROWSPAN)

def deploy_gui(im, cnstrctr, input_param_data):
    root = init_gui()
    state_data = {
        'input_params': input_param_data,
    }
    frame = cnstrctr(root, state_data, im)
    deploy_(frame)
    root.mainloop()
    # AFTER GUI DESTRUCTION ---------------------
    im = state_data['out_im']
    op_params = state_data['op_params']
    return im, op_params

def deploy_human_guided_im_op_pipeline(
        im, pipeline_data):
    pipeline_data = deepcopy(
        pipeline_data)
    for key in pipeline_data['sequence']:
        cnstrctr = stage_keys_x_constructors[key]
        cnstrctr_params = pipeline_data[
            'data'][key]['cnstrctr_params']
        im, op_params = deploy_gui(
            im, cnstrctr, cnstrctr_params)
        pipeline_data[
            'data'][key]['op_params'] = op_params
    return im, pipeline_data

def deploy_human_guided_im_transform_pipeline(cvim):
    return deploy_human_guided_im_op_pipeline(
        cvim, transform_pipeline_data)

def deploy_human_guided_im_filter_pipeline(cvim):
    return deploy_human_guided_im_op_pipeline(
        cvim, filter_pipeline_data)

# def deploy_human_guided_doc_roi_selection(
#         doc_key, doc_data, elem_keys):
#     im = doc_data[doc_key]['preproc_im']
#     im, out_data = deploy_gui(
#         im, OCRROI, {'roi_keys': elem_keys})
#     return out_data['params']

def human_select_rois(
        doc_dat, section_keys):
    im = doc_dat['preproc_im']
    im, out_data = deploy_gui(
        im, OCRROI, {'roi_keys': section_keys})
    return out_data['params']

def human_extract_elems(
        im, elems_to_extract):
    im, out_data = deploy_gui(
        im, OCR, elems_to_extract)
    return out_data['params']

def human_erase_orthogonal_lines(im):
    im, out_data = deploy_gui(
        im, OrthogonalLineEraser, {})
    return im, out_data

def human_validate_elem(im, elem_data, hist_data):
    input_data = {
        'element': elem_data,
        'historical': hist_data,
    }
    validated_data = deploy_gui(
        im, OCRValidation, input_data)
    return validated_data



if __name__ == '__main__':
    print('human.py\n')

    project_folder = os.getcwd()

    subfolders = ['2022_06_26', '2022_06_27']

    subfolder_path = "/".join([
        'test_data',
        'levike_facturi_test',
        subfolders[0]])

    history_path = "/".join([
        'test_data',
        'levike_facturi_test',
        'saga_history'])

    bills_fpath = "/".join([
        history_path,
        'intrari_facturi_ardeleanu__20_06_2022.csv'])

    items_fpath = "/".join([
        history_path,
        'intrari_articole_ardeleanu__20_06_2022.csv'])

    working_folder = "/".join([
        project_folder, subfolder_path])

    folder_data = {
        'project_folder': project_folder,
        'subfolders': subfolders,
        'working_folder': working_folder,
        'subfolder_path': subfolder_path,
        'history_path': history_path,
        'bills_fpath': bills_fpath,
        'items_fpath': items_fpath,
    }
    

    hist = load_historical_data(folder_data)

    elem_im = cv.imread('elem_im.png')
    with open('elem_data.json', 'r') as f:
        elem_data = json.load(f)

    print(elem_data)

    root = init_gui()
    state_data = {
        'input_params': {
            'element': elem_data,
            'historical': hist,
        }
    }
    frame = OCRValidation(
        root, state_data, elem_im)

    deploy_(frame)

    

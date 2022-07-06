import tkinter as tk

import os
import cv2 as cv
import numpy as np
from PIL import Image
from copy import deepcopy
import functools as ft
import json

from functional_frames import WarpingEditor, OrthogonalRotationEditor, FineRotationEditor, RescaleEditor, CropEditor, DenoiseEditor, DilateErodeEditor, ThresholdEditor, OCRROI, OCR

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

def deploy_human_guided_doc_roi_selection(
        doc_dat, section_keys):
    im = doc_dat['preproc_im']
    im, out_data = deploy_gui(
        im, OCRROI, {'roi_keys': section_keys})
    return out_data['params']

def deploy_human_guided_doc_elem_extraction(
        im, elems_to_extract):
    im, out_data = deploy_gui(
        im, OCR, elems_to_extract)
    return out_data['params']

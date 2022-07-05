
import os
import cv2 as cv
import numpy as np
from PIL import Image
from copy import deepcopy
import functools as ft
import json
from collections import OrderedDict

from functional_frames import WarpingEditor, OrthogonalRotationEditor, FineRotationEditor, RescaleEditor, CropEditor, DenoiseEditor, DilateErodeEditor, ThresholdEditor


transform_stages = OrderedDict([
    ('warping', WarpingEditor),
    ('orthogonal_rotation', OrthogonalRotationEditor),
    ('fine_rotation', FineRotationEditor),
    ('rescale', RescaleEditor),
    ('crop', CropEditor),
])

filter_stages = OrderedDict([
    ('denoise', DenoiseEditor),
    ('dilate_erode', DilateErodeEditor),
    ('threshold', ThresholdEditor),
])

def deploy_(frame):
    frame.grid(
        row=0, column=0, rowspan=FRAME_ROWSPAN)

def invoque_gui(im, cnstrctr, input_param_data):
    root = tk.Tk()
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

def invoque_human_guided_im_op_pipeline(
        im, pipeline_data):
    for key in pipeline_data['op_sequence'].keys():
        cnstrctr = pipeline_data[key]['constructor']
        cnstrctr_params = pipeline_data[key][
            'cnstrctr_params']
        im, trnsfrm_params = invoque_gui(
            im, cnstrctr, cnstrctr_params)
        pipeline_data[key]['op_params'] = trnsfrm_params
    return im, pipeline_data

# def human_guided_im_transform(im):
#     metadata = {}
#     metadata['transform_sequence'] = list(
#         transform_stages.keys())
#     metadata['trnsfrm_params'] = {}
#     for key, cnstrctr in transform_stages.keys():
#         im, trnsfrm_params = invoque_gui(
#             im, cnstrctr) ## !!! PARAM DATA REQUIRED
#         metadata['trnsfrm_params'][key] = trnsfrm_params
#     return im, metadata
    

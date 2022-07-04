
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

def perform_transform(im, cnstrctr):
    root = tk.Tk()
    state_data = {}
    frame = cnstrctr(root, state_data, im)
    deploy_(frame)
    root.mainloop()
    return state_data['out_im']

def human_guided_im_transform(im):
    metadata = {}
    metadata['transform_sequence'] = list(
        transform_stages.keys())
    metadata['trnsfrm_params'] = {}
    for key, cnstrctr in transform_stages.keys():
        im, trnsfrm_params = perform_transform(
            im, cnstrctr)
        metadata['trnsfrm_params'][key] = trnsfrm_params
    return im, metadata
    

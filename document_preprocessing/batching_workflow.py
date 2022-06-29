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

# def destruction_procedure(frame, pipeline_data):
#     # display_cv(frame.proc_im)
#     progress_data = {
#         'param_data': frame.param_data,
#         'processed_image': frame.proc_im,
#         'original_image_fname': frame.og_im_fname,
#     }
#     pipeline_data['progress_sequence'].append(
#         progress_data)
#     parent = frame.master
#     frame.destroy()
#     return parent, pipeline_data

# def construction_procedure(parent, pipeline_data):
#     if len(pipeline_data['construction_sequence']) == 0 and len(pipeline_data['pipeline_constructor_sequence']) == 0:
#         for document_data in pipeline_data['progress_sequence']:
#             out_fpath = "/".join([
#                 OUTPUT_FOLDER,
#                 ".".join([
#                     document_data['original_image_fname'].split(
#                         ".")[0],
#                     "png"])])
#             cv.imwrite(
#                 out_fpath,
#                 document_data['processed_image'])
#             print(f"written: {out_fpath}")

#     else:
#         if len(pipeline_data['pipeline_constructor_sequence']) == 0:
#             pipeline_sequence_data = pipeline_data['construction_sequence'].pop()
#             pipeline_data['pipeline_constructor_sequence'] = pipeline_sequence_data[
#                 'pipeline_constructors']
#             image_data = current_sequence_data['image_data']
#             pipeline_data['current_original_image_data'] = image_data
#             pipeline_data['current_improc_progress'] = []

#         image_data = pipeline_data['current_improc_progress'][-1]
#         constructor = pipeline_data['pipeline_constructor_sequence'].pop()
#         frame = constructor(
#             parent, image_data, pipeline_data, data_directed=True)
#         frame.grid(
#             row=0, column=0, rowspan=FRAME_ROWSPAN)
        
        

# frame_constructors =  [
#         WarpingEditor, OrthogonalRotationEditor,
#         FineRotationEditor, RescaleEditor, CropEditor
# ]
# frame_constructors.reverse()

# pipeline_data = {
#     'frame_constructors': frame_constructors,
#     'construction_sequence': []
#     'progress_sequence': [],
#     'construction_procedure': construction_procedure,
#     'destruction_procedure': destruction_procedure,
# }

# for image_data in images_data:
#     pipeline_data['construction_sequence'].append(
#         pipeline_data.append({
#             'image_data': image_data,
#             'pipeline_constructors': frame_constructors.copy()
#         }))

# # pipeline_data = {
# #     'frame_constructor': WarpingEditor,
# #     'construction_sequence': images_data,
# #     'progress_sequence': [],
# #     'construction_procedure': construction_procedure,
# #     'destruction_procedure': destruction_procedure,
# # }


def file_data__from_fname_x_folder(fname, folder):
    return {
        'fname': fname,
        'folder': folder,
        'fpath': f'{folder}/{fname}',
    }

def read_im(filepath):
    extension = filepath.split('.')[-1].lower()
    if extension == 'pdf':
        im = np.array(
            convert_from_path(filepath)[0])
    else:
        im = cv.imread(filepath)
    return im

preproc__constructor_from_stage_name = dict(
    zip(['warping', 'orthogonal_rotation',
         'fine_rotation', 'rescale', 'crop'],
        [WarpingEditor, OrthogonalRotationEditor,
         FineRotationEditor, RescaleEditor, CropEditor]))

# filename -> preproc_im
# preproc_im :: (fname, proc_im, [(proc_name, proc_params)])

transform_preproc_stages = [
    'warping' , 'orthogonal_rotation', 'fine_rotation', 'rescale', 'crop']

def doc_data__from__fdat_x_preproc_stgs(
        file_data, preproc_stages):
    fname = file_data['fname']
    folder = file_data['folder']
    fpath = file_data['fpath']
    extension = fname.split('.')[-1]
    fname_wo_extension = "".join(
        fname.split('.')[:-1])

    file_data = {
        'fname': fname, 'fpath': fpath,
        'input_folder': folder,
        'fname_wo_extension': fname_wo_extension,
        'extension': extension
    }

    im = read_im(fpath)
    pending_procs = preproc_stages.copy()
    pending_procs.reverse()
    proc_data = {
        'sequence': preproc_stages,
        'pending': pending_procs,
        'current': 'interproc',
        'finished': [],
        'im': im,
    }

    return {
        'file': file_data,
        'proc': proc_data
    }

# def trnsfrm_ppln__destructor(frame):
#     state_data = frame.state_data
#     state_data['finished_proc_data'].append(
#         {'proc_name': frame.proc_name,
#          'proc_params': frame.param_data})
#     gui_data = {'master': frame.master}
#     frame.destroy()
#     return state_data, gui_data

# def trnsfrm_ppln__constructor(
#         state_data, gui_data):
#     frame_constructor = preproc__constructor_from_stage_name[
#         state_data[
#             'pending_procs'].pop()]
#     frame = frame_constructor(
#         state_data, gui_data)
#     frame.grid(
#         row=0, column=0, rowspan=FRAME_ROWSPAN)    

def destruct_frame(frame):
    state_data = frame.state_data
    doc_dat = state_data['cw_doc_data']
    current_proc_dat = {
        'key': frame.proc_key,
        'params': frame.param_data}
    doc_dat['im'] = frame.proc_im
    doc_dat['proc']['current'] = 'interproc'
    doc_dat['proc']['finished'].append(
        current_proc_dat)
    gui_data = {'master': frame.master}
    frame.destroy()
    return state_data, gui_data

def construct_frame(state_data, gui_data):
    doc_dat = state_data['cw_doc_data']
    key = doc_dat['proc']['pending'].pop()
    doc_dat['proc']['current'] = key
    frame_constructor = preproc__constructor_from_stage_name[
        doc_dat['proc']['current']]
    frame = frame_constructor(state_data, gui_data)
    frame.grid(
        row=0, column=0, rowspan=FRAME_ROWSPAN)

def trnsfrm_ppln__collector(
        state_data, gui_data):
    pass

def trnsfrm_ppln__progressor(
        pipeline_data, gui_data):
    pass

# def is_curr_document_done(state_data):
#     state_data['x']

def trnsfrm_ppln__control_shift(frame):
    dest = trnsfrm_ppln__destructor
    state_data, gui_data = dest(
        frame)
    if state_data['pending_procs']: # not empty
        trnsfrm_ppln__constructor(
            state_data, gui_data)
    else:
        preproc_im_data = trnsfrm_ppln__collector(
            state_data, gui_data)

def state_data__from_folder_x_preproc_stages(
        folder, preproc_stages):
    print('loading images and stuff...')
    doc_dat_dat__ = doc_data__from__fdat_x_preproc_stgs
    pending_doc_data = []
    for fname in os.listdir(input_folder):
        pending_doc_data.append(
            doc_dat__(
                file_data__from_fname_x_folder(
                    fname, input_folder),
                preproc_stages))
    pending_doc_data.reverse()
    return {
        'pending_doc_data': pending_doc_data,
        'cw_doc_data': {},
        'finished_doc_data': [],
        'control_shift': trnsfrm_ppln__control_shift
    }

if __name__ == '__main__':
    print('batching_workflow')

    project_folder = os.getcwd()

    subfolders = ['2022_06_26', '2022_06_27']

    input_folder = "/".join([
        project_folder, 'test_data',
        'levike_facturi_test', subfolders[0]
    ])

    sd = state_data__from_folder_x_preproc_stages
    state_data = sd(
        input_folder, transform_preproc_stages)

    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())


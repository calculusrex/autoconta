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

# ----------------------------------------------------------------
# PIPELINE

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

def is_curr_doc_ppln_done(state_data):
    doc_dat = state_data['cw_doc_data']
    return not(bool(
        doc_dat['proc']['pending']))

def is_batch_done(state_data):
    return not(bool(
        state_data['pending_doc_data']))

def destruct_frame(frame):
    state_data = frame.state_data
    doc_dat = state_data['cw_doc_data']
    current_proc_dat = {
        'key': frame.proc_key,
        'params': frame.param_data}
    doc_dat['proc']['im'] = frame.proc_im
    doc_dat['proc']['current'] = 'interproc'
    doc_dat['proc']['finished'].append(
        current_proc_dat)
    gui_data = {'master': frame.master}
    frame.destroy()
    return state_data, gui_data

def deploy_next_ppln_frame(state_data, gui_data):
    doc_dat = state_data['cw_doc_data']
    key = doc_dat['proc']['pending'].pop()
    doc_dat['proc']['current'] = key
    frame_constructor = preproc__constructor_from_stage_name[
        doc_dat['proc']['current']]
    frame = frame_constructor(state_data, gui_data)
    frame.grid(
        row=0, column=0, rowspan=FRAME_ROWSPAN)

def stash_current_doc_data(state_data, gui_data):
    doc_dat = state_data['cw_doc_data']
    state_data['finished_doc_data'].append(
        doc_dat)
    state_data['cw_doc_data'] = 'interdoc'
    
def process_next_doc(state_data, gui_data):
    doc_dat = state_data['pending_doc_data'].pop()
    state_data['cw_doc_data'] = doc_dat
    deploy_next_ppln_frame(
        state_data, gui_data)
    
def control_shift(frame):
    state_data, gui_data = destruct_frame(
        frame)
    if not(is_curr_doc_ppln_done(state_data)):
        deploy_next_ppln_frame(
            state_data, gui_data)
    elif not(is_batch_done(state_data)):
        stash_current_doc_data(
            state_data, gui_data)
        process_next_doc(
            state_data, gui_data)
    else:
        stash_current_doc_data(
            state_data, gui_data)
        write_out(state_data)
    
def trnsfrm_ppln__collector(
        state_data, gui_data):
    pass

def trnsfrm_ppln__progressor(
        pipeline_data, gui_data):
    pass

def state_data__from_folder_x_preproc_stages(
        input_folder, preproc_stages, n_files=0):
    print('loading images and stuff...')
    doc_dat__ = doc_data__from__fdat_x_preproc_stgs
    pending_doc_data = []
    fnames = os.listdir(input_folder)
    if n_files != 0: # DEVELOPMENT / DEBUGGING
        fnames = list(
            np.random.choice(
                fnames, (n_files)))
    for fname in fnames:
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
        'control_shift': control_shift
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
        input_folder, transform_preproc_stages,
        n_files=3)

    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())

    gui_data = {'master': root}

    process_next_doc(
        state_data, gui_data)

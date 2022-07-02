import tkinter as tk

import os
import cv2 as cv
import numpy as np
from PIL import Image
from copy import deepcopy
import functools as ft
import json

from pdf2image import convert_from_path, convert_from_bytes

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

from frames import WarpingEditor, OrthogonalRotationEditor, FineRotationEditor, RescaleEditor, CropEditor, DenoiseEditor, DilateErodeEditor, ThresholdEditor

from constants import *
from im import display_cv


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

# ----------------------------------------------------------------
# PIPELINE

def folder_data__(project_folder, subfolder_path):
    input_folder = "/".join([
        project_folder, subfolder_path
    ])
    return {
        'project_folder': project_folder,
        'subfolder_chain': subfolder_path,
        'input_folder': input_folder,
    }

def file_data__from_fname_x_folder(fname, folder_data):
    input_folder = folder_data['input_folder']
    return {
        'fname': fname,
        'folder': folder_data,
        'fpath': f'{input_folder}/{fname}',
    }

def read_im(filepath):
    extension = filepath.split('.')[-1].lower()
    if extension == 'pdf':
        im = np.array(
            convert_from_path(filepath)[0])
    else:
        im = cv.imread(filepath)
    return im

def constructor_from_stage_key(key):
    return stage_keys_x_constructors[key]

# filename -> preproc_im
# preproc_im :: (fname, proc_im, [(proc_name, proc_params)])

transform_preproc_stages = [
    'warping' , 'orthogonal_rotation',
    'fine_rotation', 'rescale', 'crop']

filter_preproc_stages = [
    'denoise' , 'dilate_erode', 'threshold']

document_preprocessing_pipeline = [
    transform_preproc_stages, filter_preproc_stages,
]

def doc_data__from__fdat_x_pipelines(
        file_data, pipelines):
    fname = file_data['fname']
    folder_data = file_data['folder']
    fpath = file_data['fpath']
    extension = fname.split('.')[-1]
    fname_wo_extension = "".join(
        fname.split('.')[:-1])

    file_data = {
        'fname': fname, 'fpath': fpath,
        'folder': folder_data,
        'fname_wo_extension': fname_wo_extension,
        'extension': extension
    }

    im = read_im(fpath)
    pending_pipelines = deepcopy(pipelines)
    pending_pipelines.reverse()
    pending_procs = pending_pipelines.pop()
    pending_procs.reverse()
    proc_data = {
        'pipelines': pipelines,
        'pending_pipelines': pending_pipelines,
        'pending_procs': pending_procs,
        'current': 'interproc',
        'finished': [],
        'im': im,
    }

    return {
        'file': file_data,
        'proc': proc_data
    }

def state_data__from_folder_x_pipelines(
        input_folder_data, pipelines, n_files=0):
    print('loading images and stuff...')
    pending_doc_data = []
    fnames = os.listdir(
        input_folder_data['input_folder'])
    if n_files != 0: # DEVELOPMENT / DEBUGGING
        fnames = list(
            np.random.choice(
                fnames, (n_files)))
    doc_dat__ = doc_data__from__fdat_x_pipelines
    for fname in fnames:
        pending_doc_data.append(
            doc_dat__(
                file_data__from_fname_x_folder(
                    fname, input_folder_data),
                pipelines))
    pending_doc_data.reverse()
    return {
        'pending_doc_data': pending_doc_data,
        'cw_doc': {},
        'finished_doc_data': [],
        'control_shift': control_shift
    }

def is_curr_doc_ppln_done(state_data):
    doc_dat = state_data['cw_doc']
    return not(bool(
        doc_dat['proc']['pending_procs']))

def is_batch_done(state_data):
    return not(bool(
        state_data['pending_doc_data']))

def are_no_pending_procs(document_data):
    return not(
        bool(
            document_data['proc']['pending_procs']))

def are_all_pplns_done(doc_dat):
    return not(
        bool(doc_dat['proc']['pending_pipelines']))

def batch__are_all_pplns_done(state_data):
    doc_data = state_data['finished_doc_data']
    return ft.reduce(
        lambda a, b: a and b,
        map(are_all_pplns_done,
            doc_data),
        True)

def destruct_frame(frame):
    state_data = frame.state_data
    doc_dat = state_data['cw_doc']
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

def load_next_pipeline(document_data):
    # assert(
    #     not(bool(document_data['proc']['pending_procs'])))
    ppln = document_data['proc']['pending_pipelines'].pop()
    ppln.reverse()
    document_data['proc']['pending_procs'] = ppln

def deploy_next_ppln_frame(state_data, gui_data):
    doc_dat = state_data['cw_doc']
    if are_no_pending_procs(doc_dat):
        load_next_pipeline(doc_dat)
    key = doc_dat['proc']['pending_procs'].pop()
    doc_dat['proc']['current'] = key
    frame_constructor = constructor_from_stage_key(
        doc_dat['proc']['current'])
    frame = frame_constructor(state_data, gui_data)
    frame.grid(
        row=0, column=0, rowspan=FRAME_ROWSPAN)

def stash_current_doc_data(state_data, gui_data):
    doc_dat = state_data['cw_doc']
    state_data['finished_doc_data'].append(
        doc_dat)
    state_data['cw_doc'] = 'interdoc'
    
def process_next_doc(state_data, gui_data):
    doc_dat = state_data['pending_doc_data'].pop()
    state_data['cw_doc'] = doc_dat
    deploy_next_ppln_frame(
        state_data, gui_data)
    
def batch__load_next_pipeline(state_data):
    doc_data = state_data['finished_doc_data']
    for doc_dat in doc_data:
        load_next_pipeline(doc_dat)
    doc_data.reverse()
    state_data['pending_doc_data'] = doc_data
    state_data['finished_doc_data'] = []
    
def process_next_ppln(state_data, gui_data):
    batch__load_next_pipeline(state_data)
    process_next_doc(
        state_data, gui_data)

def output_root_folder(doc_data):
    out_folder_path = "/".join([
        doc_data['file']['folder']['project_folder'],
        "__".join([
            doc_data['file']['folder']['subfolder_chain'],
            "out"])])
    if not(os.path.isdir(out_folder_path)):
        os.makedirs(out_folder_path)
    return out_folder_path

def output_doc_folders(doc_data, root_folder):
    out_im_folder_path = "/".join([
        root_folder, "images"])
    out_meta_folder_path = "/".join([
        root_folder, "metadata"])
    for folder_path in [
            root_folder,
            out_im_folder_path, out_meta_folder_path]:
        if not(os.path.isdir(folder_path)):
            os.makedirs(folder_path)
    return {
        'im': out_im_folder_path,
        'meta': out_meta_folder_path,
    }        

def output_doc_paths(doc_data, root_folder):
    folders = output_doc_folders(
        doc_data, root_folder)
    im_fname = ".".join([
        "__".join([
            doc_data['file']['fname_wo_extension'],
            "im_out"]),
        "png"])
    meta_fname = ".".join([
        "__".join([
            doc_data['file']['fname_wo_extension'],
            "improc_metadata"]),
        "json"])
    im_fpath = "/".join([
        folders['im'], im_fname])
    meta_fpath = "/".join([
        folders['meta'], meta_fname])
    return {
        'im': im_fpath,
        'meta': meta_fpath
    }

def document_metadata(doc_data, output_files_metadata):
    proc = doc_data['proc']
    improc_data = {
        'pipelines': proc['pipelines'],
        'pending_pipelines': proc['pending_pipelines'],
        'pending_procs': proc['pending_procs'],
        'current': proc['current'],
        'finished': proc['finished'],
    }
    return {
        'input_file': doc_data['file'],
        'output_files': output_files_metadata,
        'improc': improc_data,
    }        

def write_out(state_data):
    out_root_folder = output_root_folder(
        state_data[
            'finished_doc_data'][0])
    for doc_data in state_data['finished_doc_data']:
        out_file_paths = output_doc_paths(
            doc_data, out_root_folder)
        metadata = document_metadata(
            doc_data, {
                'root_folder': out_root_folder,
                'paths': out_file_paths})
        cv.imwrite(
            out_file_paths['im'], doc_data['proc']['im'])
        with open(out_file_paths['meta'], 'w') as f:
            f.write(
                json.dumps(metadata))
    
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
    elif not(batch__are_all_pplns_done(state_data)):
        stash_current_doc_data(
            state_data, gui_data)
        process_next_ppln(
            state_data, gui_data)
    else:
        stash_current_doc_data(
            state_data, gui_data)
        write_out(state_data)



if __name__ == '__main__':
    print('batching_workflow')

    project_folder = os.getcwd()

    subfolders = ['2022_06_26', '2022_06_27']

    subfolder_path = "/".join([
        'test_data',
        'levike_facturi_test',
        subfolders[0]])

    input_folder_data = folder_data__(
        project_folder, subfolder_path)

    sd = state_data__from_folder_x_pipelines
    state_data = sd(
        input_folder_data, document_preprocessing_pipeline,
        n_files=4)

    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())

    gui_data = {'master': root}

    process_next_doc(
        state_data, gui_data)

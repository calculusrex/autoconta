import tkinter as tk

import os
import cv2 as cv
import numpy as np
from PIL import Image
from copy import deepcopy
import functools as ft
import json

from functional_frames import OCR, OCRROI, DocumentDescriber
from constants import *
from im import display_cv

from document_preprocessing import folder_data__
from load_documents import has_extension

stage_keys_x_constructors = {
    # 'ocr': OCR,
    'ocr': OCRROI,
    'document_describer': DocumentDescriber,
}

def constructor_from_key(key):
    return stage_keys_x_constructors[key]

ocr_pipeline = ['ocr']

ocr_proc_params = {
    'roi_keys': {
        'header': [
            'furnizor', 'nr_doc', 'dată_emitere'],
        'body': [
            'denumire_produs/serviciu', 'cantitate',
            'val_fără_tva', 'preţ_vânzare'],
        'footer': ['total'],
#        'auxilliary': ['chitanţă']
    }
}

ocr_proc_param_path = {
    'furnizor': ['roi_keys', 'header'],
    'nr_doc': ['roi_keys', 'header'],
    'dată_emitere': ['roi_keys', 'header'],
    'denumire_produs/serviciu': ['roi_keys', 'body'],
    'cantitate': ['roi_keys', 'body'],
    'val_fără_tva': ['roi_keys', 'body'],
    'preţ_vânzare': ['roi_keys', 'body'],
    'total': ['roi_keys', 'footer'],
    'chitanţă': ['roi_keys', 'auxilliary'],
}


def folder_data__(project_folder, subfolder_path):
    input_folder = "/".join([
        project_folder, subfolder_path
    ])
    return {
        'project_folder': project_folder,
        'subfolder_chain': subfolder_path,
        'images': "/".join([
            input_folder, "images"]),
        'metadata': "/".join([
            input_folder, "metadata"]),
    }

def doc_fnames__(input_folder_data):
    image_fnames = list(
        filter(has_extension('png'),
               os.listdir(
                   input_folder_data['images'])))
    image_fnames.sort()

    doc_fnames__from_image_files = list(
        map(lambda cs: cs.split('__')[0],
            image_fnames))

    metadata_fnames = list(filter(
        has_extension('json'),
        os.listdir(
            input_folder_data['metadata'])))
    metadata_fnames.sort()
    
    doc_fnames__from_metadata_files = list(
        map(lambda cs: cs.split('__')[0],
            metadata_fnames))

    doc_fnames__from_image_files.sort()
    doc_fnames__from_metadata_files.sort()
    are_same_filenames_in_both_folders = ft.reduce(
        lambda a, b: a and b,
        map(lambda tup: tup[0] == tup[1],
            zip(doc_fnames__from_image_files,
                doc_fnames__from_metadata_files)),
        True)
    assert(are_same_filenames_in_both_folders)
    
    return {
        'documents': doc_fnames__from_image_files,
        'images': image_fnames,
        'metadata': metadata_fnames,
    }

def transposed_doc_fnames__(input_folder_data):
    doc_fnames = doc_fnames__(input_folder_data)
    data = {}
    for doc_key in doc_fnames['documents']:
        im_fname = next(filter(
            lambda cs: cs.split('__')[0] == doc_key,
            doc_fnames['images']))
        meta_fname = next(filter(
            lambda cs: cs.split('__')[0] == doc_key,
            doc_fnames['metadata']))
        im_fpath = "/".join([
            input_folder_data['images'], im_fname])
        meta_fpath = "/".join([
            input_folder_data['metadata'], meta_fname])
        data[doc_key] = {
            'im_fname': im_fname, 'meta_fname': meta_fname,
            'im_fpath': im_fpath, 'meta_fpath': meta_fpath, 
        }
    return data

def doc_data__(input_folder_data):
    transposed_doc_fnames = transposed_doc_fnames__(
        input_folder_data)
    data = {}
    for key in transposed_doc_fnames.keys():
        meta_fpath = transposed_doc_fnames[key]['meta_fpath']
        with open(meta_fpath, 'r') as f:
            metadata = json.load(f)
        doc_dat = {}
        doc_dat['fname'] = transposed_doc_fnames[key]
        doc_dat['im'] = cv.imread(
            transposed_doc_fnames[key]['im_fpath'])
        doc_dat['meta'] = metadata
        data[key] = doc_dat
    return data

def init_state_data__from_doc_data(
        doc_data, pipeline, param_data):
    docs = []
    for key in doc_data.keys():
        docs.append(doc_data[key])
    docs.reverse()
    return {
        'pending_docs': docs,
        'finished_docs': [],
        'pipeline': pipeline,
        'proc_params': param_data,
        'cw_doc': 'interdoc',
        'control_shift': control_shift,
    }

def destruct_frame(frame):
    state_data = frame.state_data
    extracted_proc_data = {
        'key': frame.proc_key,
        'params': frame.param_data,
        'extracted': frame.extracted_data}
    cw_doc = state_data['cw_doc']
    cw_doc['finished_procs'].append(
        extracted_proc_data)
    cw_doc['cw_proc'] = 'interproc'
    gui_data = {'master': frame.master}
    frame.destroy()
    return state_data, gui_data

def are_still_pending_docs(state_data):
    return not(
        bool(
            state_data['pending_docs']))

def load_next_doc(state_data):
    cw_doc = state_data['pending_docs'].pop()
    cw_doc['pending_procs'] = state_data['pipeline'].copy()
    cw_doc['pending_procs'].reverse()
    cw_doc['finished_procs'] = []
    cw_doc['cw_proc'] = 'interproc'
    state_data['cw_doc'] = cw_doc

def load_next_proc(state_data):
    cw_doc = state_data['cw_doc']
    proc_key = cw_doc['pending_procs'].pop()
    cw_doc['cw_proc'] = proc_key

def construct_frame(state_data, gui_data):
    frame_constructor =  constructor_from_key(
        state_data['cw_doc']['cw_proc'])
    return frame_constructor(
        state_data, gui_data)

def deploy_frame(frame, gui_data):
    frame.grid(
        row=0, column=0, rowspan=FRAME_ROWSPAN)
    root.mainloop()

def deploy_next_pending_proc(state_data, gui_data):
    load_next_proc(state_data)
    frame = construct_frame(state_data, gui_data)
    deploy_frame(frame, gui_data)

def deploy_pipeline(state_data, gui_data):
    deploy_next_pending_proc(
        state_data, gui_data)
    
def deploy_next_pending_doc(state_data, gui_data):
    load_next_doc(state_data)
    deploy_pipeline(
        state_data, gui_data)

def write_out(state_data):
    pass
    
def control_shift(frame):
    state_data, gui_data = destruct_frame(
        frame)
    if are_still_pending_procs(state_data):
        deploy_next_pending_proc(
            state_data, gui_data)
    if are_still_pending_docs(state_data):
        deploy_next_pending_doc(
            state_data, gui_data)
    else:
        pass
        # write_out(state_data)

if __name__ == '__main__':
    print('ocr.py\n')

    project_folder = os.getcwd()

    subfolders = ['2022_06_26', '2022_06_27']
    out_subfolders = list(map(
        lambda cs: "__".join([cs, "out"]),
        subfolders))

    subfolder_path = "/".join([
        'test_data',
        'levike_facturi_test',
        out_subfolders[0]])
    
    input_folder_data = folder_data__(
        project_folder, subfolder_path)

    doc_data = doc_data__(input_folder_data)

    # doc_keys = list(doc_data.keys())


    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())

    pipeline = ['ocr'] # ['document_describer']
    proc_params = {}
    
    state_data = init_state_data__from_doc_data(
        doc_data, pipeline, ocr_proc_params)

    gui_data = {'master': root}

    deploy_next_pending_doc(
        state_data, gui_data)

    print('we have controll back')


    # if not(os.path.isdir('test_out_images')):
    #     os.mkdir('test_out_images')
    # for key in doc_data.keys():
    #     im = Image.fromarray(
    #         doc_data[key]['im'])
    #     im.save(f'test_out_images/{key}__.png', 'png')
        
    # im = Image.fromarray(
    #     doc_data[doc_keys[0]]['im'])



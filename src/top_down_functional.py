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
from im import display_cv, crop

from document_preprocessing import folder_data__
from human import deploy_human_guided_im_transform_pipeline, deploy_human_guided_im_filter_pipeline, deploy_human_guided_doc_roi_selection, deploy_human_guided_doc_elem_extraction
from load_documents import load_raw_document_data, load_preproc_doc_data

elements_of_interest = {
    'header': [
        'furnizor', 'nr_doc', 'dată_emitere'],
    'body': [
        'denumire_produs/serviciu', 'cantitate',
        'val_fără_tva', 'preţ_vânzare'],
    'footer': ['total'],
    # 'chitanță': [],
}

def splice_in(doc_data, xss, keys, subkeys):
    for key, xs in zip(keys, xss):
        for subkey, x in zip(subkeys, xs):
            doc_data[key][subkey] = x
    return doc_data

def write_preproc_document_data_out(folder_data, doc_data):
    write_document_data_out(
        folder_data, doc_data,
        'preproc_im', 'preprocessed')

def write_document_data_out(
        folder_data, doc_data,
        out_im_key, out_folder_name):
    wf = folder_data['working_folder']
    output_folder = "/".join([
        wf, out_folder_name])
    if not(os.path.isdir(output_folder)):
        os.makedirs(output_folder)
    serializable_output_data = {}
    for doc_key in doc_data.keys():
        cv.imwrite(
            "/".join([
                output_folder,
                ".".join([
                    "__".join([
                        doc_key, 'preprocessed']),
                    'png'])]),
            doc_data[doc_key][out_im_key])
        out_doc_dat = {}
        for field_key in doc_data[doc_key].keys():
            if 'im' not in field_key:
                out_doc_dat[field_key] = doc_data[
                    doc_key][field_key]
        serializable_output_data[doc_key] = out_doc_dat
    metadata_fpath = "/".join([
        output_folder, "preprocessing_metadata.json"])
    with open(metadata_fpath, 'w') as f:
        f.write(
            json.dumps(
                serializable_output_data))

# # :: (TK, string) -> [im]
# def preprocess_documents(doc_data):
#     cvims = load_cv_images(
#         input_folder)
#     transformed_cvims = list(
#         map(deploy_human_guided_im_transform_pipeline,
#             cvims))
#     filtered_cvims = list(
#         map(deploy_human_guided_im_filter_pipeline,
#             transformed_cvims))
#     return filtered_cvims

# :: SPARSE COMPOUND -> SPARSE COMPOUND
def preprocess_documents(doc_data):
    def transform__(key):
        return deploy_human_guided_im_transform_pipeline(
            doc_data[key]['raw_im'])
    def filter__(key):
        return deploy_human_guided_im_filter_pipeline(
            doc_data[key]['transformed_im'])

    keys = list(doc_data.keys())
    transformed_doc_data = splice_in(
        doc_data,
        list(
            map(transform__,
                keys)),
        keys,
        ['transformed_im', 'transform_pipeline_data'])

    filtered_doc_data = splice_in(
        transformed_doc_data,
        list(
            map(filter__,
                keys)),
        keys,
        ['preproc_im', 'filter_pipeline_data'])

    return filtered_doc_data

def data_entry__preprocessing_phase(
        folder_data, write_out=False):
    doc_data = load_raw_document_data(
        folder_data['working_folder'],
        fst_n=2) # !!! DEBUG
    preproc_doc_data = preprocess_documents(doc_data) # HITL
    if write_out:
        write_preproc_document_data_out(
            folder_data, preproc_doc_data)
    return preproc_doc_data

# acting on a single document
def corners__from_roi_data(roi_select_data):
    in_data = roi_select_data
    out_data = {}
    for roi_key in in_data.keys():
        p1 = (in_data[roi_key]['p0']['im']['x'],
              in_data[roi_key]['p0']['im']['y'])
        p2 = (in_data[roi_key]['p1']['im']['x'],
              in_data[roi_key]['p1']['im']['y'])
        out_data[roi_key] = (p1, p2)
    return out_data

def crop_by_rois_data(im, rois_data):
    out_data = {}
    for roi_key in rois_data.keys():
        out_data[roi_key] = crop(
            im, *rois_data[roi_key])
    return out_data

# def documents_segments(doc_data):
#     def select_rois__(doc_key):
#         return deploy_human_guided_doc_roi_selection(
#             doc_key, doc_data, list(doc_sections.keys()))

#     def crop__(im_x_cornerss_data):
#         im, cornerss_data = im_x_cornerss_data
#         out_data = {}
#         for key in cornerss_data:
#             a, b = cornerss_data[key]
#             out_data[key] = crop(im, a, b)
#         return out_data

#     doc_keys = list(doc_data.keys())

#     roi_select_data = list(
#         map(select_rois__,
#             doc_keys))
#     doc_ims = map(
#         lambda key: doc_data[key]['preproc_im'],
#         doc_keys)
#     roi_cornerss = map(
#         corners__from_roi_data,
#         roi_select_data)
#     roi_imss = list(
#         map(crop__, zip(doc_ims, roi_cornerss)))
    
#     return dict(zip(doc_keys, roi_imss))

# {{{Number}}} -> ((Number))
def points_as_tuples(data):
    return (
        (data['p0']['im']['x'], data['p0']['im']['y']),
        (data['p1']['im']['x'], data['p1']['im']['y']),
    )

def select_doc_section_rois(doc_dat, section_keys):
    return deploy_human_guided_doc_roi_selection(
        doc_dat, section_keys)

def augment_wth_doc_sections(doc_dat, elemss_of_interest):
    doc_section_keys = list(elemss_of_interest.keys())
    rois_data = select_doc_section_rois(
        doc_dat, doc_section_keys)
    doc_dat['sections'] = {}
    for roi_key in rois_data.keys():
        doc_dat['sections'][roi_key] = {}
        section_im = crop(
            doc_dat['preproc_im'],
            *points_as_tuples(
                rois_data[roi_key]))
        doc_dat['sections'][roi_key]['data'] = rois_data[roi_key]
        doc_dat['sections'][roi_key]['im'] = section_im

def ocr_extract_elements(im, elems_to_extract):
    return deploy_human_guided_doc_elem_extraction(
        im, elems_to_extract)

def augment_wth_extracted_data(doc_dat, elemss_of_interest):
    for section_key in doc_dat['sections'].keys():
        elems_to_extract = elemss_of_interest[section_key]
        extracted_elem_data = ocr_extract_elements(
            doc_dat['sections'][section_key]['im'],
            elems_to_extract)
        doc_dat['sections'][section_key][
            'extracted'] = extracted_elem_data

def extract_data(doc_data, elemss_of_interest):
    for key in doc_data.keys():
        augment_wth_doc_sections(
            doc_data[key], elemss_of_interest)
        augment_wth_extracted_data(
            doc_data[key], elemss_of_interest)
        
    return doc_data
    

def data_entry__extraction_phase(
        folder_data, elemss_of_interest, preprocess=False):
    if preprocess:
        preproc_doc_data = data_entry__preprocessing_phase(
            folder_data)
    else:
        preproc_doc_data = load_preproc_doc_data(
            folder_data['working_folder'])
    extracted_data = extract_data(
        preproc_doc_data,
        elemss_of_interest)
    # export_to_pag_interface(
    #     extracted_data)
    return extracted_data
    
if __name__ == '__main__':
    print('top_down_functional.py\n')

    project_folder = os.getcwd()

    subfolders = ['2022_06_26', '2022_06_27']

    subfolder_path = "/".join([
        'test_data',
        'levike_facturi_test',
        subfolders[0]])

    working_folder = "/".join([
        project_folder, subfolder_path])

    folder_data = {
        'project_folder': project_folder,
        'subfolders': subfolders,
        'working_folder': working_folder,
        'subfolder_path': subfolder_path,
    }
    
    # xtracted_data = data_entry__preprocessing_phase(
    #     folder_data, write_out=True)

    xtracted = data_entry__extraction_phase(
        folder_data, elements_of_interest)

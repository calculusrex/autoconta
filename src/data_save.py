import os
import numpy as np
import json
import cv2 as cv
import functools as ft
import pickle
from pprint import pprint

from im import read_im


# def dict_from_np_array(arr):
#     return {
#         'type': 'numpy_array',
#         'dtype': arr.dtype.name,
#         'nested_list': arr.tolist(),
#     }

# def np_array_from_dict(data):
#     return np.array(
#         data['nested_list'],
#         dtype=np.dtype(data['dtype']))

# def is_leaf(data):
#     return ft.reduce(
#         lambda a, b: a or b,
#         map(lambda typ: type(data) == typ,
#             [str, int, float, type(None), bool]),
#         False)

# def is_array(data):
#     return type(data) == np.ndarray

# def is_list(data):
#     return type(data) == list

# def is_dict(data):
#     return type(data) == dict

# def is_serial_repr(data):
#     return ft.reduce(
#         lambda a, b: a and b,
#         [
#             type(data) == dict,
#             'type' in data.keys(),
#         ],
#         True)

# def is_serialized_array(data):
#     if is_serial_repr(data):
#         if data['type'] == 'numpy_array':
#             return True
#     return False

# def serializable_structure_from(data):
#     if is_leaf(data):
#         return data
#     elif is_array(data):
#         return dict_from_np_array(data)
#     elif is_list(data):
#         return list(
#             map(serializable_structure_from,
#                 data))
#     elif is_dict(data):
#         out = {}
#         for key in data.keys():
#             out[key] = serializable_structure_from(
#                 data[key])
#         return out
#     else:
#         raise Exception(
#             "\n".join([
#                 f'-- CALCULUSREX --',
#                 f'Data type unaccounted for: {type(data)}',
#                 f'{data}'
#             ]))

# def nonserializable_structure_from(data):
#     if is_leaf(data):
#         return data
#     elif is_list(data):
#         return list(
#             map(nonserializable_structure_from,
#                 data))
#     elif is_serialized_array(data):
#         return np_array_from_dict(data)
#     elif is_dict(data):
#         out = {}
#         for key in data.keys():
#             out[key] = nonserializable_structure_from(
#                 data[key])
#         return out
#     else:
#         raise Exception(
#             "\n".join([
#                 f'-- CALCULUSREX --',
#                 f'Data type unaccounted for: {type(data)}',
#                 f'{data}'
#             ]))


# def dumps(data_structure):
#     return json.dumps(
#         serializable_structure_from(
#             data_structure))

# def loads(data_string):
#     return nonserializable_structure_from(
#         json.loads(
#             data_string))

## GOAL : Make a function pair that can save and load a compound data structure to disk.

def collect_arrays__(data, path, to_be_populated):
    if is_leaf(data):
        pass
    elif is_array(data):
        to_be_populated[path[1:]] = data
    elif is_list(data):
        for i in range(len(data)):
            collect_arrays__(
                data[i], f'{path}.{i}', to_be_populated)
    elif is_dict(data):
        for key in data.keys():
            collect_arrays__(
                data[key], f'{path}.{key}', to_be_populated)
    else:
        raise Exception(
            "\n".join([
                f'-- CALCULUSREX --',
                f'Data type unaccounted for: {type(data)}',
                f'{data}'
            ]))

def remove_arrays(data, path):    
    if is_leaf(data):
        return data
    elif is_array(data):
        return path[1:] # without the first point
    elif is_list(data):
        elems = []
        for i in range(len(data)):
            elems.append(
                remove_arrays(
                    data[i], f'{path}.{i}'))
        return elems
    elif is_dict(data):
        out = {}
        for key in data.keys():
            out[key] = remove_arrays(
                data[key], f'{path}.{key}')
        return out
    else:
        raise Exception(
            "\n".join([
                f'-- CALCULUSREX --',
                f'Data type unaccounted for: {type(data)}',
                f'{data}'
            ]))

def segregate_arrays(data):
    arrays = {}
    collect_arrays__(
        data, '', arrays)
    data_w_o_arrays = remove_arrays(data, '')
    return {
        'arrays': arrays,
        'serializable': data_w_o_arrays
    }

def place_arrays_back__(data, arrays):
    if is_leaf(data):
        if data in arrays.keys():
            return arrays[data]
        else:
            return data
    elif is_leaf(data):
        return data
    elif is_list(data):
        elems = []
        for i in range(len(data)):
            elems.append(
                place_arrays_back__(
                    data[i], arrays))
        return elems
    elif is_dict(data):
        out = {}
        for key in data.keys():
            out[key] = place_arrays_back__(
                data[key], arrays)
        return out
    else:
        raise Exception(
            "\n".join([
                f'-- CALCULUSREX --',
                f'Data type unaccounted for: {type(data)}',
                f'{data}'
            ]))

def integrate_data(data):
    arrays = data['arrays']
    serializable = data['serializable']
    unserializable = place_arrays_back__(serializable, arrays)
    return unserializable

def save_structure_to_disk(data, fpath):
    segged = segregate_arrays(data)
    if os.path.isdir(fpath):
        raise Exception(
            "\n".join([
                f'-- CALCULUSREX --',
                f'folderpath: {fpath} is already present on disk',
            ]))       
    else:
        os.makedirs(fpath)
    for key in segged['arrays'].keys():
        fname = f'{fpath}/{key}.png'
        cv.imwrite(
            fname, segged['arrays'][key])
    with open(f'{fpath}/serializable.json', 'w') as f:
        f.write(
            json.dumps(
                segged['serializable']))

def load_structure_from_disk(fpath):
    fnames = os.listdir(fpath)
    json_fnames = list(
        filter(lambda cs: cs.split('.')[-1] == 'json',
               fnames))
    png_fnames = list(
        filter(lambda cs: cs.split('.')[-1] == 'png',
               fnames))
    if not(json_fnames):
        raise Exception(
            "\n".join([
                f'-- CALCULUSREX --',
                f'NO JSON FILES IN FOLDER: {fpath}',
            ]))
    with open(f'{fpath}/serializable.json', 'r') as f:
        serializable_data = json.loads(f.read())
    array_data = {}
    for png_fname in png_fnames:
        key = '.'.join(png_fname.split('.')[:-1])
        array_data[key] = cv.imread(
            f'{fpath}/{png_fname}')
    out = integrate_data({
        'arrays': array_data,
        'serializable': serializable_data
    })
    return out

if __name__ == '__main__':
    print('data_save.py')

    # image_folder = 'test_data/levike_facturi_test/2022_06_26/raw_from_the_scanner'

    # impath = "/".join([
    #     image_folder,
    #     np.random.choice(
    #         os.listdir(image_folder))])

    # im = read_im(impath)

    # pickle.dumps(im)

    'test_data/levike_facturi_test/2022_06_26/preprocessed/'

    json_metadata_fname = 'test_data/levike_facturi_test/2022_06_26/preprocessed/preprocessing_metadata.json'

    im_37_fname = 'test_data/levike_facturi_test/2022_06_26/preprocessed/IMG_0037__preprocessed.png'

    im_50_fname = 'test_data/levike_facturi_test/2022_06_26/preprocessed/IMG_0050__preprocessed.png'

    with open(json_metadata_fname, 'r') as f:
        metadata = json.loads(f.read())
    metadata['IMG_0037']['preproc_im'] = read_im(im_37_fname)
    metadata['IMG_0050']['preproc_im'] = read_im(im_50_fname)

    fpath = f'{os.getcwd()}/fuschnick'

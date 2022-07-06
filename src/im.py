import cv2 as cv
import pytesseract as tesseract
import os
from PIL import Image, ImageTk
import numpy as np

from pdf2image import convert_from_path, convert_from_bytes

def read_im(filepath):
    extension = filepath.split('.')[-1].lower()
    if extension == 'pdf':
        im = np.array(
            convert_from_path(filepath)[0])
    else:
        im = cv.imread(filepath)
    return im

def load_image(impath):
    return read_im(impath)

def rgb2hex(r,g,b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'

def display_cv(im):
    pil_im = Image.fromarray(im)
    pil_im.show()

def df_from_im(im):
    return tesseract.image_to_data(
        im, output_type=tesseract.Output.DATAFRAME)

def dict_from_im(im):
    df = df_from_im(im)
    data = []
    for index, row in df.iterrows():
        dat = {}
        for key in row.keys():
            dat[key] = row[key]
        data.append(dat)            
    return data

def draw_rectangles__cv(im, df):
    im_out = im.copy()
    for index, row in df.iterrows():
        x, y, w, h = row['left'], row['top'], row['width'], row['height']
        im_out = cv.rectangle(
            im_out,
            (x, y),
            (x + w, y + h),
            (0, 0 , 0),
            1
        )
    return im_out

def cv_ocr(im):
    df = df_from_im(im)
    return draw_rectangles__cv(
        im, df)

def im_rescale(im, factor=1):
    h, w, _ = im.shape
    pil = Image.fromarray(im)
    return np.array(
        pil.resize(
            (int(w * factor), int(h * factor)),
            resample=Image.LANCZOS
        )
    )

def rescaled_pil_from_cv(im, factor=1):
    h, w = im.shape[:2]
    pil = Image.fromarray(im)
    return pil.resize(
        (int(w * factor), int(h * factor)),
        resample=Image.LANCZOS
    )

# def tkim_from_cv(global_data, im, scale_factor=1):
#     pil = rescaled_pil_from_cv(im, factor=scale_factor)
#     tkim = ImageTk.PhotoImage(pil)
#     global_data['tkim_images'].append(tkim)
#     return tkim

def tkim_from_cv(im, scale_factor=1):
    pil = rescaled_pil_from_cv(im, factor=scale_factor)
    return ImageTk.PhotoImage(pil)


def df_rescale(df0, factor=1):
    keys = ['left', 'top', 'width', 'height']
    df = df0.copy()
    for key in keys:
        df[key] = df[key].apply(lambda x: int(x * factor))
    return df

def cv_rescale(im, df, factor=1):
    return (
        im_rescale(im, factor),
        df_rescale(df, factor)
    )

def crop(im, a, b):
    # x0, y0 = a
    # x1, y1 = b
    # y_top, y_bot = min(y0, y1), max(y0, y1)
    # x_left, x_right = min(y0, y1), max(y0, y1)

    upper_left_corner, lower_right_corner = a, b

    x_left, y_top = upper_left_corner
    x_right, y_bot = lower_right_corner
    
    slce = im[y_top:y_bot, x_left:x_right]
    return slce.copy()

def cv_ocr(im, factor=1):
    df = df_from_im(im)
    if factor != 1:
        im1, df1 = cv_rescale(
            im, df, factor=factor)
        im1 = draw_rectangles__cv(
            im1, df1)
        return im1, df1
    else:
        im1 = draw_rectangles__cv(
            im, df)
        return im1, df

def imwarp(im, input_points):
    a, b, c, d = input_points
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d

    w1 = int(round(np.mean([
        np.sqrt((dx - ax)**2 + (dy - ay)**2),
        np.sqrt((cx - bx)**2 + (cy - by)**2),
    ])))
    h1 = int(round(np.mean([
        np.sqrt((bx - ax)**2 + (by - ay)**2),
        np.sqrt((dx - cx)**2 + (dy - cy)**2),
    ])))

    a1 = (0, 0)
    b1 = (0, h1)
    c1 = (w1, h1)
    d1 = (w1, 0)
    output_points = a1, b1, c1, d1

    transform_matrix = cv.getPerspectiveTransform(
        np.float32(input_points),
        np.float32(output_points))
    
    return cv.warpPerspective(
        im, transform_matrix, (w1, h1), flags=cv.INTER_LINEAR)




# New image dimensions after rotation
# (such that no part of the image gets clipped out after the rotation)
def rot_new_dims(im, angle):
    h0, w0 = im.shape[:2]
    theta = np.deg2rad(angle)
    wa, wb = np.abs(w0*np.cos(theta)), np.abs(h0*np.sin(theta))
    ha, hb = np.abs(h0*np.cos(theta)), np.abs(w0*np.sin(theta))
    w1, h1 = int(wa + wb), int(ha + hb)
    return w1, h1
    
def rotate_without_clipping(im, angle):
    h0, w0 = im.shape[:2]
    w1, h1 = rot_new_dims(im, angle)
    center = w0/2, h0/2
    scale = 1
    M = cv.getRotationMatrix2D(
        center, angle, scale)
    M[0, 2] += w1/2 - w0/2
    M[1, 2] += h1/2 - h0/2
    return cv.warpAffine(
        im, M, (w1, h1))

def rotate(im, angle):
    h, w = im.shape[:2]
    center = w/2, h/2
    scale = 1
    M = cv.getRotationMatrix2D(
        center, angle, scale)
    return cv.warpAffine(
        im, M, (w, h))

# def rotate_1deg(im, direction):
#     return rotate(im, direction * 1)

# def rotate_5deg(im, direction):
#     return rotate(im, direction * 5)

def orthogonal_rotate(im, factor90deg):
    factor90deg = factor90deg % 4
    cases = {
        0: lambda im: im,
        1: lambda im: cv.rotate(im, cv.ROTATE_90_COUNTERCLOCKWISE),
        2: lambda im: cv.rotate(im, cv.ROTATE_180),
        3: lambda im: cv.rotate(im, cv.ROTATE_90_CLOCKWISE),
    }
    return cases[factor90deg](im)
rotate_by_90deg = orthogonal_rotate

### FILTERING ----------------------------------------------------------

def denoise(im, kernel_size, n_iterations):
    out = im.copy()
    for _ in range(n_iterations):
        out = cv.medianBlur(out, kernel_size)
    return out

def dilate_erode(im, kernel_size, n_iterations):
    if kernel_size > 0:
        f = cv.erode
    else:
        f = cv.dilate
    ksz = abs(kernel_size)
    kernel = np.ones(
        (ksz, ksz), np.uint8)
    return f(im, kernel, iterations=n_iterations)

def threshold(im, block_size, constant):
    if block_size > 1:
        return cv.adaptiveThreshold(
            im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, block_size, constant
        )
    else:
        return im.copy()


if __name__ == '__main__':
    print('im.py', '\n')

    import os
    import cv2 as cv

    # ----------------------------------------------------------
    clean_invoice_fname = 'test_data/609d5d3c4d120e370de52b70_invoice-lp-light-border.png'
    samples_path = 'test_data/samples_2021'
    samples_fnames = list(
        map(lambda fnm: f'{samples_path}/{fnm}',
            os.listdir('test_data/samples_2021')))
    #im = cv.imread(clean_invoice_fname)
    im = cv.imread(samples_fnames[0])
    im_h, im_w, _ = im.shape
    # ----------------------------------------------------------

    bw = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    bw = cv.adaptiveThreshold(
        bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 16+1, 7
    )
    
    display_cv(bw)

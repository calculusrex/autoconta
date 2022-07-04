import tkinter as tk
# import tk
# import pytesseract as tesseract
# import os
# import numpy as np
import cv2 as cv
from pprint import pprint
import functools as ft
import collections

from canvas import canvas_from_im, DocumentCanvas, ValidationCanvas
from im import imwarp, rgb2hex, display_cv, rotate, orthogonal_rotate, rotate_without_clipping, display_cv, denoise, dilate_erode, threshold, crop, im_rescale, dict_from_im #, rotate_by_90deg
from constants import *


corner_data = {
    'UPPER_LEFT': {'outward': {'x_factor': -1,
                               'y_factor': -1},
                   'inward': {'x_factor': 1,
                              'y_factor': 1},
                   'selection_point_index': 0},

    'LOWER_LEFT': {'outward': {'x_factor': -1,
                               'y_factor': 1},
                   'inward': {'x_factor': 1,
                              'y_factor': -1},
                   'selection_point_index': 1},

    'LOWER_RIGHT': {'outward': {'x_factor': 1,
                                'y_factor': 1},
                    'inward': {'x_factor': -1,
                               'y_factor': -1},
                    'selection_point_index': 2},

    'UPPER_RIGHT': {'outward': {'x_factor': 1,
                                'y_factor': -1},
                    'inward': {'x_factor': -1,
                               'y_factor': 1},
                    'selection_point_index': 3}
}

def flatten_once(xss):
    ys = []
    for xs in xss:
        for x in xs:
            ys.append(x)
    return ys        

def augment_data(dat1, dat2):
    for key in dat2.keys():
        dat1[key] = dat2[key]

def quad_val_coords__(p0_coords, p1_coords):
    return (
        p0_coords['canv']['x'],
        p0_coords['canv']['y'],
        p1_coords['canv']['x'],
        p1_coords['canv']['y'])
        
### GENERAL EDITOR --------------------------------------------------------------------

class ImProcEditor(tk.Frame):
    def __init__(self, state_data, gui_data):
        super().__init__(gui_data['master'])
        self.state_data = state_data

        self.doc_data = self.state_data['cw_doc']
        if 'proc' in self.doc_data.keys():
            self.og_im = self.doc_data['proc']['im']
        else:
            self.og_im = self.doc_data['im']
        self.proc_im = self.og_im.copy()

        self.labels = {}
        self.add_label(self.__class__.__name__)
        self.add_label('hover')
        self.add_label('click')

        # self.bind('<space>', self.skip)

        self.main_canvas = None
        self.main_canvas_bindings = [
            ('<Motion>',
             lambda e: self.mouse_label_event_config(e, 'hover')),
            ('<Button-1>',
             lambda e: self.mouse_label_event_config(e, 'click')),
            ('<Escape>',
             self.cancel), ('<Control-c>', self.cancel),
            ('<Control-z>',
             lambda e: self.previous_stage()),
        ]
        self.install_main_canvas()

    def mouse_coords__(self, event):
        canv_x, canv_y = event.x, event.y
        im_x, im_y = list(
            self.inv_scale_by_main_canv_sf(
                [canv_x, canv_y]))
        return {
            'canv': {'x': canv_x,
                     'y': canv_y},
            'im': {'x': im_x,
                   'y': im_y}}

    def inv_scale_by_main_canv_sf(self, ns):
        sf = self.main_canvas.scale_factor
        return map(
            lambda n: int(round(n / sf)),
            ns)

    def scale_by_main_canv_sf(self, ns):
        return map(
            lambda n: n * self.main_canvas.scale_factor,
            ns)
        
    def skip(self, event):
        self.param_data = {}
        self.next_stage()

    def add_label(self, key):
        self.labels[key] = tk.Label(self, text=key)
        self.labels[key].grid(
            row=len(self.labels) - 1, column=2)

    def mouse_label_event_config(self, event, label_key):
        self.labels[label_key].config(
            text=f'{label_key}\nx: {event.x}\ny: {event.y}')

    def load_bindings(self, widget, bindings):
        for keypress, function in bindings:
            widget.bind(keypress, function)

    def augment_main_canvas_bindings(self, additional_bindings):
        self.load_bindings(
            self.main_canvas, additional_bindings)
        self.main_canvas_bindings.extend(
            additional_bindings)

    def install_main_canvas(self):
        self.main_canvas = DocumentCanvas(
            self, self.proc_im)
        self.main_canvas.grid(row=0, column=0, rowspan=FRAME_ROWSPAN)
        self.load_bindings(
            self.main_canvas, self.main_canvas_bindings)
        self.main_canvas.focus_set()

    def update_main_canvas(self, grid_offset=None):
        self.main_canvas.destroy()
        self.install_main_canvas()
        if grid_offset:
            self.main_canvas.install_grid(
                offset=grid_offset)

    def cancel(self, event):
        parent = self.master
        frame_constructor = self.__class__
        frame = frame_constructor(parent, self.og_im.copy(), self.pipeline_data)
        self.destroy()
        frame.grid(
            row=0, column=0, rowspan=FRAME_ROWSPAN)

    def next_stage(self):
        self.state_data['control_shift'](self)
        # if self.data_directed:
        #     parent = self.master
        #     pipeline_data = self.pipeline_data
        #     pipeline_data['destruction_procedure'](
        #         self, pipeline_data)
        #     pipeline_data['construction_procedure'](
        #         parent, pipeline_data)
        # else:
        #     self.pipeline_data['improc_params'][self.__class__.__name__] = self.param_data
        #     progress_data = {
        #         'og_im': self.og_im.copy(), 'proc_im': self.proc_im.copy()
        #     }
        #     # SAVE --------------------------------------------------------------------
        #     cv.imwrite(f'improc_images/{self.__class__.__name__}.png', self.proc_im)
        #     # -------------------------------------------------------------------------
        #     self.pipeline_data['improc_progress'][self.__class__.__name__] = progress_data
        #     # pprint(self.pipeline_data)
        #     # print()
        #     parent = self.master
        #     next_frame_constructor = self.pipeline_data['relative_constructors'][
        #         self.__class__.__name__]['following']
        #     frame = next_frame_constructor(
        #         parent, self.proc_im.copy(), self.pipeline_data)
        #     self.destroy()
        #     frame.grid(
        #         row=0, column=0, rowspan=FRAME_ROWSPAN)

    def previous_stage(self):
        parent = self.master
        previous_frame_constructor = self.pipeline_data['relative_constructors'][
            self.__class__.__name__]['previous']
        frame = previous_frame_constructor(
            parent,
            self.pipeline_data[
                'improc_progress'][previous_frame_constructor.__name__]['og_im'],
            self.pipeline_data)
        self.destroy()
        frame.grid(
            row=0, column=0, rowspan=FRAME_ROWSPAN)
        

# class ImProcEditor(tk.Frame):
#     def __init__(self, master, image_data, pipeline_data, data_directed=False):
#         super().__init__(master)
#         self.data_directed = data_directed
#         self.pipeline_data = pipeline_data
#         self.grid(row=0, column=0, rowspan=FRAME_ROWSPAN)
#         self.og_im = image_data['im']
#         self.og_im_fname = image_data['fname']
        
#         self.proc_im = image_data['im'].copy()
        
#         self.labels = {}
#         self.add_label(self.__class__.__name__)
#         self.add_label('hover')
#         self.add_label('click')

#         self.bind('<space>', self.skip)

#         self.main_canvas = None
#         self.main_canvas_bindings = [
#             ('<Motion>', lambda e: self.mouse_label_event_config(e, 'hover')),
#             ('<Button-1>', lambda e: self.mouse_label_event_config(e, 'click')),
#             ('<Escape>', self.cancel), ('<Control-c>', self.cancel),
#             ('<Control-z>', lambda e: self.previous_stage()),
#         ]
#         self.install_main_canvas()

#     def skip(self, event):
#         self.param_data = {}
#         self.next_stage()

#     def add_label(self, key):
#         self.labels[key] = tk.Label(self, text=key)
#         self.labels[key].grid(
#             row=len(self.labels) - 1, column=2)

#     def mouse_label_event_config(self, event, label_key):
#         self.labels[label_key].config(
#             text=f'{label_key}\nx: {event.x}\ny: {event.y}')

#     def load_bindings(self, widget, bindings):
#         for keypress, function in bindings:
#             widget.bind(keypress, function)

#     def augment_main_canvas_bindings(self, additional_bindings):
#         self.load_bindings(
#             self.main_canvas, additional_bindings)
#         self.main_canvas_bindings.extend(
#             additional_bindings)

#     def install_main_canvas(self):
#         self.main_canvas = DocumentCanvas(
#             self, self.proc_im)
#         self.main_canvas.grid(row=0, column=0, rowspan=FRAME_ROWSPAN)
#         self.load_bindings(
#             self.main_canvas, self.main_canvas_bindings)
#         self.main_canvas.focus_set()

#     def update_main_canvas(self, grid_offset=None):
#         self.main_canvas.destroy()
#         self.install_main_canvas()
#         if grid_offset:
#             self.main_canvas.install_grid(
#                 offset=grid_offset)

#     def cancel(self, event):
#         parent = self.master
#         frame_constructor = self.__class__
#         frame = frame_constructor(parent, self.og_im.copy(), self.pipeline_data)
#         self.destroy()
#         frame.grid(
#             row=0, column=0, rowspan=FRAME_ROWSPAN)

#     def next_stage(self):
#         if self.data_directed:
#             parent = self.master
#             pipeline_data = self.pipeline_data
#             pipeline_data['destruction_procedure'](
#                 self, pipeline_data)
#             pipeline_data['construction_procedure'](
#                 parent, pipeline_data)
#         else:
#             self.pipeline_data['improc_params'][self.__class__.__name__] = self.param_data
#             progress_data = {
#                 'og_im': self.og_im.copy(), 'proc_im': self.proc_im.copy()
#             }
#             # SAVE --------------------------------------------------------------------
#             cv.imwrite(f'improc_images/{self.__class__.__name__}.png', self.proc_im)
#             # -------------------------------------------------------------------------
#             self.pipeline_data['improc_progress'][self.__class__.__name__] = progress_data
#             # pprint(self.pipeline_data)
#             # print()
#             parent = self.master
#             next_frame_constructor = self.pipeline_data['relative_constructors'][
#                 self.__class__.__name__]['following']
#             frame = next_frame_constructor(
#                 parent, self.proc_im.copy(), self.pipeline_data)
#             self.destroy()
#             frame.grid(
#                 row=0, column=0, rowspan=FRAME_ROWSPAN)

#     def previous_stage(self):
#         parent = self.master
#         previous_frame_constructor = self.pipeline_data['relative_constructors'][
#             self.__class__.__name__]['previous']
#         frame = previous_frame_constructor(
#             parent,
#             self.pipeline_data[
#                 'improc_progress'][previous_frame_constructor.__name__]['og_im'],
#             self.pipeline_data)
#         self.destroy()
#         frame.grid(
#             row=0, column=0, rowspan=FRAME_ROWSPAN)
        

### WARPING ----------------------------------------------------------------------------      

class WarpingEditor(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)

        self.proc_key = 'warping'

        self.augment_main_canvas_bindings([
            ('<Motion>', self.hover),
            ('<Button-1>', self.click),
            ('<space>', self.auto_place_points),
        ])
        self.main_canvas.install_crosshairs()
        self.main_canvas.focus_set()

        self.selection_points = []
        self.laid_selection_lines = []
        self.floating_selection_line = self.main_canvas.create_line(
            0, 0, 0, 0, fill=MAGENTA, width=LINE_WIDTH)
        self.closing_selection_line = self.main_canvas.create_line(
            0, 0, 0, 0, fill=MAGENTA, width=LINE_WIDTH)

    def hover(self, event):
        self.mouse_label_event_config(event, 'hover')
        self.main_canvas.update_crosshairs(event)
        canv_x, canv_y = event.x, event.y
        selpnts = self.selection_points
        if len(selpnts) > 0 and len(selpnts) < 4:
            prev_x, prev_y = self.selection_points[-1]
            prev_x *= self.main_canvas.scale_factor
            prev_y *= self.main_canvas.scale_factor
            self.main_canvas.coords(
                self.floating_selection_line,
                prev_x, prev_y, canv_x, canv_y)
        if len(self.selection_points) == 3:
            init_x, init_y = self.selection_points[0]
            init_x *= self.main_canvas.scale_factor
            init_y *= self.main_canvas.scale_factor
            self.main_canvas.coords(
                self.closing_selection_line,
                init_x, init_y, canv_x, canv_y)

    def click(self, event):
        self.mouse_label_event_config(event, 'click')
        im_x = int(round(event.x / self.main_canvas.scale_factor))
        im_y = int(round(event.y / self.main_canvas.scale_factor))
        if len(self.selection_points) < 4:
            self.selection_points.append(
                (im_x, im_y))

        if len(selpnts) > 1 and len(selpnts) < 4:
            x0, y0 = self.selection_points[-2]
            self.laid_selection_lines.append(
                self.main_canvas.create_line(
                    *map(lambda n:n*self.main_canvas.scale_factor,
                         [x0, y0, im_x, im_y]),
                    fill=MAGENTA, width=2))

        if len(self.selection_points) == 4:
            self.bind('<space>', self.apply_warp)
            for line_index in self.laid_selection_lines:
                self.main_canvas.delete(line_index)            
            self.main_canvas.delete(self.floating_selection_line)
            self.main_canvas.delete(self.closing_selection_line)
            self.selection_polygon = self.main_canvas.create_polygon(
                *map(lambda n: n * self.main_canvas.scale_factor,
                     flatten_once(self.selection_points)),
                outline=MAGENTA, width=LINE_WIDTH, fill='')
            self.proc_im = imwarp(
                self.og_im, self.selection_points)
            self.install_tuning_canvas()

    def auto_place_points(self, event):
        self.selection_points = [
            (0, 0),
            (0, self.main_canvas.im_h),
            (self.main_canvas.im_w, self.main_canvas.im_h),
            (self.main_canvas.im_w, 0),
        ]
        self.selection_polygon = self.main_canvas.create_polygon(
            *map(lambda n: n * self.main_canvas.scale_factor,
                 flatten_once(self.selection_points)),
            outline=MAGENTA, width=LINE_WIDTH, fill='')
        self.proc_im = imwarp(
            self.og_im, self.selection_points)
        self.install_tuning_canvas()        
            
    def install_tuning_canvas(self):
        self.tuning_canvas = DocumentCanvas(
            self, self.proc_im)
        self.tuning_canvas.grid(
            row=0, column=1, rowspan=FRAME_ROWSPAN)
        self.tuning_canvas_bindings = [
            ('<Motion>', self.tuning_hover),
            ('<Button-4>',
             lambda e: self.tune(e,16,radial_direction='outward')),
            ('<Button-5>',
             lambda e: self.tune(e,16,radial_direction='inward')),
            ('<Shift-Button-4>',
             lambda e: self.tune(e,4,radial_direction='outward')),
            ('<Shift-Button-5>',
             lambda e: self.tune(e,4,radial_direction='inward')),
        ]
        self.additional_main_frame_bindings = [
            ('w',
             lambda e: self.tune(e,16,cartesian_direction=(0,-1))),
            ('a',
             lambda e: self.tune(e,16,cartesian_direction=(-1,0))),
            ('s',
             lambda e: self.tune(e,16,cartesian_direction=(0,1))),
            ('d',
             lambda e: self.tune(e,16,cartesian_direction=(1,0))),
            ('W',
             lambda e: self.tune(e,4,cartesian_direction=(0,-1))),
            ('A',
             lambda e: self.tune(e,4,cartesian_direction=(-1,0))),
            ('S',
             lambda e: self.tune(e,4,cartesian_direction=(0,1))),
            ('D',
             lambda e: self.tune(e,4,cartesian_direction=(1,0))),
            ('<space>', self.apply_warp),
        ]

        self.load_bindings(
            self.tuning_canvas, self.tuning_canvas_bindings)
        self.load_bindings(
            self, self.additional_main_frame_bindings)
        self.tuning_corner_highlight = self.tuning_canvas.create_rectangle(
            0, 0, 0, 0, outline=MAGENTA, width=LINE_WIDTH)

        self.focus_set()
        self.tuning_canvas.install_crosshairs()

    def update_tuning_canvas(self):
        self.tuning_canvas.destroy()
        self.install_tuning_canvas()
    
    def tuning_hover(self, event):
        self.mouse_label_event_config(event, 'hover')
        self.tuning_canvas.update_crosshairs(event)
        canv_x, canv_y = event.x, event.y
        canv_w, canv_h = self.tuning_canvas.canv_w, self.tuning_canvas.canv_h
        l = 32
        if canv_x < canv_w / 2:
            if canv_y < canv_h / 2:
                highlighted_tuning_corner_key = 'UPPER_LEFT'
                highlight_rect_coos = (
                    0, 0, l, l)
            else:
                highlighted_tuning_corner_key = 'LOWER_LEFT'
                highlight_rect_coos = (
                    0, canv_h - l, l, canv_h)
        else:
            if canv_y < self.tuning_canvas.canv_h / 2:
                highlighted_tuning_corner_key = 'UPPER_RIGHT'
                highlight_rect_coos = (
                    canv_w - l, 0, canv_w, l)
            else:
                highlighted_tuning_corner_key = 'LOWER_RIGHT'
                highlight_rect_coos = (
                    canv_w - l, canv_h - l, canv_w, canv_h)
        self.highlighted_tuning_corner_key = highlighted_tuning_corner_key
        # self.highlight_rect_coos = highlight_rect_coos
        self.tuning_canvas.coords(
            self.tuning_corner_highlight, *highlight_rect_coos)

    def tune(self, event, offset, cartesian_direction=None, radial_direction=None):
        dat = corner_data[self.highlighted_tuning_corner_key]
        x, y = self.selection_points[dat['selection_point_index']]
        if cartesian_direction:
            x_factor, y_factor = cartesian_direction
            x += x_factor * offset
            y += y_factor * offset            
        if radial_direction:
            x += dat[radial_direction]['x_factor'] * offset
            y += dat[radial_direction]['y_factor'] * offset            
        self.selection_points[dat['selection_point_index']] = (x, y)
        self.main_canvas.coords(
            self.selection_polygon,
            *map(lambda n: n * self.main_canvas.scale_factor,
                 flatten_once(self.selection_points)))
        self.proc_im = imwarp(
            self.og_im, self.selection_points)
        self.update_tuning_canvas()

    def apply_warp(self, event): # !!! TO BE COMPLETED
        self.proc_im = imwarp(
            self.og_im, self.selection_points)
        self.param_data = {'selection_points': self.selection_points.copy()}
        self.next_stage()

        
### ORTHOGONAL ROTATION --------------------------------------------------------------

class OrthogonalRotationEditor(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)

        self.proc_key = 'orthogonal_rotation'

        self.add_label('angle')
        self.orthogonal_rotation_ticks = 0
        self.augment_main_canvas_bindings([
            ('<space>', self.apply_orthogonal_rotation),
            ('<Button-4>', lambda _: self.rotate_orthogonal(1)), # trig direc. (counterclock.)
            ('<Button-5>', lambda _: self.rotate_orthogonal(-1)), # trig direction (clockwise)
        ])
        
    def rotate_orthogonal(self, tick):
        self.orthogonal_rotation_ticks = (self.orthogonal_rotation_ticks + tick) % 4
        self.labels['angle'].config(
            text=f'angle\n{self.orthogonal_rotation_ticks * 90}')
        self.proc_im = orthogonal_rotate(
            self.og_im, self.orthogonal_rotation_ticks)
        self.update_main_canvas()

    def apply_orthogonal_rotation(self, event):
        self.param_data = {
            'orthogonal_rotation_ticks': self.orthogonal_rotation_ticks}
        self.next_stage()

### FINE ROTATION --------------------------------------------------------------

class FineRotationEditor(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)

        self.proc_key = 'fine_rotation'

        self.add_label('angle')
        self.fine_rotation_angle = 0 # degrees
        self.augment_main_canvas_bindings([
            ('<space>', self.apply_fine_rotation),
            ('<Shift-Button-4>', lambda _: self.rotate_fine(0.25)), # trig dir. (counterclck)
            ('<Shift-Button-5>', lambda _: self.rotate_fine(-0.25)), # trig dir. (clockwise)
            ('<Button-4>', lambda _: self.rotate_fine(5)), # trig direc. (counterclock.)
            ('<Button-5>', lambda _: self.rotate_fine(-5)), # trig direction (clockwise)
        ])
        self.main_canvas.install_grid(offset=32)
        
    def rotate_fine(self, angle): # angle is in degrees
        self.fine_rotation_angle = (self.fine_rotation_angle + angle) % 360
        self.labels['angle'].config(
            text=f'angle\n{self.fine_rotation_angle}')
        self.proc_im = rotate(
            self.og_im, self.fine_rotation_angle)
        self.update_main_canvas(grid_offset=32)

    def apply_fine_rotation(self, event):
        self.proc_im = rotate_without_clipping(
            self.og_im, self.fine_rotation_angle)
        self.param_data = {'rotation_angle': self.fine_rotation_angle}
        self.next_stage()

### RESCALE -----------------------------------------------------------------------

class RescaleEditor(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)

        self.proc_key = 'rescale'        

        self.add_label('og_letter_height')
        self.roi_points = []
        self.letter_height_ys = []

        self.augment_main_canvas_bindings([
            ('<Button-1>', self.click_roi),
            ('<Motion>', self.hover_roi)])
        self.main_canvas.install_crosshairs()

    def click_roi(self, event):
        self.mouse_label_event_config(event, 'click')
        if len(self.roi_points) <= 1:
            canv_x, canv_y = event.x, event.y
            self.roi_points.append((
                int(round(canv_x / self.main_canvas.scale_factor)),
                int(round(canv_y / self.main_canvas.scale_factor)),
            ))

        if len(self.roi_points) == 1:
            self.roi_rectangle = self.main_canvas.create_rectangle(
                canv_x, canv_y, canv_x, canv_y, outline=MAGENTA, width=LINE_WIDTH)

        if len(self.roi_points) == 2:
            self.roi_im = crop(
                self.og_im, *self.roi_points)
            self.roi_canvas = DocumentCanvas(self, self.roi_im)
            self.roi_canvas.grid(row=0, column=1, rowspan=25)
            self.roi_canvas_bindings = [
                ('<Button-1>', self.click_letter_height),
                ('<Motion>', self.hover_letter_height),
            ]
            self.load_bindings(
                self.roi_canvas, self.roi_canvas_bindings)
            self.first_letter_height_line = self.roi_canvas.create_line(
                0, 0, 0, 0, fill=MAGENTA, width=LINE_WIDTH)
    
    def hover_roi(self, event):
        self.mouse_label_event_config(event, 'hover')
        self.main_canvas.update_crosshairs(event)
            
        if len(self.roi_points) == 1:
            fixed_x, fixed_y = self.roi_points[0]
            fixed_x *= self.main_canvas.scale_factor
            fixed_y *= self.main_canvas.scale_factor
            curr_x, curr_y = event.x, event.y
            self.main_canvas.coords(
                self.roi_rectangle,
                fixed_x, fixed_y, curr_x, curr_y)

    def click_letter_height(self, event):
        self.mouse_label_event_config(event, 'click')
        if len(self.letter_height_ys) <= 1:
            canv_x, canv_y = event.x, event.y
            self.letter_height_ys.append(
                int(round(canv_y / self.roi_canvas.scale_factor)),
            )

        if len(self.letter_height_ys) == 1:
            self.roi_canvas.coords(
                self.first_letter_height_line,
                0, canv_y, self.roi_canvas.winfo_width() - 1, canv_y)
            self.second_letter_height_line = self.roi_canvas.create_line(
                0, 0, 0, 0,
                fill=MAGENTA, width=LINE_WIDTH)

        if len(self.letter_height_ys) == 2:
            ay, by = self.letter_height_ys
            self.og_letter_height = abs(ay - by)
            self.letter_resize_scale_factor = TARGET_LETTER_HEIGHT / self.og_letter_height
            self.proc_im = im_rescale(
                self.og_im, self.letter_resize_scale_factor)
            self.param_data = {
                'letter_resize_scale_factor': self.letter_resize_scale_factor}
            self.next_stage()
                
    def hover_letter_height(self, event):
        self.mouse_label_event_config(event, 'hover')
        curr_x, curr_y = event.x, event.y

        if len(self.letter_height_ys) == 0:
            self.roi_canvas.coords(
                self.first_letter_height_line,
                0, curr_y, self.roi_canvas.winfo_width() - 1, curr_y)

        if len(self.letter_height_ys) == 1:
            ay = self.letter_height_ys[0]
            self.roi_canvas.coords(
                self.second_letter_height_line,
                0, curr_y, self.roi_canvas.winfo_width() - 1, curr_y)

            by = int(round(curr_y / self.roi_canvas.scale_factor))
            og_letter_height = abs(by - ay)
            self.labels['og_letter_height'].config(
                text=f'og_letter_height\n{og_letter_height}')            

### CROP -----------------------------------------------------------------------------

class CropEditor(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)

        self.proc_key = 'crop'        

        self.selection_points = []
        self.selection_rectangle = self.main_canvas.create_rectangle(
            0, 0, 0, 0, outline=MAGENTA, width=LINE_WIDTH)
        self.augment_main_canvas_bindings([
            ('<space>', self.skip),
            ('<Motion>', self.hover_selection_point),
            ('<Button-1>', self.set_selection_point),
        ])
        self.main_canvas.install_crosshairs()

    def set_selection_point(self, event):
        self.mouse_label_event_config(event, 'click')
        canv_x, canv_y = event.x, event.y
        curr_x = int(round(canv_x / self.main_canvas.scale_factor))
        curr_y = int(round(canv_y / self.main_canvas.scale_factor))
        self.selection_points.append((curr_x, curr_y))

        if len(self.selection_points) == 2:
            self.proc_im = crop(
                self.og_im, *self.selection_points)
            self.param_data = {
                'crop_selection_points': self.selection_points}
            self.next_stage()

    def hover_selection_point(self, event):
        self.mouse_label_event_config(event, 'hover')
        self.main_canvas.update_crosshairs(event)
        curr_x, curr_y = event.x, event.y

        if len(self.selection_points) == 1:
            prev_x, prev_y = self.selection_points[0]
            prev_x *= self.main_canvas.scale_factor
            prev_y *= self.main_canvas.scale_factor
            self.main_canvas.coords(
                self.selection_rectangle,
                prev_x, prev_y, curr_x, curr_y)

### FILTERING EDITOR ---------------------------------------------------------------

class FilteringEditor(ImProcEditor):
    def __init__(self, state_data, gui_data, filter_function,
                 allow_negative_kernel_size=False):
        super().__init__(state_data, gui_data)
        self.fltr = filter_function
        if allow_negative_kernel_size:
            self.min_kernel_size_assertion = lambda inst: True
        else:
            self.min_kernel_size_assertion = lambda inst: inst.kernel_size >= 3

        self.add_label('iteration_n')
        self.add_label('kernel_size')
        self.iteration_n = 1
        self.kernel_size = 1
        self.augment_main_canvas_bindings([
            ('<space>', self.apply_filter),
            ('<Button-4>', self.increase_kernel_size),
            ('<Button-5>', self.decrease_kernel_size),
            ('<Control-Button-4>', self.increase_iteration_n),
            ('<Control-Button-5>', self.decrease_iteration_n),
        ])

    def explore_parameters(self, event, kernel_size_offset=None, iteration_n_offset=None):
        if iteration_n_offset:
            self.iteration_n += iteration_n_offset
            self.labels['iteration_n'].config(
                text=f'iteration_n\n{self.iteration_n}')
        if kernel_size_offset:
            self.kernel_size += kernel_size_offset
            self.labels['kernel_size'].config(
                text=f'kernel_size\n{self.kernel_size}')
        self.proc_im = self.fltr(
            self.og_im, self.kernel_size, self.iteration_n)
        self.update_main_canvas()

    def increase_kernel_size(self, event):
        return self.explore_parameters(
            event, kernel_size_offset=2)

    def decrease_kernel_size(self, event):
        if self.min_kernel_size_assertion(self):
            return self.explore_parameters(
                event, kernel_size_offset=-2)

    def increase_iteration_n(self, event):
        return self.explore_parameters(
            event, iteration_n_offset=1)

    def decrease_iteration_n(self, event):
        if self.iteration_n >= 1:
            return self.explore_parameters(
                event, iteration_n_offset=-1)

    def apply_filter(self, event):
        self.param_data = {
            'kernel_size': self.kernel_size,
            'iteration_n': self.iteration_n
        }
        self.next_stage()

### DENOISE -------------------------------------------------------------------------

class DenoiseEditor(FilteringEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(
            state_data, gui_data, denoise)
        self.proc_key = 'denoise'

### DILATE ERODE ----------------------------------------------------------------

class DilateErodeEditor(FilteringEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(
            state_data, gui_data,
            dilate_erode, allow_negative_kernel_size=True)
        self.proc_key = 'dilate_erode'

### THRESHOLD -------------------------------------------------------------------

class ThresholdEditor(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)

        self.proc_key = 'threshold'

        self.block_size = 32 + 1
        self.constant = 5
        self.bw_im = cv.cvtColor(
            self.og_im, cv.COLOR_BGR2GRAY)
        self.proc_im = threshold(
            self.bw_im, self.block_size, self.constant)
        self.add_label('block_size')
        self.add_label('constant')
        self.augment_main_canvas_bindings([
            ('<space>', self.apply_threshold),
            ('<Button-4>', self.increase_block_size),
            ('<Button-5>', self.decrease_block_size),
            ('<Control-Button-4>', self.increase_constant),
            ('<Control-Button-5>', self.decrease_constant),
        ])
        self.update_main_canvas()

    def explore_parameters(self, event, block_size_offset=None, constant_offset=None):
        if constant_offset:
            self.constant += constant_offset
            self.labels['constant'].config(
                text=f'constant\n{self.constant}')
        if block_size_offset:
            self.block_size += block_size_offset
            self.labels['block_size'].config(
                text=f'block_size\n{self.block_size}')
        if self.block_size >= 1:
            self.proc_im = threshold(
                self.bw_im, self.block_size, self.constant)
            self.update_main_canvas()

    def increase_block_size(self, event):
        return self.explore_parameters(
            event, block_size_offset=2)
    def decrease_block_size(self, event):
        if self.block_size >= 3:
            return self.explore_parameters(
                event, block_size_offset=-2)
    def increase_constant(self, event):
        return self.explore_parameters(
            event, constant_offset=1)
    def decrease_constant(self, event):
        if self.constant >= 1:
            return self.explore_parameters(
                event, constant_offset=-1)

    def apply_threshold(self, event):
        self.param_data = {
            'block_size': self.block_size,
            'constant': self.constant
        }
        self.next_stage()

### OPRICAL CHARACTER RECOGNITION (OCR) ----------------------

def diag_crnr_coords__(box_data):
    x0, y0 = box_data['left'], box_data['top']
    w, h = box_data['width'], box_data['height']
    x1, y1 = x0 + w, y0 + h
    return x0, y0, x1, y1

def flatten_sg_lvl_list_tree(xss):
    return ft.reduce(
        lambda a, b: a + b,
        map(lambda key: xss[key],
            xss.keys()),
        [])


class DocumentDescriber(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)
    

class OCRROI(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)
        self.roi_key_data = self.state_data[
            'proc_params']['roi_keys']
        # self.roi_keys = flatten_sg_lvl_list_tree(
        #     self.roi_key_data)
        self.roi_keys = list(
            self.roi_key_data.keys())
        self.pending_roi_keys = self.roi_keys.copy()
        self.pending_roi_keys.reverse()
        self.roi_data = {}
        self.sel_pnts = []
        self.p0_coords = None
        create_rectangle = self.main_canvas.create_rectangle
        self.selection_rectangle = create_rectangle(
            0, 0, 0, 0, outline=MAGENTA, width=LINE_WIDTH)
        self.augment_main_canvas_bindings([
            ('<Motion>', self.hover),
            ('<Button-1>', self.click),
        ])
        self.main_canvas.focus_set()
        self.add_label('cw_roi')
        self.labels['cw_roi'].config(
            text=f'selectează regiune:\n{self.current_roi_key()}')

    def draw_selection_rectangle(self, p1_coords):
        self.main_canvas.coords(
            self.selection_rectangle,
            *quad_val_coords__(
                self.p0_coords, p1_coords))
        
    def hover(self, event):
        self.mouse_label_event_config(event, 'hover')
        hover_coords = self.mouse_coords__(event)
        if bool(self.p0_coords):
            self.draw_selection_rectangle(
                hover_coords)

    def click(self, event):
        self.mouse_label_event_config(event, 'click')
        click_coords = self.mouse_coords__(event)
        if bool(self.pending_roi_keys): # not empty
            print(self.pending_roi_keys)
            if not(bool(self.p0_coords)):
                # print(self.p0_coords)
                self.select_first_point(click_coords)
            elif len(self.pending_roi_keys) == 1:
                self.select_scnd_and_stash_roi(
                    click_coords)
                self.pass_controll_back_to_the_script()                
            else:
                self.select_scnd_and_stash_roi(
                    click_coords)


    def pass_controll_back_to_the_script(self):
        self.destroy()
        # self.master.destroy()

    def select_first_point(self, click_coords):
        self.p0_coords = click_coords
                
    def current_roi_key(self):
        if bool(self.pending_roi_keys):
            return self.pending_roi_keys[-1]
        else:
            return None

    def select_scnd_and_stash_roi(self, click_coords):
        self.roi_data[
            self.current_roi_key()] = {
                'p0': self.p0_coords,
                'p1': click_coords}
        self.p0_coords = None
        self.pending_roi_keys.pop()
        # print(self.roi_data[
        #     self.current_roi_key()])
        self.labels['cw_roi'].config(
            text=f'selectează regiune:\n{self.current_roi_key()}')
        # print(self.roi_data)


    def is_second_roi_selection(self):
        return len(self.sel_pnts) == 1

class OCR(ImProcEditor):
    def __init__(self, state_data, gui_data):
        super().__init__(state_data, gui_data)
        self.perform_ocr()
        self.plot_ocr_bounding_boxes()
        # self.bind('<space>', self.apply_ocr),
        
    def perform_ocr(self):
        self.ocr_data = dict_from_im(
            self.proc_im) # !!! or self.og_im, idk which's better

    def plot_ocr_bounding_boxes(self):
        for box_data in self.ocr_data:
            self.main_canvas.create_rectangle(
                *self.scale_by_main_canv_sf(
                    diag_crnr_coords__(
                        box_data)),
                outline=MAGENTA, width=LINE_WIDTH)

        
        
            

class OCR__obsolete(ImProcEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
        self.perform_ocr()
        self.bind('<space>', self.apply_ocr),

    def perform_ocr(self):
        self.ocr_data = dict_from_im(
            self.proc_im) # !!! or self.og_im, idk which's better
        for box_data in self.ocr_data:
            x0, y0 = box_data['left'], box_data['top']
            w, h = box_data['width'], box_data['height']
            x1, y1 = x0 + w, y0 + h
            self.main_canvas.create_rectangle(
                *map(lambda n: n * self.main_canvas.scale_factor,
                     [x0, y0, x1, y1]),
                outline=MAGENTA, width=LINE_WIDTH)

    def next_stage(self):
        self.pipeline_data['improc_params'][self.__class__.__name__] = self.param_data
        progress_data = {'ocr_data': self.ocr_data}
        self.pipeline_data['improc_progress'][self.__class__.__name__] = progress_data
        parent = self.master

        # VALIDATION --------------------------------------------------------------
        ocr_validation_stack = self.ocr_data.copy()
        ocr_validation_stack.reverse()
        
        # -------------------------------------------------------------------------
        
        # -------------------------------------------------------------------------
        # NEXT CONSTRUCTOR IN PIPELINE IS DELAYED FOR AFTER THE VALIDATION STAGES -
        # next_frame_constructor = self.pipeline_data['relative_constructors'][
        #     self.__class__.__name__]['following']
        # frame = next_frame_constructor(
        #     parent, self.proc_im.copy(), self.pipeline_data)
        # -------------------------------------------------------------------------

        self.destroy()
        frame.grid(
            row=0, column=0, rowspan=FRAME_ROWSPAN)

    def apply_ocr(self):
        self.param_data = {} # !!! This should contain the tesseract ocr parameters
        self.next_stage()


class OCRValidation(tk.Frame):
    def __init__(self, master, ocr_box_im, ocr_box_data, pipeline_data):
        super().__init__(master)
        self.im = ocr_box_im
        self.box_data = ocr_box_data
        self.pipeline_data = pipeline_data

        self.canvas = ValidationCanvas(self, self.im)
        self.canvas.grid(row=0, column=0)
        

# class OCRValidation(ImProcEditor):
#     def __init__(self, master, cvim, pipeline_data):
#         super().__init__(master, cvim, pipeline_data)
#         self.load_ocr_data()
        
#     def load_ocr_data(self):
#         previous_frame_constructor = self.pipeline_data['relative_constructors'][
#             self.__class__.__name__]['previous']
#         self.ocr_data = pipeline_data['improc_progress'][
#             previous_frame_constructor.__name__]['ocr_data']

#     def construct_validation_pipeline(self):
#         for i in range(len(self.ocr_data)):
#             if i < len(ocr_data) - 1:
#                 following_constructor = OCRValidationBox
#                 following_ocr_box_data = 

#         # self.pipeline = list(map(
#         #     lambda box_data: OCRValidationBox(self, self.og_im, pipeline_data, box_data),
#         #     self.ocr_data))

### PIPELINE -------------------------------------------------------------------------

## Section Obsoleted

# # def stage_data_from_constructor_and_assoc_data(constructor, assoc_data, pipeline_index):
# #     return {
# #         'constructor': constructor,
# #         'assoc_data': assoc_data,
# #     }

# # def pipeline_data__(constructor_list, stage_associated_data,
# # ):

# #     pipeline_data = list(
# #         map(stage_data_from_constructor_and_assoc_data
# #             zip(constructor_list, # frame constructor
# #                 stage_associated_data, # associated data (if any)
# #                 # next constructor
# #                 # previous constructor

    
    
# #     pipeline_data['improc_params'] = {}
# #     pipeline_data['improc_progress'] = {}
# #     return pipeline_data

# def pipeline_data__from__constructor_list(constructor_list):
#     pipeline_data = {}
#     pipeline_data['sequence'] = list(
#         map(lambda x: x.__name__,
#             constructor_list))
#     pipeline_data['relative_constructors'] = {}
#     for i in range(len(constructor_list)):
#         constructor = constructor_list[i]

#         if i < len(constructor_list) - 1:
#             following_constructor = constructor_list[i+1]
#         else:
#             following_constructor = constructor_list[0]

#         if i == 0:
#             previous_constructor = constructor_list[-1]
#         else:
#             previous_constructor = constructor_list[i-1]

#         pipeline_data['relative_constructors'][constructor.__name__] = {
#             'following': following_constructor,
#             'previous': previous_constructor,
#         }

#     pipeline_data['improc_params'] = {}
#     pipeline_data['improc_progress'] = {}
#     return pipeline_data


# def run_pipeline(constructor_list, im):
#     root = tk.Tk()
#     root.bind('<Control-q>', lambda event: root.destroy())

#     pipeline_data = pipeline_data__from__constructor_list(
#         constructor_list)

#     improc = constructor_list[0](
#         root, im, pipeline_data)
#     improc.grid(row=0, column=0, rowspan=FRAME_ROWSPAN)

#     # root.mainloop()

# ### -----------------------------------------------------------------------------------

# if __name__ == '__main__':
#     print('frames.py')

#     import os
#     import cv2 as cv

#     #clean_invoice_fname = '../test_data/609d5d3c4d120e370de52b70_invoice-lp-light-border.png'
#     samples_path = '../test_data/samples_2021'
#     samples_fnames = list(
#         map(lambda fnm: f'{samples_path}/{fnm}',
#             os.listdir('../test_data/samples_2021')))
#     im = cv.imread(samples_fnames[2])

#     ## ----------------------------------------------------------------------

#     # improc_pipeline_sequence = [
#     #     WarpingEditor,
#     #     OrthogonalRotationEditor,
#     #     FineRotationEditor,
#     #     RescaleEditor,
#     #     CropEditor,
#     #     DenoiseEditor,
#     #     ThresholdEditor,
#     #     DilateErodeEditor,
#     #     OCR,
#     # ]

#     # run_pipeline(
#     #     improc_pipeline_sequence, im)

#     ## ----------------------------------------------------------------------

#     im = cv.imread('improc_images/DilateErodeEditor.png')

#     pipeline_sequence = [OCR]
#     run_pipeline(pipeline_sequence, im)
    

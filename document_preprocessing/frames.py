import tkinter as tk
# import pytesseract as tesseract
# import os
# import numpy as np
import cv2 as cv

from canvas import canvas_from_im, DocumentCanvas
from im import imwarp, rgb2hex, display_cv, rotate, orthogonal_rotate, rotate_without_clipping, display_cv, denoise, dilate_erode, threshold, crop, im_rescale #, rotate_by_90deg
from constants import *

editor_data = {
    'warp': {
        'labels': {
            'hover': {'initial_text': 'hover'},
            'click': {'initial_text': 'click'},
        },
        'label_order': ['hover', 'click']
    },
    'rotation': {},
    'rescale': {},
    'denoise': {},
    'threshold': {},
    'dilate_erode': {},
}

def augment_data(dat1, dat2):
    for key in dat2.keys():
        dat1[key] = dat2[key]

### GENERAL EDITOR --------------------------------------------------------------------

class ImProcEditor(tk.Frame):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master)
        self.pipeline_data = pipeline_data
        self.grid(row=0, column=0, rowspan=FRAME_ROWSPAN)
        self.og_im = cvim
        self.proc_im = cvim.copy()
        
        self.labels = {}
        self.add_label(self.__class__.__name__)
        self.add_label('hover')
        self.add_label('click')

        self.main_canvas = None
        self.main_canvas_bindings = [
            ('<Motion>', lambda e: self.mouse_label_event_config(e, 'hover')),
            ('<Button-1>', lambda e: self.mouse_label_event_config(e, 'click')),
            ('<Escape>', self.cancel), ('<Control-c>', self.cancel),
            ('<space>', self.skip)
        ]
        self.install_main_canvas()

    def skip(self, event):
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
            row=0, column=0, rowspan=20)

    def next_stage(self):
        parent = self.master
        next_frame_constructor = self.pipeline_data['relative_constructors'][
            self.__class__.__name__]['following']
        frame = next_frame_constructor(
            parent, self.proc_im.copy(), self.pipeline_data)
        self.destroy()
        frame.grid(
            row=0, column=0, rowspan=20)

### WARPING ----------------------------------------------------------------------------      

class WarpingEditor(ImProcEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
        self.main_canvas.bind('<Motion>', self.hover)
        self.main_canvas.bind('<Button-1>', self.click)
        self.main_canvas.install_crosshairs()

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
        if len(self.selection_points) > 0 and len(self.selection_points) < 4:
            prev_x, prev_y = self.selection_points[-1]
            prev_x *= self.main_canvas.scale_factor
            prev_y *= self.main_canvas.scale_factor
            self.main_canvas.coords(
                self.floating_selection_line, prev_x, prev_y, canv_x, canv_y)
        if len(self.selection_points) == 3:
            init_x, init_y = self.selection_points[0]
            init_x *= self.main_canvas.scale_factor
            init_y *= self.main_canvas.scale_factor
            self.main_canvas.coords(
                self.closing_selection_line, init_x, init_y, canv_x, canv_y)

    def click(self, event):
        self.mouse_label_event_config(event, 'click')
        im_x = int(round(event.x / self.main_canvas.scale_factor))
        im_y = int(round(event.y / self.main_canvas.scale_factor))
        self.selection_points.append(
            (im_x, im_y))
        if len(self.selection_points) > 1 and len(self.selection_points) < 4:
            x0, y0 = self.selection_points[-2]
            self.laid_selection_lines.append(
                self.main_canvas.create_line(
                    *map(lambda n: n * self.main_canvas.scale_factor,
                         [x0, y0, im_x, im_y]),
                    fill=MAGENTA, width=2))
        if len(self.selection_points) == 4:
            self.proc_im = imwarp(
                self.proc_im, self.selection_points)
            self.next_stage()

        
### ORTHOGONAL ROTATION --------------------------------------------------------------

class OrthogonalRotationEditor(ImProcEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
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
        self.next_stage()

### FINE ROTATION --------------------------------------------------------------

class FineRotationEditor(ImProcEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
        self.add_label('angle')
        self.fine_rotation_angle = 0 # degrees
        self.augment_main_canvas_bindings([
            ('<space>', self.apply_fine_rotation),
            ('<Shift-Button-4>', lambda _: self.rotate_fine(1)), # trig direc. (counterclock.)
            ('<Shift-Button-5>', lambda _: self.rotate_fine(-1)), # trig direction (clockwise)
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
        self.next_stage()

### RESCALE -----------------------------------------------------------------------

class RescaleEditor(ImProcEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
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
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
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
    def __init__(self, master, cvim, pipeline_data, filter_function,
                 allow_negative_kernel_size=False):
        super().__init__(master, cvim, pipeline_data)
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
        self.next_stage()

### DENOISE -------------------------------------------------------------------------

class DenoiseEditor(FilteringEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data, denoise)

### DILATE ERODE ----------------------------------------------------------------

class DilateErodeEditor(FilteringEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data, dilate_erode,
                         allow_negative_kernel_size=True)

### THRESHOLD -------------------------------------------------------------------

class ThresholdEditor(ImProcEditor):
    def __init__(self, master, cvim, pipeline_data):
        super().__init__(master, cvim, pipeline_data)
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
        self.next_stage()

### PIPELINE -------------------------------------------------------------------------

def run_pipeline(constructor_list):
    root = tk.Tk()
    root.bind('<Control-q>', lambda event: root.destroy())

    pipeline_data = {}
    pipeline_data['sequence'] = list(
        map(lambda x: x.__name__,
            constructor_list))
    pipeline_data['relative_constructors'] = {}
    for i in range(len(constructor_list)):
        constructor = constructor_list[i]

        if i < len(improc_pipeline) - 1:
            following_constructor = constructor_list[i+1]
        else:
            following_constructor = constructor_list[0]

        if i == 0:
            previous_constructor = constructor_list[-1]
        else:
            previous_constructor = constructor_list[i-1]

        pipeline_data['relative_constructors'][constructor.__name__] = {
            'following': following_constructor,
            'previous': previous_constructor,
        }

    improc = constructor_list[0](
        root, im, pipeline_data)
    improc.grid(row=0, column=0, rowspan=FRAME_ROWSPAN)

    root.mainloop()

### -----------------------------------------------------------------------------------

if __name__ == '__main__':
    print('frames.py')

    import os
    import cv2 as cv

    #clean_invoice_fname = '../test_data/609d5d3c4d120e370de52b70_invoice-lp-light-border.png'
    samples_path = '../test_data/samples_2021'
    samples_fnames = list(
        map(lambda fnm: f'{samples_path}/{fnm}',
            os.listdir('../test_data/samples_2021')))
    im = cv.imread(samples_fnames[2])

    ## ----------------------------------------------------------------------

    improc_pipeline = [
        WarpingEditor,
        OrthogonalRotationEditor,
        FineRotationEditor,
        RescaleEditor,
        CropEditor,
        DenoiseEditor,
        ThresholdEditor,
        DilateErodeEditor,
    ]

    run_pipeline(improc_pipeline)

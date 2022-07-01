import colorsys
import numpy as np

def rgb2hex(r,g,b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'

def hue2hex(hue):
    return rgb2hex(
        *map(lambda x: int(x * 255),
             colorsys.hsv_to_rgb(
                 hue, 0.85, 0.75)))

hues = list(np.arange(0, 1, 0.15) + 0.1)
ocr_box_level_colors = list(map(hue2hex, hues))



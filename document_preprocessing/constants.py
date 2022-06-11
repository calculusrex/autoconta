
# WHITE = '#ffffff'
# BLACK = '#000000'
# GRAY = '#363636'

def rgb2hex(r,g,b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'

PIPELINE_DATA_FOLDER = 'pipeline_data'

FRAME_ROWSPAN = 25
LINE_WIDTH = 2
TARGET_LETTER_HEIGHT = 32

WHITE = rgb2hex(255, 255, 255)
BLACK = rgb2hex(0, 0, 0)
GRAY = rgb2hex(64, 64, 64)
MAGENTA = '#FF00FF'

FOREGROUND = GRAY
BACKGROUND = WHITE

AUTO = rgb2hex(12, 198, 128)
HUMAN = rgb2hex(12, 128, 198)
ATTENTION = rgb2hex(198, 12, 128)

TITLE = 'Tesseract + OpenCV'

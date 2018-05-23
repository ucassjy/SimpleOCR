import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
  FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
  FONT = ImageFont.load_default()

def _draw_single_box(image, left, bottom, right, top, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    # (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([left, bottom, right,
                top, left], width=thickness, fill=color)
    # text_bottom = bottom[1]
    # Reverse list and print from bottom to top.
    # text_width, text_height = font.getsize(display_str)
    # margin = np.ceil(0.05 * text_height)
    # draw.rectangle(
        # [(left[0], text_bottom - text_height - 2 * margin), (left[0] + text_width, text_bottom)],
        # fill=color)
    # draw.text(
        # (left[0] + margin, text_bottom - text_height - margin),
        # display_str, fill='black', font=font)

    return image

def draw_bounding_boxes(image, gt_boxes, im_info):
    num_boxes = gt_boxes.shape[0]
    gt_boxes_new = gt_boxes.copy()
    gt_boxes_new[:,:5] = np.round(gt_boxes_new[:,:5].copy() * im_info[2])
    disp_image = Image.fromarray(np.uint8(image[0]))

    for i in range(num_boxes):
        this_class = 1
        [x,y,h,w,theta] = gt_boxes[i,:]
        cos_abs = np.abs(np.cos(theta))
        sin_abs = np.abs(np.sin(theta))
        x_min = x - (h * sin_abs + w * cos_abs) / 2.0
        x_max = x + (h * sin_abs + w * cos_abs) / 2.0
        y_min = y - (w * sin_abs + h * cos_abs) / 2.0
        y_max = y + (w * sin_abs + h * cos_abs) / 2.0
        if theta < 0 or theta > 90:
            left   = (x_min, y_min + w * sin_abs)
            bottom = (x_min + w * cos_abs, y_min)
            right  = (x_max, y_min + h * cos_abs)
            top    = (x_min + h * sin_abs, y_max)
        else:
            left   = (x_min, y_min + h * cos_abs)
            bottom = (x_min + h * sin_abs, y_min)
            right  = (x_max, y_min + w * sin_abs)
            top    = (x_min + w * cos_abs, y_max)            

        disp_image = _draw_single_box(disp_image,
                                           left,
                                           bottom,
                                           right,
                                           top,
                                'N%02d-C%02d' % (i, this_class),
                                FONT,
                                color=STANDARD_COLORS[this_class % NUM_COLORS])

    image[0, :] = np.array(disp_image)
    return image

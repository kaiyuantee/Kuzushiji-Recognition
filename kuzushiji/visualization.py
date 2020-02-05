"""
Modified visualization module, based on
https://www.kaggle.com/anokas/kuzushiji-visualisation
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .data_utils import DATA_ROOT, TRAIN_ROOT, UNI_MAP

BOX_COLOR = (255, 0, 0)


def load_font(fontsize: int):
    """ Download this font for Kuzushiji text visualization
        wget -q --show-progress \
            https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip
        unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf \
            > Datasets/NotoSansCJKjp-Regular.otf
        rm NotoSansCJKjp-hinted.zip
    """
    return ImageFont.truetype(
        str(DATA_ROOT / 'NotoSansCJKjp-Regular.otf'), size=fontsize, encoding='utf-8')


def image_visualization(img, img_id, df, unimap=UNI_MAP, fontsize=50):
    """
    This function takes in a filename of an image, and the labels in the string format given in a submission csv, and returns an image with the characters and predictions annotated.
    Copied and slightly modified from:
    https://www.kaggle.com/anokas/kuzushiji-visualisation
    """
    # Convert annotation string to array
#     labels_split = labels.split(' ')
#     if len(labels_split) < 3:
#       return img
#     labels = np.array(labels_split).reshape(-1, 3)
    labels = np.array(df.loc[df.image_id == img_id, 'labels'].iloc[0].split(" ")).reshape(-1, 3)
    # Read image
    img = cv2.imread(img, 1)
    imsource = Image.fromarray(np.array(img)).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)
    font=load_font(fontsize=fontsize)
    for codepoint, x, y in labels:
        x, y = int(x), int(y)
        char = unimap.get(codepoint,codepoint)  # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x-10, y-10, x+10, y+10), fill=(255, 0, 0, 255))
        char_draw.text((x+25, y-fontsize*(3/4)), char, fill=(255, 0, 0, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB")  # Remove alpha for saving in jpg format.
    return np.asarray(imsource)


def visualize_box(image: np.ndarray, bbox, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = \
        int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)


def visualize_boxes(image: np.ndarray, boxes, **kwargs):
    image = image.copy()
    for idx, bbox in enumerate(boxes):
        visualize_box(image, bbox, **kwargs)
    return image

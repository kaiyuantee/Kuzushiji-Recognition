import pandas as pd
import numpy as np
import torch
import torch.utils.data
import random
import albumentations as albu
from albumentations.pytorch import ToTensor
from pathlib import Path
from typing import Callable
from ..data_utils import get_image_path, read_image, get_target_boxes_labels  # from other directory


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame, aug: Callable, root: Path, skip_empty: bool):
        self.df = df
        self.aug = aug
        self.skip_empty = skip_empty
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img = read_image(get_image_path(item, self.root))
        h, w, _ = img.shape
        bboxes, labels = get_target_boxes_labels(item)
        bboxes[:, 2] = (np.minimum(bboxes[:, 0] + bboxes[:, 2], w) - bboxes[:, 0])
        bboxes[:, 3] = (np.minimum(bboxes[:, 1] + bboxes[:, 3], h) - bboxes[:, 1])
        xy = {'image': img,
              'bboxes': bboxes,
              'labels': np.ones_like(labels, dtype=np.long)
              }
        xy = self.aug(**xy)
        if not xy['bboxes'] and self.skip_empty:
            return self[random.randint(0, len(self.df) - 1)]
        img = xy['image']
        boxes = torch.tensor(xy['bboxes']).reshape((len(xy['bboxes']), 4))

        # conversion for pytorch detection format
        boxes[:, 2] += boxes[:, 0]  # x+w
        boxes[:, 3] += boxes[:, 1]  # y+h
        target = {'boxes': boxes,
                  'labels': torch.tensor(xy['labels'], dtype=torch.long),
                  'idx': torch.tensor(idx)}
        return img, target


def augmentation(train: bool) -> Callable:

    initial_size = 2048
    crop_min_max_height = (400, 533)
    crop_width = 512
    crop_height = 384
    if train:
        aug = [albu.LongestMaxSize(max_size=initial_size),
               albu.RandomSizedCrop(min_max_height=crop_min_max_height,
                                    width=crop_width,
                                    height=crop_height,
                                    w2h_ratio=crop_width/crop_height),
               albu.HueSaturationValue(hue_shift_limit=7,
                                       sat_shift_limit=10,
                                       val_shift_limit=10),
               albu.RandomBrightnessContrast(),
               albu.RandomGamma()]
    else:
        test_size = int(initial_size * crop_height / np.mean(crop_min_max_height))
        print('Test image max sizes is {} pixels'.format(test_size))
        # for tta probably
        aug = [albu.LongestMaxSize(max_size=test_size)]

    aug.extend([ToTensor()])
    return albu.Compose(aug, bbox_params={'format': 'coco',
                                          'min_area': 0,
                                          'min_visibility': 0.5,
                                          'label_fields': ['labels']})

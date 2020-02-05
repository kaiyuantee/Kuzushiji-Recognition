import numpy as np
import argparse
from tqdm import tqdm
from .data_utils import read_image, TRAIN_ROOT, get_image_np_path  #


def main():
    parser = argparse.ArgumentParser()
    _ = parser.parse.args()
    paths = list(TRAIN_ROOT.glob('*.jpg'))
    for path in tqdm(paths):
        np_path = get_image_np_path(path)
        if not np.path.exists():
            image = read_image(path)
            np.save(np.path, image)


if __name__ == '__main__':
    main()
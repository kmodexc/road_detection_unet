from __future__ import print_function
from pandas import array
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random
from PIL import Image

Nothing = [255,0,0]
Road = [255,0,255]

def adjustLabel(img):
    img = np.array(img)
    outimage = np.zeros((img.shape[0],img.shape[1],1))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            color = img[x,y]
            if (color == Road).all():
                outimage[x,y] = [1.0]
            elif (color == Nothing).all():
                outimage[x,y] = [0.0]
            else:
                outimage[x,y] = [0.0]
    return outimage

class RoadDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        img_paths = list(img_paths)
        self.input_img_paths = [x[0] for x in img_paths]
        self.target_img_paths = [x[1] for x in img_paths]
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size)
            img = adjustLabel(img)
            y[j] = img
        return x, y


def get_image_paths(_type='training'):
    input_dir = f"data/data_road/{_type}/image_2"
    target_dir = f"data/data_road/{_type}/gt_image_2"

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    print("Number of samples:", len(input_img_paths))

    return zip(input_img_paths, target_img_paths)

def get_train_val_split(paths):
    # Split our img paths into a training and a validation set
    paths = list(paths)
    val_samples = 20
    random.Random(1337).shuffle(paths)
    inp_p = [x[0] for x in paths]
    lab_p = [x[1] for x in paths]
    train_input_img_paths = inp_p[:-val_samples]
    train_target_img_paths = lab_p[:-val_samples]
    val_input_img_paths = inp_p[-val_samples:]
    val_target_img_paths = lab_p[-val_samples:]
    return zip(train_input_img_paths,train_target_img_paths), zip(val_input_img_paths,val_target_img_paths)


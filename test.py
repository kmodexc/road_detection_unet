from model import *
from data import *
import tensorflow as tf
import random
from PIL import Image

def save_gs_img(data,path):
    assert data.shape == (256,256,1)
    assert data.max() <= 1.0
    assert data.min() >= 0.0
    conv_arr = np.array([[p[0]*255 for p in row] for row in data],dtype=np.uint8)
    assert conv_arr.shape == (256,256)
    im = Image.fromarray(conv_arr)
    im = im.convert('RGB')
    im.save(path)

paths = get_image_paths()

callbacks = [
    keras.callbacks.ModelCheckpoint("road_detection.h5", save_best_only=True)
]

train_gen = RoadDataset(1,(256,256),paths)

model = unet()

save_gs_img(train_gen[0][1][0],"label.png")

pdim = model.predict(train_gen[0][0])

save_gs_img(pdim[0],"output.png")




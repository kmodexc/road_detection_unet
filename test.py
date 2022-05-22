from model import *
from data import *
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint

paths = get_image_paths()

callbacks = [
    ModelCheckpoint("road_detection.h5", save_best_only=True)
]

train_gen = RoadDataset(1,(256,256),paths)

checkpoint_file = "road_detection.h5"

if os.path.isfile(checkpoint_file):
    print("Load from checkpoint")
    model = unet(checkpoint_file)
else:
    model = unet()

imind = 10
save_rgb_img(train_gen[imind][0][0],"input.png")
save_gs_img(train_gen[imind][1][0],"label.png")
pdim = model.predict(train_gen[imind][0])
save_gs_img(pdim[0],"output.png")
cut_rgb_img(train_gen[imind][0][0],pdim[0],"masked_imput.png")

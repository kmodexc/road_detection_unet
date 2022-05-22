from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from model import *
from data import *

paths = get_image_paths()
trainpaths, valpaths = get_train_val_split(paths)

train_gen = RoadDataset(1, (256,256), trainpaths)
val_gen = RoadDataset(1, (256,256), valpaths)

checkpoint_file = "road_detection.h5"

callbacks = [
    ModelCheckpoint(checkpoint_file, save_best_only=True)
]

if os.path.isfile(checkpoint_file):
    model = unet(checkpoint_file)
else:
    model = unet()

model.fit(train_gen,epochs=20,validation_data=val_gen,callbacks=callbacks)

imind = 6
save_rgb_img(train_gen[imind][0][0],"input.png")
save_gs_img(train_gen[imind][1][0],"label.png")
pdim = model.predict(train_gen[imind][0])
save_gs_img(pdim[0],"output.png")
cut_rgb_img(train_gen[imind][0][0],pdim[0],"masked_imput.png")
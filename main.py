from model import *
from data import *

paths = get_image_paths()
trainpaths, valpaths = get_train_val_split(paths)

train_gen = RoadDataset(1, (256,256), trainpaths)
val_gen = RoadDataset(1, (256,256), valpaths)

callbacks = [
    keras.callbacks.ModelCheckpoint("road_detection.h5", save_best_only=True)
]

model = unet("road_detection.h5")
model.fit(train_gen,epochs=1,validation_data=val_gen,callbacks=callbacks)
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from model import *
from data import *

paths = get_image_paths()
trainpaths, valpaths = get_train_val_split(paths)

train_gen = RoadDataset(1, (256,256), trainpaths)
val_gen = RoadDataset(1, (256,256), valpaths)

checkpoint_file = None

callbacks = [
    ModelCheckpoint(checkpoint_file, save_best_only=True)
]

if checkpoint_file is not None and os.path.isfile(checkpoint_file):
    model = unet(checkpoint_file)
else:
    model = unet()

print(len(train_gen[0]))
print(train_gen[0][0].shape)
print(train_gen[0][1].shape)

print(model.summary())

model.fit(train_gen,epochs=20,validation_data=val_gen,callbacks=callbacks)
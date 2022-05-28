from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from model import *
from data import *

train_gen, val_gen = get_ds_city()

checkpoint_file = "road_unet_v3.h5"

model = unet(checkpoint_file)

callbacks = [ModelCheckpoint(checkpoint_file ,save_best_only=True)]
model.fit(train_gen,epochs=50,validation_data=val_gen,callbacks=callbacks)
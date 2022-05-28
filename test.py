from model import *
from data import *

train_gen, val_gen = get_ds_combined()

checkpoint_file = "road_unet_v3.h5"
model = unet(checkpoint_file)

print("avg=",eval(model,val_gen,n=50))

predict_and_save(model,train_gen,6)

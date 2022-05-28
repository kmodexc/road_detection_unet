import tensorflow.experimental.tensorrt as trt
from model import *
from data import *

checkpoint_file = "road_unet_v3.h5"
model = unet(checkpoint_file)
export_dir = "export"
model.save(export_dir)

converter = trt.Converter(input_saved_model_dir=export_dir)
converter.convert()
tensorrt_dir = "tensorrt"
converter.save(tensorrt_dir)
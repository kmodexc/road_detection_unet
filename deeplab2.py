import os
import tensorflow as tf
from data import RoadDataset, get_image_paths,save_rgb_img,save_gs_img,cut_rgb_img
import numpy as np

MODEL_NAME = 'max_deeplab_s_backbone_os16_axial_deeplab_cityscapes_trainfine_saved_model'
model_dir = "dlmodel"
model = tf.saved_model.load(os.path.join(model_dir, MODEL_NAME))

paths = get_image_paths()
train_gen = RoadDataset(1,(512,512),paths)

im_ind = 5

save_rgb_img(train_gen[im_ind][0][0],"input.png")

out = model(tf.cast(train_gen[im_ind][0][0],tf.uint8))
sempred = out['semantic_pred'][0]
sempred = np.array([[[1.0] if x==0 else [0.0] for x in row] for row in sempred])
save_gs_img(sempred,'output.png')
cut_rgb_img(train_gen[im_ind][0][0],sempred,"maked_input.png")
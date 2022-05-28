from __future__ import print_function
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import load_img,normalize
from tensorflow.keras.metrics import MeanIoU
from keras.metrics import MeanIoU
import numpy as np 
import os
import numpy as np
import random
from PIL import Image
import cv2 as cv

CERTANTY = 0.7

def adjustLabel(img,road):
    img = np.array(img)
    outimage = np.zeros((img.shape[0],img.shape[1],1))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            color = img[x,y]
            if (color == road).all():
                outimage[x,y] = [1.0]
            else:
                outimage[x,y] = [0.0]
    return outimage

class RoadDataset(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self, batch_size, img_size, img_paths,road):
        assert batch_size > 0
        self.road = road
        self.batch_size = batch_size
        self.img_size = img_size
        img_paths = list(img_paths)
        assert len(img_paths) > 0
        self.input_img_paths = [x[0] for x in img_paths]
        self.target_img_paths = [x[1] for x in img_paths]
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        assert i >= 0
        assert i < len(self.input_img_paths)
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        assert len(batch_input_img_paths) > 0
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            img = np.array(img)
            norm_img = np.zeros_like(img)
            img = cv.normalize(img,  norm_img, 0, 255, cv.NORM_MINMAX)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size)
            img = adjustLabel(img,self.road)
            y[j] = img
        return x, y

def get_ds_combined(inp_res = (512,512)):
    road = [1,1,1]
    paths = get_image_paths_combined()
    trainpaths, valpaths = get_train_val_split(paths)
    train_gen = RoadDataset(1, inp_res, trainpaths, road)
    val_gen = RoadDataset(1, inp_res, valpaths, road)
    return train_gen, val_gen

def get_ds_road(inp_res = (512,512)):
    road = [255,0,255]
    paths = get_image_paths_data_road()
    trainpaths, valpaths = get_train_val_split(paths)
    train_gen = RoadDataset(1, inp_res, trainpaths, road)
    val_gen = RoadDataset(1, inp_res, valpaths, road)
    return train_gen, val_gen

def get_ds_city(inp_res = (512,512),fraction=0.1):
    road = [128,64,128]
    cities = [
        'aachen','bochum','bremen','cologne','darmstadt','dusseldorf','erfurt',
        'hamburg','hanover','jena','krefeld','monchengladbach','strasbourg',
        'tubingen','ulm','weimar','zurich'
    ]
    paths = []
    for city in cities:
        paths += list(get_image_paths_cityscapes(city))
    trainpaths, valpaths = get_train_val_split(paths,fraction=fraction)
    train_gen = RoadDataset(1, inp_res, trainpaths, road)
    val_gen = RoadDataset(1, inp_res, valpaths, road)
    return train_gen, val_gen

def get_image_paths_cityscapes(city="aachen"):
    input_dir = f"data/leftImg8bit/train/{city}"
    target_dir = f"data/gtFine/train/{city}"
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
            if fname.endswith("color.png") and not fname.startswith(".")
        ]
    )
    print("Number of samples:", len(input_img_paths))
    return zip(input_img_paths, target_img_paths)

def get_image_paths_combined():
    input_dir = f"data/combined_dataset/JPEGImages"
    target_dir = f"data/combined_dataset/SegmentationClass"
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png") and fname.startswith("race")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".") and fname.startswith("race")
        ]
    )
    print("Number of samples:", len(input_img_paths))
    return zip(input_img_paths, target_img_paths)

def get_image_paths_data_road(_type='training'):
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

def get_train_val_split(paths,fraction=1.0):
    # Split our img paths into a training and a validation set
    paths = list(paths)
    random.Random().shuffle(paths)
    npaths = int(fraction*len(paths))
    assert fraction <= 1.0
    assert npaths <= len(paths)
    inp_p = [x[0] for x in paths]
    lab_p = [x[1] for x in paths]
    assert len(inp_p) == len(lab_p)
    inp_p = inp_p[-npaths:]
    lab_p = lab_p[-npaths:]
    assert len(inp_p) == len(lab_p)
    val_samples = min(int(len(inp_p)/10),30)
    train_input_img_paths = inp_p[:-val_samples]
    train_target_img_paths = lab_p[:-val_samples]
    val_input_img_paths = inp_p[-val_samples:]
    val_target_img_paths = lab_p[-val_samples:]
    #assert len(paths) == (len(val_input_img_paths) + len(train_input_img_paths))
    return zip(train_input_img_paths,train_target_img_paths), zip(val_input_img_paths,val_target_img_paths)

def save_gs_img(data,path=None):
    assert len(data.shape) == 3
    assert data.shape[-1] == 1
    assert data.shape[0] == data.shape[1]
    assert data.max() <= 1.0
    assert data.min() >= 0.0
    conv_arr = np.array([[p[0]*255 for p in row] for row in data],dtype=np.uint8)
    im = Image.fromarray(conv_arr)
    im = im.convert('RGB')
    if path is None:
        im.show()
    else:
        im.save(path)

def save_rgb_img(data,path=None):
    assert len(data.shape) == 3
    assert data.shape[-1] == 3
    assert data.shape[0] == data.shape[1]
    im = Image.fromarray(data.astype(np.uint8))
    if path is None:
        im.show()
    else:
        im.save(path)

def cut_rgb_img(data,mask,path=None):
    assert len(data.shape) == 3
    assert data.shape[-1] == 3
    assert data.shape[0] == data.shape[1]
    assert len(mask.shape) == 3
    assert mask.shape[-1] == 1
    assert mask.shape[0] == data.shape[1]
    outimage = np.zeros_like(data)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] > CERTANTY:
                outimage[x,y] = data[x,y]
    im = Image.fromarray(outimage.astype(np.uint8))
    if path is None:
        im.show()
    else:
        im.save(path)

def get_mask(mask):
    assert mask.shape[-1] == 1
    outimage = np.zeros_like(mask)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] > CERTANTY:
                outimage[x,y] = 1.0
    return outimage

def get_miou(output,trouth):
    m = MeanIoU(num_classes=2)
    m.update_state(trouth,get_mask(output))
    return m.result().numpy()

def eval(model,dataset,n = 20):
    avg = 0
    rv = int(random.random()*len(dataset))
    for i in range(n):
        pdim = model.predict(dataset[(rv+i) % len(dataset)][0])
        avg += get_miou(pdim[0],dataset[(rv+i) % len(dataset)][1][0])
    return avg/n

def predict_and_save(model,dataset,i):
    save_rgb_img(dataset[i][0][0],"input.png")
    save_gs_img(dataset[i][1][0],"label.png")
    pdim = model.predict(dataset[i][0])
    save_gs_img(pdim[0],"output.png")
    cut_rgb_img(dataset[i][0][0],pdim[0],"masked_input.png")

def get_colors(nparr):
    colors = {}
    for row in nparr:
        for pix in row:
            spix = f"{pix[0]},{pix[1]},{pix[2]}"
            if spix not in colors:
                colors[spix] = 0
            else:
                colors[spix] += 1
    return colors

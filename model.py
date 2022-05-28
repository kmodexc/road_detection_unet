from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import * 
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.applications import MobileNetV3Small
from keras.models import *
from keras.layers import *
from keras.optimizers import * 
from keras.losses import *
from keras.metrics import Accuracy
from keras.applications import MobileNetV3Small
import pix2pix
import os

def unet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    
    x = BatchNormalization()(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(conv2)

    x = BatchNormalization()(x)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    x = MaxPooling2D(pool_size=(2, 2))(conv3)

    x = BatchNormalization()(x)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    x = MaxPooling2D(pool_size=(2, 2))(drop4)

    x = BatchNormalization()(x)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    x = Dropout(0.5)(conv5)

    x = BatchNormalization()(x)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = BatchNormalization()(conv9)
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(
        optimizer = Adam(learning_rate = 1e-7), 
        loss = BinaryCrossentropy(), 
        metrics = ['accuracy'])
    
    #model.summary()

    if(os.path.isfile(pretrained_weights) and pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def mnv3_pixtopix(pretrained_weights=None) -> Model:

    input_shape=[512, 512, 3]

    base_model = MobileNetV3Small(input_shape,include_top=False)

    # Use the activations of these layers
    layer_names = [
        're_lu_2',
        're_lu_2',
        're_lu_20',
        're_lu_2',
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=2, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    x = Conv2D(1, 1, activation = 'sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer = Adam(learning_rate = 1e-3), 
        loss=BinaryCrossentropy(),
        metrics=['accuracy'])
    
    #model.summary()

    if(pretrained_weights and os.path.isfile(pretrained_weights)):
        model.load_weights(pretrained_weights)

    return model